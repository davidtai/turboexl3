#!/usr/bin/env python3
"""Check that turboexl3's Metal lives only in src/kernels/**.metal and that a change to it is only what was intended.

usage: python3 tools/check_texts.py ORIG_SRC [NEW_SRC] [-v]

  ORIG_SRC  the sources before the change (a checkout's src/; for the move out of Zig, the tree with the Metal still
            inside it, e.g. afb52be: `git worktree add /tmp/t3-orig afb52be`, then /tmp/t3-orig/src)
  NEW_SRC   the sources after it (default: this repository's src/)
  -v        every kernel and text const, one line each

All on the host, no Zig needed. Both trees are tokenized and their texts evaluated: string literals, `@embedFile`, `++`,
consts, `module.const`, `if` (a comptime condition: only the taken branch, as Zig analyzes it), `&.{ ... }`, labeled blocks
and the files' own functions, with sentinel / replace / replaceOnce / spliceOnce / textSlice / textJoin / pins emulated as
the Zig computes them, and `@hasDecl(sdk, name)` answered as the test build's sdk declares (SDK_DECLS).

Equal but for comments and whitespace: the same C++ tokens (`ctoks`), with line breaks counted where the preprocessor
sees them (a directive's end, a function-like macro's parenthesis). A kernel's source may differ in whitespace alone: MLX
reads the source's words (an input's `_shape` / `_strides` / `_ndim`, the thread attributes) to write the signature, so a
comment there could change the kernel; its header may differ in comments too.

1. Kernels. Every `newKernel(name, inputs, outputs, source, header, ...)` call's source and header are byte-identical,
   or equal but for comments and whitespace (each such kernel listed by name). A call inside a declaration the change
   removed (REMOVED) is listed by kernel name, and so is a call of a kernel the change added (ADDED_KERNELS: a name the
   original file does not call), with whether its source and header are new or an original kernel's; any other missing
   or added call fails.
2. Text consts. Every top-level const both trees evaluate to a string is identical where both define it, or equal but
   for comments and whitespace (listed by name).
3. Declarations. Every other top-level declaration (functions, launch code, structs, tests) is token for token the
   original, except: text consts (2), tests (4), printMetal (5), the splice helpers / piece consts the move removed
   or added, the declarations a change removed (REMOVED), added (ADDED) or edited (CHANGED), the files it added
   (ADDED_FILES: checked by 1, 2, 6, 7 and 8 like every file, their declarations new), and a kernel free routed through
   the consumer's kernel API (FREE: `_ = mlx.mlx_fast_metal_kernel_free(k)` became `freeKernel(k)`), each listed by name;
   anything else fails.
4. Tests. A test that searched the kernel texts for Metal now reads those strings from .metal files: its tokens are the
   original's except one inserted `const pin = ...pins(@embedFile(...));` and, where a string was, `pin[k]`, a piece
   const or `@embedFile(...)` whose value is that string, and the FREE rewrite (comments may differ; a renamed test is
   listed in RENAMED_TESTS). Needles: every pin line, string literal and text const a test searches with is found in
   every text const and kernel text exactly as often as before.
5. Dumps. Every TURBOEXL3_METAL_DUMP translation unit printMetal prints is byte-identical, or equal but for comments and
   whitespace (listed; both printMetals emulated).
6. Files. Every @embedFile path exists and every .metal file under src/kernels is embedded.
7. No Metal in Zig. No `\\\\` literal and no string literal holding Metal (a Metal token such as threadgroup, simd_,
   uint, half, float, device, #include, [[; or a punctuated string found verbatim in a .metal file) in any .zig under
   src/, except the ones listed by file and exact value with their reason (EXCEPTIONS).
8. Kernel frees. No `mlx.mlx_fast_metal_kernel_free` call in src/*.zig: every kernel turboexl3 drops goes through
   `decode.freeKernel`, the consumer's kernel API (mlx-c's own free would skip mlx-stream's launch-cache eviction, and
   the plugin's server owns its kernels).
9. HIP. The AMD kernels live only in src/kernels/hip/**.hip and .hpp: every such file is a group source or a header
   that src/hip/flags.zig lists (and every listed file exists); no string literal in any .zig under src/ (src/hip/
   included) holds HIP device code (HIP_TOKENS); and each HIP file is listed as the original (byte for byte, or equal but
   for comments and whitespace), changed or added, with the entry points (`extern "C"` kernels and their macros) of
   every added or changed file.

Exit status 0 when everything holds.
"""
import hashlib
import os
import re
import sys


# ------------------------------------------------------------------------------------------------------------- tokens

class Tok:
    __slots__ = ('kind', 'text', 'value', 'line')

    def __init__(self, kind, text, value, line):
        self.kind, self.text, self.value, self.line = kind, text, value, line

    def key(self):
        return (self.kind, self.text if self.kind != 'mlstr' else self.value)

    def __repr__(self):
        return f'{self.kind}:{self.text if self.kind != "mlstr" else "<multiline>"}@{self.line}'


ESC = {'n': '\n', 'r': '\r', 't': '\t', '\\': '\\', "'": "'", '"': '"'}


def unescape(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c != '\\':
            out.append(c)
            i += 1
            continue
        e = s[i + 1]
        if e in ESC:
            out.append(ESC[e])
            i += 2
        elif e == 'x':
            out.append(chr(int(s[i + 2:i + 4], 16)))
            i += 4
        elif e == 'u':
            j = s.index('}', i)
            out.append(chr(int(s[i + 3:j], 16)))
            i = j + 1
        else:
            raise ValueError(f'unknown escape \\{e}')
    return ''.join(out)


class TokenizeError(Exception):
    pass


def tokenize(src, name):
    toks, i, n, line = [], 0, len(src), 1
    while i < n:
        c = src[i]
        if c == '\n':
            line += 1
            i += 1
        elif c in ' \t\r':
            i += 1
        elif src.startswith('//', i):
            j = src.find('\n', i)
            j = n if j < 0 else j
            toks.append(Tok('comment', src[i:j], None, line))
            i = j
        elif src.startswith('\\\\', i):
            # a multiline string literal: this line and every directly following line that starts with \\
            vals, start = [], line
            while True:
                j = src.find('\n', i)
                j = n if j < 0 else j
                vals.append(src[i + 2:j])
                k = j + 1
                while k < n and src[k] in ' \t':
                    k += 1
                if j < n and src.startswith('\\\\', k):
                    line += 1
                    i = k
                    continue
                # Zig would also join \\ lines across blank or comment lines: refuse that layout instead of guessing
                m = k
                while m < n and src[m] in ' \t\n':
                    m += 1
                if src.startswith('\\\\', m) or (src.startswith('//', m) and re.match(r'(//[^\n]*\n[ \t]*)+\\\\', src[m:])):
                    raise TokenizeError(f'{name}:{line}: multiline literal lines separated by blank or comment lines')
                i = j
                break
            toks.append(Tok('mlstr', None, '\n'.join(vals), start))
        elif c == '"':
            j = i + 1
            while src[j] != '"':
                j += 2 if src[j] == '\\' else 1
            toks.append(Tok('str', src[i:j + 1], unescape(src[i + 1:j]), line))
            i = j + 1
        elif c == "'":
            j = i + 1
            while src[j] != "'":
                j += 2 if src[j] == '\\' else 1
            toks.append(Tok('char', src[i:j + 1], None, line))
            i = j + 1
        elif c == '@' and i + 1 < n and src[i + 1] == '"':
            j = i + 2
            while src[j] != '"':
                j += 2 if src[j] == '\\' else 1
            toks.append(Tok('id', src[i:j + 1], None, line))
            i = j + 1
        elif c == '@' or c.isalpha() or c == '_':
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] == '_'):
                j += 1
            toks.append(Tok('builtin' if c == '@' else 'id', src[i:j], None, line))
            i = j
        elif c.isdigit():
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] == '_' or (src[j] == '.' and j + 1 < n and src[j + 1].isdigit())):
                j += 1
            toks.append(Tok('num', src[i:j], None, line))
            i = j
        elif src.startswith('++', i):
            toks.append(Tok('op', '++', None, line))
            i += 2
        else:
            toks.append(Tok('op', c, None, line))
            i += 1
    return toks


def read_bytes(path):
    with open(path, 'rb') as f:
        return f.read()


# ------------------------------------------------------------------------------------------------------- evaluation

class EvalError(Exception):
    pass


OPEN, CLOSE = ('{', '(', '['), ('}', ')', ']')


class Unit:
    """one .zig file: its comment-free tokens, top-level declarations, consts, functions, tests, locals and imports"""

    def __init__(self, name, toks):
        self.name = name
        self.t = [x for x in toks if x.kind != 'comment']
        self.consts, self.fns, self.tests, self.locals, self.imports = {}, {}, {}, {}, {}
        self.decls = []  # (key, first token, past the last token)
        t = self.t
        # every declaration at any depth: the .zig imports (a function may import a module itself) and the local consts
        for k in range(len(t) - 1):
            if self.is_decl(k):
                eq = self.decl_eq(k)
                end = self.expr_end(eq + 1)
                if t[eq + 1].text == '@import' and t[eq + 2].text == '(' and t[eq + 3].kind == 'str' \
                        and t[eq + 3].value.endswith('.zig') and t[eq + 4].text == ')' and eq + 5 == end:
                    self.imports.setdefault(t[k + 1].text, t[eq + 3].value)
                self.locals.setdefault(t[k + 1].text, (eq + 1, end))
        # the top level
        k = 0
        while k < len(t):
            start = k
            if t[k].text == 'pub':
                k += 1
            while t[k].text in ('extern', 'export', 'inline', 'noinline', 'threadlocal'):
                k += 2 if t[k + 1].kind == 'str' else 1  # extern "c"
            kw = t[k].text
            if kw == 'fn' and self.body_less(k):
                end = self.expr_end(k)
                self.decls.append((f'fn {t[k + 1].text}', start, end + 1))
                k = end + 1
                continue
            if kw in ('const', 'var'):
                eq = self.decl_eq(k)
                end = self.expr_end(eq + 1)
                if kw == 'const':
                    self.consts[t[k + 1].text] = (eq + 1, end)
                self.decls.append((f'{kw} {t[k + 1].text}', start, end + 1))
                k = end + 1
            elif kw == 'fn':
                fname = t[k + 1].text
                close = self.match(k + 2)
                params = []
                for (a, b) in self.split_commas(k + 3, close):
                    names = [y.text for y in t[a:b] if y.kind == 'id' and y.text != 'comptime']
                    if names:
                        params.append(names[0])
                body = close + 1
                while t[body].text != '{':  # past the return type (which may hold `struct { ... }` or `error{ ... }`)
                    if t[body].text in ('struct', 'enum', 'union', 'opaque', 'error'):
                        if t[body + 1].text == '(':
                            body = self.match(body + 1)
                        if t[body + 1].text == '{':
                            body = self.match(body + 1)
                    body += 1
                bend = self.match(body)
                self.fns[fname] = (params, body, bend)
                self.decls.append((f'fn {fname}', start, bend + 1))
                k = bend + 1
            elif kw == 'test':
                tname = t[k + 1].value if t[k + 1].kind == 'str' else f'#{len(self.tests)}'
                body = k + 2 if t[k + 1].kind == 'str' else k + 1
                bend = self.match(body)
                self.tests[tname] = (body, bend)
                self.decls.append((f'test {tname}', start, bend + 1))
                k = bend + 1
            elif kw == 'comptime':
                bend = self.match(k + 1)
                self.decls.append((f'comptime #{len(self.decls)}', start, bend + 1))
                k = bend + 1
            else:
                raise EvalError(f'{name}: unexpected top-level token {t[k]!r}')

    def is_sdk(self, name):
        """`name` is this file's `const name = @import("sdk");`"""
        if name not in self.locals:
            return False
        a, b = self.locals[name]
        return [x.text for x in self.t[a:b]] == ['@import', '(', '"sdk"', ')']

    def body_less(self, k):
        """the fn at k is a prototype (`extern "c" fn f(...) T;`): a `;` comes before any `{` past its parameters"""
        m = self.match(k + 2) + 1
        while self.t[m].text not in ('{', ';'):
            m += 1
        return self.t[m].text == ';'

    def is_decl(self, k):
        """t[k] is a declaration's `const` (not a pointer or slice type's: `[]const u8`, `*const T`)"""
        t = self.t
        return t[k].kind == 'id' and t[k].text == 'const' and k + 1 < len(t) and t[k + 1].kind == 'id' \
            and (k == 0 or t[k - 1].text in (';', '{', '}', 'pub'))

    def decl_eq(self, k):
        """the index of the `=` of the declaration at k (past its type, if any)"""
        eq = k + 2
        while self.t[eq].text != '=':
            eq += 1
        return eq

    def match(self, k):
        """the index of the bracket closing the one at k"""
        t, depth = self.t, 0
        for m in range(k, len(self.t)):
            if t[m].text in OPEN:
                depth += 1
            elif t[m].text in CLOSE:
                depth -= 1
                if depth == 0:
                    return m
        raise EvalError('unbalanced')

    def expr_end(self, k):
        """the index of the `;` ending the expression that starts at k"""
        t, depth = self.t, 0
        while True:
            if t[k].text in OPEN:
                depth += 1
            elif t[k].text in CLOSE:
                depth -= 1
            elif t[k].text == ';' and depth == 0:
                return k
            k += 1

    def split_commas(self, a, b):
        out, depth, s = [], 0, a
        for m in range(a, b):
            x = self.t[m].text
            if x in OPEN:
                depth += 1
            elif x in CLOSE:
                depth -= 1
            elif x == ',' and depth == 0:
                out.append((s, m))
                s = m + 1
        if s < b:
            out.append((s, b))
        return out


def n_replace_once(s, needle, with_):
    if s.count(needle) != 1:
        raise EvalError(f'replaceOnce: {s.count(needle)} occurrences of {needle!r}')
    return s.replace(needle, with_)


def n_text_slice(src, a, b):
    i = src.find(a)
    if i < 0:
        raise EvalError(f'textSlice: start not found {a!r}')
    j = src.find(b, i)
    if j < 0:
        raise EvalError(f'textSlice: end not found {b!r}')
    return src[i:j + len(b)]


def n_pins(text):
    """test_support.pins: one string per line, a final newline ending the last line"""
    body = text[:-1] if text.endswith('\n') else text
    return body.split('\n')


NATIVES = {
    'sentinel': lambda s: s,
    'replace': lambda s, needle, with_: s.replace(needle, with_),
    'replaceOnce': n_replace_once,
    'spliceOnce': n_replace_once,
    'textSlice': n_text_slice,
    'textJoin': lambda parts: ''.join(parts),
    'pins': n_pins,
}


class Program:
    def __init__(self, src_dir, units):
        self.dir, self.units, self.cache = src_dir, units, {}

    def const(self, unit, name, local_ok=False):
        key = (unit.name, name)
        if key in self.cache:
            if self.cache[key] is None:
                raise EvalError('cycle')
            return self.cache[key]
        if name in unit.consts:
            a, b = unit.consts[name]
        elif local_ok and name in unit.locals:
            a, b = unit.locals[name]
        else:
            raise EvalError(f'{unit.name}: no const {name}')
        self.cache[key] = None
        try:
            v = Eval(self, unit, a, b, {}).whole()
        except Exception as e:  # anything this small evaluator does not cover: not evaluable
            del self.cache[key]
            raise e if isinstance(e, EvalError) else EvalError(f'{type(e).__name__}: {e}')
        self.cache[key] = v
        return v

    def call(self, unit, name, args):
        if name in NATIVES:
            return NATIVES[name](*args)
        if name not in unit.fns:
            raise EvalError(f'{unit.name}: no fn {name}')
        params, a, b = unit.fns[name]
        if len(params) != len(args):
            raise EvalError(f'{name}: arity')
        r = Eval(self, unit, a, b + 1, dict(zip(params, args))).block()
        if r is None or r[0] != 'return':
            raise EvalError(f'{name}: no return')
        return r[1]

    def embed(self, path):
        full = os.path.normpath(os.path.join(self.dir, path))
        if not os.path.isfile(full):
            raise EvalError(f'no file {path}')
        return read_bytes(full).decode('utf-8')


class Eval:
    def __init__(self, prog, unit, a, b, env):
        self.p, self.u, self.t, self.k, self.end, self.env = prog, unit, unit.t, a, b, env

    def peek(self, d=0):
        if self.k + d >= self.end:
            raise EvalError('past the end')
        return self.t[self.k + d]

    def la(self, d):
        """the text of the token d ahead, None past the range (lookahead)"""
        return self.t[self.k + d].text if self.k + d < self.end else None

    def take(self, text=None):
        x = self.peek()
        if text is not None and x.text != text:
            raise EvalError(f'expected {text!r}, got {x!r}')
        self.k += 1
        return x

    def whole(self):
        v = self.expr()
        if self.k != self.end:
            raise EvalError(f'trailing tokens at {self.peek()!r}')
        return v

    def expr(self):
        if self.peek().text == 'comptime':
            self.take()
            return self.expr()
        if self.peek().text == 'if':  # comptime-known: only the taken branch is evaluated (Zig analyzes no other)
            self.take()
            self.take('(')
            c = self.expr()
            self.take(')')
            if c:
                a = self.expr()
                self.take('else')
                self.skip(stop_else=False)
                return a
            self.skip(stop_else=True)
            self.take('else')
            return self.expr()
        v = self.term()
        while self.k < self.end and self.peek().text == '++':
            self.take()
            w = self.term()
            if type(v) is not type(w):
                raise EvalError('++ of mixed types')
            v = v + w
        return v

    def skip(self, stop_else):
        """past one branch without evaluating it: up to its `else` (`stop_else`), else to the `,` `;` or closing bracket
        that ends the enclosing expression; a nested `if` takes its own `else`"""
        depth = ifs = 0
        while self.k < self.end:
            x = self.t[self.k].text
            if depth == 0:
                if x == 'if':
                    ifs += 1
                elif x == 'else':
                    if ifs:
                        ifs -= 1
                    elif stop_else:
                        return
                elif x in (',', ';') or x in CLOSE:
                    if stop_else:
                        raise EvalError('if without else')
                    return
            if x in OPEN:
                depth += 1
            elif x in CLOSE:
                depth -= 1
            self.k += 1
        if stop_else:
            raise EvalError('if without else')

    def term(self):
        v = self.primary()
        while self.la(0) == '[' and self.la(2) == ']' and self.peek(1).kind == 'num':  # an index: pin[k]
            if not isinstance(v, list):
                raise EvalError('index of a non-list')
            v = v[int(self.peek(1).text)]
            self.k += 3
        return v

    def primary(self):
        x = self.peek()
        if x.kind in ('mlstr', 'str'):
            self.take()
            return x.value
        if x.kind == 'builtin' and x.text == '@embedFile':
            self.take()
            self.take('(')
            p = self.take()
            self.take(')')
            return self.p.embed(p.value)
        if x.kind == 'builtin' and x.text == '@hasDecl':  # @hasDecl(sdk, "name"): what the test build's sdk declares
            self.take()
            self.take('(')
            mod = self.take().text
            self.take(',')
            decl = self.take().value
            self.take(')')
            if not self.u.is_sdk(mod):
                raise EvalError(f'@hasDecl of {mod}, not the sdk')
            return decl in SDK_DECLS
        if x.kind == 'builtin' and x.text == '@import':  # @import("x.zig").name, .name(args)
            self.take()
            self.take('(')
            f = self.take().value
            self.take(')')
            self.take('.')
            member = self.take().text
            if f not in self.p.units:
                raise EvalError(f'no unit {f}')
            return self.member(self.p.units[f], member)
        if x.text == '(':
            self.take()
            v = self.expr()
            self.take(')')
            return v
        if x.text == '&' and self.la(1) == '.' and self.la(2) == '{':
            self.k += 3
            out = []
            while self.peek().text != '}':
                out.append(self.expr())
                if self.peek().text == ',':
                    self.take()
            self.take('}')
            return out
        if x.kind == 'id':
            if x.text in ('true', 'false'):
                self.take()
                return x.text == 'true'
            if self.la(1) == ':' and self.la(2) == '{':  # a labeled block
                self.k += 2
                r = self.block()
                if r is None or r[0] != 'break':
                    raise EvalError('labeled block without break')
                return r[1]
            if self.la(1) == '(':
                self.take()
                return self.p.call(self.u, x.text, self.args())
            self.take()
            if self.la(0) == '.' and self.la(1) is not None and self.peek(1).kind == 'id':
                self.take()
                member = self.take().text
                if x.text not in self.u.imports:
                    raise EvalError(f'{x.text} is not a .zig import')
                return self.member(self.p.units[self.u.imports[x.text]], member)
            if x.text in self.env:
                return self.env[x.text]
            return self.p.const(self.u, x.text)
        raise EvalError(f'unsupported term {x!r}')

    def args(self):
        self.take('(')
        out = []
        while self.peek().text != ')':
            out.append(self.expr())
            if self.peek().text == ',':
                self.take()
        self.take(')')
        return out

    def member(self, unit, name):
        if self.la(0) == '(':
            return self.p.call(unit, name, self.args())
        return self.p.const(unit, name)

    def block(self):
        """`{ statements }`: ('return' | 'break', value), or None; past the block's `}` either way"""
        start = self.k
        self.take('{')
        while self.peek().text != '}':
            x = self.peek()
            if x.text == 'comptime' and self.la(1) == '{':
                self.take()
                r = self.block()
                if r is not None:
                    self.k = self.u.match(start) + 1
                    return r
            elif x.kind == 'builtin' and x.text == '@setEvalBranchQuota':
                self.k = self.u.expr_end(self.k) + 1
            elif x.text == 'return':
                self.take()
                v = self.expr()
                self.take(';')
                self.k = self.u.match(start) + 1
                return ('return', v)
            elif x.text == 'break':
                self.take()
                self.take(':')
                self.take()
                v = self.expr()
                self.take(';')
                self.k = self.u.match(start) + 1
                return ('break', v)
            elif x.text in ('var', 'const'):
                self.take()
                name = self.take().text
                while self.peek().text != '=':
                    self.take()
                self.take('=')
                self.env[name] = self.expr()
                self.take(';')
            elif x.kind == 'id' and self.la(1) == '=':
                self.take()
                self.take('=')
                self.env[x.text] = self.expr()
                self.take(';')
            else:
                raise EvalError(f'unsupported statement {x!r}')
        self.take('}')
        return None


def value_of(prog, unit, a, b, env=None):
    try:
        return Eval(prog, unit, a, b, env if env is not None else {}).whole()
    except Exception as e:
        return EvalError(f'{type(e).__name__}: {e}')


def kernel_calls(prog, unit):
    """(name, line, source, header, token index) of every newKernel call in the unit (an EvalError where not evaluable)"""
    t, out = unit.t, []
    for k in range(len(t) - 1):
        if t[k].text == 'newKernel' and t[k + 1].text == '(':
            close = unit.match(k + 1)
            args = unit.split_commas(k + 2, close)
            vals = []
            for (a, b) in (args[3], args[4]):
                if b - a == 1 and t[a].kind == 'id' and t[a].text not in unit.consts:
                    try:
                        vals.append(prog.const(unit, t[a].text, local_ok=True))
                    except EvalError as e:
                        vals.append(e)
                else:
                    vals.append(value_of(prog, unit, a, b))
            out.append((t[args[0][0]].value, t[k].line, vals[0], vals[1], k))
    return out


def enclosing(unit, k):
    """the key of the top-level declaration holding token k"""
    return next((key for (key, a, b) in unit.decls if a <= k < b), None)


# --------------------------------------------------------------------------------------------- comments and whitespace

C_PUNCT = ('>>=', '<<=', '...', '->*', '::', '->', '++', '--', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||', '+=', '-=',
           '*=', '/=', '%=', '&=', '|=', '^=', '##', '.*')


def ctoks(text, comments=False):
    """a Metal text's C++ tokens, whitespace dropped, and comments too unless `comments` (then each is a token of its exact
    text). Line breaks count only where the preprocessor sees them: a directive's tokens end with '<eol>', a function-like
    macro's name is followed by '<fn>' (no space before its parenthesis), and an #include's <name> is one token."""
    out, i, n = [], 0, len(text)
    bol = True  # only whitespace and comments since the line began
    pp = False  # inside a directive
    while i < n:
        c = text[i]
        if c == '\n':
            if pp:
                out.append('<eol>')
                pp = False
            bol = True
            i += 1
        elif c in ' \t\r\f\v':
            i += 1
        elif text.startswith('\\\n', i):  # a line splice
            i += 2
        elif text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
            if comments:
                out.append(text[i:j])
            i = j
        elif text.startswith('/*', i):  # one space to the preprocessor, even across lines
            j = text.find('*/', i + 2)
            if j < 0:
                raise ValueError('unterminated /* comment')
            if comments:
                out.append(text[i:j + 2])
            i = j + 2
        elif c == '#' and bol:
            out.append('<pp>')
            pp, bol = True, False
            i += 1
        else:
            bol = False
            if c in '"\'':
                j = i + 1
                while j < n and text[j] != c:
                    j += 2 if text[j] == '\\' else 1
                out.append(text[i:j + 1])
                i = j + 1
            elif c == '<' and out[-2:] == ['<pp>', 'include']:
                j = text.index('>', i)
                out.append(text[i:j + 1])
                i = j + 1
            elif c.isalpha() or c == '_':
                j = i + 1
                while j < n and (text[j].isalnum() or text[j] == '_'):
                    j += 1
                out.append(text[i:j])
                if out[-3:-1] == ['<pp>', 'define'] and text.startswith('(', j):
                    out.append('<fn>')
                i = j
            elif c.isdigit() or (c == '.' and i + 1 < n and text[i + 1].isdigit()):  # a pp-number
                j = i + 1
                while j < n:
                    if text[j] in 'eEpP' and j + 1 < n and text[j + 1] in '+-':
                        j += 2
                    elif text[j].isalnum() or text[j] in '._':
                        j += 1
                    else:
                        break
                out.append(text[i:j])
                i = j
            else:
                p = next((p for p in C_PUNCT if text.startswith(p, i)), c)
                out.append(p)
                i += len(p)
    if pp:
        out.append('<eol>')
    return out


def same_but_cw(a, b, comments=False):
    """a and b differ only in comments and whitespace (in whitespace alone with `comments`)"""
    try:
        return ctoks(a, comments) == ctoks(b, comments)
    except ValueError:
        return False


# ------------------------------------------------------------------------------------------------------- the free path

FREE_ORIG = [('id', '_'), ('op', '='), ('id', 'mlx'), ('op', '.'), ('id', 'mlx_fast_metal_kernel_free')]


def free_new(f):
    """how NEW's file f calls the one kernel free (decode.zig's `freeKernel`)"""
    return [('id', 'freeKernel')] if f == 'decode.zig' else [('id', 'decode'), ('op', '.'), ('id', 'freeKernel')]


def free_rewritten(keys, f):
    """ORIG's token keys with every `_ = mlx.mlx_fast_metal_kernel_free` as NEW's file f calls the free, and the count"""
    out, k, n, rep = [], 0, 0, free_new(f)
    while k < len(keys):
        if keys[k:k + len(FREE_ORIG)] == FREE_ORIG:
            out.extend(rep)
            k += len(FREE_ORIG)
            n += 1
        else:
            out.append(keys[k])
            k += 1
    return out, n


# ------------------------------------------------------------------------------------------------------------ tests

def primary_end(t, k):
    """the end of a string-valued primary at k: "..." | name | name[NUM] | @embedFile("...") (None if none starts there)"""
    if t[k].kind in ('str', 'mlstr'):
        return k + 1
    if t[k].kind == 'builtin' and t[k].text == '@embedFile' and t[k + 1].text == '(' and t[k + 2].kind == 'str' and t[k + 3].text == ')':
        return k + 4
    if t[k].kind == 'id' and t[k + 1].text not in ('(', '.', '='):
        if t[k + 1].text == '[' and t[k + 2].kind == 'num' and t[k + 3].text == ']':
            return k + 4
        return k + 1
    return None


def align_test(oname, nname, f, ou, nu, oprog, nprog):
    """(strings moved to files, frees rewritten, problem or None): ORIG's test body against NEW's, see check 4"""
    oa, ob = ou.tests[oname]
    na, nb = nu.tests[nname]
    o, n = ou.t, nu.t
    i, j, env, moved, frees = oa, na, {}, 0, 0
    fnew = free_new(f)
    while i <= ob or j <= nb:
        if j <= nb and n[j].text == 'const' and n[j + 1].text == 'pin' and n[j + 2].text == '=' and not (i <= ob and o[i].text == 'const' and o[i + 1].text == 'pin'):
            end = nu.expr_end(j + 3)
            v = value_of(nprog, nu, j + 3, end)
            if isinstance(v, EvalError):
                return moved, frees, f'its pin line: {v}'
            env['pin'] = v
            j = end + 1
            continue
        if i <= ob and j <= nb and [x.key() for x in o[i:i + len(FREE_ORIG)]] == FREE_ORIG and [x.key() for x in n[j:j + len(fnew)]] == fnew:
            i += len(FREE_ORIG)
            j += len(fnew)
            frees += 1
            continue
        if i <= ob and j <= nb and o[i].key() == n[j].key():
            i += 1
            j += 1
            continue
        oe = primary_end(o, i) if i <= ob else None
        ne = primary_end(n, j) if j <= nb else None
        if oe is None or ne is None:
            return moved, frees, f'tokens differ at original line {o[i].line if i <= ob else "end"} / new line {n[j].line if j <= nb else "end"}'
        ov = value_of(oprog, ou, i, oe)
        nv = value_of(nprog, nu, j, ne, env)
        if isinstance(ov, EvalError) or isinstance(nv, EvalError) or ov != nv:
            return moved, frees, f'original line {o[i].line}: {ov!r:.80} != new line {n[j].line}: {nv!r:.80}'
        moved += 1
        i, j = oe, ne
    return moved, frees, None


def test_named(f, u, tname, otext, ntext):
    """{key: (ORIG value, NEW value)} of every text const test `tname` of NEW's unit u names (`x` or `module.x`)"""
    a, b = u.tests[tname]
    t, out = u.t, {}
    for k in range(a, b):
        x = t[k]
        if x.kind == 'id' and t[k - 1].text != '.':
            if t[k + 1].text == '.' and x.text in u.imports and t[k + 2].kind == 'id':
                key = (u.imports[x.text], t[k + 2].text)
            else:
                key = (f, x.text)
            if key in otext and key in ntext:
                out[key] = (otext[key], ntext[key])
    return out


def test_literals(u, tname):
    """the string literals test `tname` of NEW's unit u holds, and the pin files it reads"""
    a, b = u.tests[tname]
    t, lits, pins = u.t, set(), []
    for k in range(a, b):
        if t[k].kind == 'str' and t[k].value:
            if t[k - 2].text == '@embedFile' and t[k].value.startswith('kernels/test/pins/'):
                pins.append(t[k].value)
            elif t[k - 2].text not in ('@embedFile', '@import'):
                lits.add(t[k].value)
    return lits, pins


DUMP_TESTS = ('turboexl3 prefill metal sources: every kernel as a standalone translation unit (TURBOEXL3_METAL_DUMP=1)',
              'turboexl3 prefill 128-row tiles metal sources: the route, both GEMMs\' launches (both bodies), mid and finish '
              'as standalone translation units (TURBOEXL3_METAL_DUMP=1)')
ORIG_ATTRS = (', uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint simdgroup_index_in_threadgroup '
              '[[simdgroup_index_in_threadgroup]], uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], uint '
              'thread_index_in_threadgroup [[thread_index_in_threadgroup]]) {{\n{s}\n}}\n@@METAL_END\n')


def metal_args(t, a, b):
    """the original's `[_]MetalArg{ .{ .name = "x", .ty = "half", .out = true, .shape = true }, ... }`"""
    out, k = [], a
    while k < b:
        if t[k].text == '.' and t[k + 1].text == '{':
            arg = {'out': False, 'shape': False}
            k += 2
            while t[k].text != '}':
                if t[k].text == '.' and t[k + 1].kind == 'id' and t[k + 2].text == '=':
                    v = t[k + 3]
                    arg[t[k + 1].text] = v.value if v.kind == 'str' else (v.text == 'true')
                    k += 4
                else:
                    k += 1
            out.append(arg)
        k += 1
    return out


def print_orig(name, header, consts, args, body):
    """the original printMetal's output"""
    s = f'@@METAL_BEGIN {name}\n{header}\n{consts}\n[[kernel]] void {name}('
    b = 0
    for a in args:
        s += f'{"" if b == 0 else ", "}{"device" if a["out"] else "const device"} {a["ty"]}* {a["name"]} [[buffer({b})]]'
        b += 1
        if a['shape']:
            s += f', const constant int* {a["name"]}_shape [[buffer({b})]]'
            b += 1
    return s + ORIG_ATTRS.replace('{{', '{').replace('}}', '}').replace('{s}', body)


def zig_format(fmt, vals):
    out, i, v = [], 0, iter(vals)
    while i < len(fmt):
        if fmt.startswith('{{', i) or fmt.startswith('}}', i):
            out.append(fmt[i])
            i += 2
        elif fmt.startswith('{s}', i):
            out.append(next(v))
            i += 3
        elif fmt[i] in '{}':
            raise EvalError(f'format spec at {fmt[i:i + 4]!r}')
        else:
            out.append(fmt[i])
            i += 1
    return ''.join(out)


def dump_outputs(prog, unit, tname):
    """every unit printMetal prints in a dump test, by the tree's printMetal: the original's (MetalArg tables) or the new's"""
    t = unit.t
    a, b = unit.tests[tname]
    orig = any(key == 'const MetalArg' for (key, _, _) in unit.decls)
    if orig:
        pa, pb = unit.fns['printMetal'][1:]
        fmts = [x.value for x in t[pa:pb] if x.kind == 'str']
        assert fmts[:5] == ['@@METAL_BEGIN {s}\n{s}\n{s}\n[[kernel]] void {s}(', '{s}{s} {s}* {s} [[buffer({d})]]', '', ', ',
                            'device'] and ORIG_ATTRS in fmts, 'the original printMetal is not the one emulated'
    else:
        pa, pb = unit.fns['printMetal'][1:]
        k = next(m for m in range(pa, pb) if t[m].text == 'print' and t[m + 1].text == '(')
        fmt = t[k + 2].value
        tup = [(x, y) for (x, y) in unit.split_commas(k + 6, unit.match(k + 5))]
    env, outs = {}, []
    k = a + 1
    while k < b:
        if t[k].text == 'const' and t[k + 1].kind == 'id':
            name = t[k + 1].text
            eq = unit.decl_eq(k)
            end = unit.expr_end(eq + 1)
            if any(y.text == 'MetalArg' for y in t[eq + 1:end]):
                env[name] = metal_args(t, eq + 1, end)
            else:
                v = value_of(prog, unit, eq + 1, end, env)
                if isinstance(v, EvalError):
                    raise v
                env[name] = v
            k = end + 1
            continue
        if t[k].text == 'printMetal' and t[k + 1].text == '(':
            close = unit.match(k + 1)
            vals = []
            for (p, q) in unit.split_commas(k + 2, close):
                if t[p].text == '&':
                    vals.append(env[t[p + 1].text])
                else:
                    v = value_of(prog, unit, p, q, env)
                    if isinstance(v, EvalError):
                        raise v
                    vals.append(v)
            if orig:
                outs.append(print_orig(*vals))
            else:
                penv = dict(zip(unit.fns['printMetal'][0], vals))
                outs.append(zig_format(fmt, [value_of(prog, unit, x, y, penv) for (x, y) in tup]))
            k = close + 1
            continue
        k += 1
    return outs


# --------------------------------------------------------------------------------------------------------- the scan

METAL_TOK = re.compile(r'threadgroup|simd_|simdgroup|\buint\d*\b|\bulong\b|\bushort\b|\bhalf\d?\b|\bfloat\d?\b|\bbfloat\w*|#include'
                       r'|#define|#if|\[\[|\bconstexpr\b|\bdevice\b|\bconstant\b|metal::|as_type|reinterpret_cast|const_cast|mem_flags'
                       r'|\batomic|\bkernel\b')
# the declarations the test build's `sdk` has (mlx-stream's sdk/root.zig and plugin/sdk.zig alike: neither declares
# `steel`, so steel.zig takes the vendored kernels/steel_nax.metal)
SDK_DECLS = {'mlx', 'kernel_launch'}
# string literals that hold Metal-looking text but are not Metal, by file and exact value
EXCEPTIONS = {
    ('spec.zig', '{"expert_quant":{"format":"exl3","k":4,"codebook":"mul2"}}'): 'JSON test input of the expert_quant parser',
    ('spec.zig', '{"expert_quant":{"format":"affine","k":4,"codebook":"mul1"}}'): 'JSON test input of the expert_quant parser',
    ('steel.zig', '#pragma once'): "kept()'s line matcher over the consumer's MLX steel headers (sdk.steel), not kernel code",
    ('steel.zig', '#include "mlx/'): "kept()'s line matcher over the consumer's MLX steel headers (sdk.steel), not kernel code",
}
# consts whose value the move changed on purpose (every kernel built from them is still checked in 1), and functions that
# compose kernel texts whose bodies changed (their texts are checked in 1 and 2)
RECUT = {('decode.zig', 'rk_line'): 'now the whole line the routed bodies are cut at (indent, newline); the one test that '
                                    'compared the bare statement reads it from its pin file',
         ('decode.zig', 'rk_line_y'): 'the same, the v2 rk line',
         ('decode.zig', 'fin7_4'): 'cut before its SwiGLU line (fin7_swiglu, then fin7_5): text_fin7c swaps that line for the '
                                   'clamp; text_fin7 and text_gu3sf keep their values (2)',
         ('prefill.zig', 'mid_4'): 'cut before its SwiGLU line (mid_swiglu, then mid_5): the clamped mids swap that line for '
                                   'the clamp; text_mid and text_mid_t128 keep their values (2)'}
RECOMPOSED = {('decode.zig', 'fn routePrepText'): 'the router fold over files (text_route_prep / _prepg: 1, 2)'}
# declarations a change removed on purpose, by file and key, with the reason; their newKernel calls are listed in 1
REMOVED = {}
for _name in ('Kernels4', 'text_gu4', 'text_down4', 'gu4_inputs', 'down4_inputs'):
    REMOVED[('decode.zig', f'const {_name}')] = ('decode form 1b (Kernels4), dead: nothing built it, and gu4 / down4 read '
                                               'natural-order positions where turboexl3_prep writes lane order')
for _name in ('Kernels2', 'text_gu2', 'text_down2', 'text_reduce2', 'gu2_inputs', 'down2_inputs', 'down2_outputs', 'reduce2_inputs'):
    REMOVED[('decode.zig', f'const {_name}')] = 'decode form 2 (Kernels2), dead: nothing built it'
# declarations a change added on purpose
ADDED = {('decode.zig', 'fn freeKernel'): "the one kernel free, through the consumer's kernel API (README.md's Wiring)"}
for _f in ('decb.zig', 'decode.zig', 'root.zig'):
    ADDED[(_f, 'const sources')] = 'sources.zig, where the texts more than one kernel splices now live (same values: 2)'
ADDED[('root.zig', 'const gdn_out')] = 'gdn_out.zig, the GDN tail in one launch'
ADDED[('root.zig', 'fn gdnOutPrep')] = 'the GDN tail\'s API: z\'s finish, the gated norm and out_proj\'s prepare in one launch'
ADDED[('root.zig', 'fn moeSwiglu')] = 'moe\'s body with the SwiGLU\'s clamp (limit 0: moe; moeClamped)'
ADDED[('root.zig', 'const clamp_logged')] = 'the clamped SwiGLU\'s one-time engage line'
ADDED[('ref.zig', 'const Hits')] = 'what moeRowClamped clamped (the clamped GPU test requires it to act)'
ADDED[('ref.zig', 'fn swigluClamped')] = 'the reference\'s clamped SwiGLU of one element'
ADDED[('ref.zig', 'fn moeRowClamped')] = 'moeRowDec with the clamped SwiGLU, moeClamped\'s reference'
# files a change added on purpose, by name, with the reason
ADDED_FILES = {'sources.zig': 'the texts more than one kernel splices (had128, the gate gemv and sum), MLX-free, for '
                              'turboexl3\'s kernels and a consumer\'s Metal',
               'gdn_out.zig': 'the GDN tail\'s kernel (turboexl3_gdn_out_prep: its pieces around sources.had128) and launch',
               'gdn_out_test.zig': 'the GDN tail\'s GPU tests against linearFinish, MLX\'s gated norm and linearPrep',
               'hip_tests.zig': 'the AMD (HIP) kernels\' GPU tests against format.zig (zig build hip-test -Dhip=true); no Metal',
               'clamped_test.zig': 'the clamped SwiGLU\'s tests: its texts\' compositions and the reference (CPU), the clamped '
                                   'launches\' bytes where nothing clamps and moeClamped against ref.moeRowClamped at '
                                   'DeepSeek-V4.1-Flash\'s geometry (GPU)'}
# other declarations a change edited on purpose, by file and key, with the reason
CHANGED = {('tests.zig', 'comptime #0'): 'the test root lists the files the change added'}
for _f, _key in (('decode.zig', 'const Geometry'), ('prefill.zig', 'const Geometry')):
    CHANGED[(_f, _key)] = 'the SwiGLU\'s clamp (limit, default 0): part of the geometry, so of the kernel caches\' keys'
CHANGED[('decode.zig', 'const Kernels')] = 'refuses a clamped geometry (the clamp is Kernels3\'s)'
CHANGED[('decode.zig', 'const Kernels5')] = 'refuses a clamped geometry (dn5 has no clamp)'
CHANGED[('decode.zig', 'const Kernels3')] = ('a clamped geometry\'s fin7 is turboexl3_fin7c with LIMIT appended to its constants; '
                                             'moeForm refuses form 3 and sharedAdmitted the shared heads there; below a top-k of 4 '
                                             'moeForm finishes the down with form 3\'s dfin (dfin7 writes each block from its first 128 '
                                             'threads) and sharedAdmitted declines')
CHANGED[('prefill.zig', 'const T128Kernels')] = 'init takes the clamp: a clamped geometry\'s mid is turboexl3_pf_t128_midc'
CHANGED[('prefill.zig', 'const Prefill')] = ('a clamped geometry\'s mids are turboexl3_pf_midc / _mid_vc, LIMIT appended to '
                                             'every mid launch\'s constants there (midTempl)')
for _key in ('fn kernelsFor', 'fn prefillFor'):
    CHANGED[('root.zig', _key)] = 'the kernel cache matches the clamp too'
CHANGED[('root.zig', 'fn moe')] = 'its body moved to moeSwiglu (limit 0)'
CHANGED[('root.zig', 'fn moePrefill')] = 'takes the clamp into the prefill geometry'
CHANGED[('root.zig', 'fn moeClamped')] = 'serves the clamped SwiGLU on moeSwiglu instead of refusing'
CHANGED[('root.zig', 'const shared_state')] = 'registers clamp_logged'
CHANGED[('ref.zig', 'fn moeRowDec')] = 'moeRowClamped at limit 0 (the same arithmetic)'
# kernels a change added on purpose, by file and kernel name, with the reason; their newKernel calls are listed in 1
ADDED_KERNELS = {('dense.zig', 'turboexl3_dense_dgemv_lean'): 'the lean GEMV (Config.dense_lean): text_dgemv with its '
                                                              'codeword loop taken by input element',
                 ('decode.zig', 'turboexl3_gu3sf'): 'the fused shared expert\'s gate|up launch with fin7s folded in '
                                                    '(Config.fused_fin_fold): gu3s, then fin7s\'s statements',
                 ('decode.zig', 'turboexl3_vfold_zero'): 'the zeroing of gu3sf\'s arrival counters: dense\'s DEC-ONE '
                                                         'zeroing text',
                 ('decode.zig', 'turboexl3_fin7c'): 'fin7 with the clamped SwiGLU (root.moeClamped): text_fin7 with its '
                                                    'SwiGLU line swapped for the clamp piece',
                 ('prefill.zig', 'turboexl3_pf_midc'): 'the mid with the clamped SwiGLU: text_mid with its SwiGLU line '
                                                       'swapped for the clamp piece',
                 ('prefill.zig', 'turboexl3_pf_mid_vc'): 'the vector-layout mid with the clamped SwiGLU (text_mid_v, the '
                                                         'same swap)',
                 ('prefill.zig', 'turboexl3_pf_t128_midc'): 'the 128-row tiles\' mid with the clamped SwiGLU '
                                                            '(text_mid_t128, the same swap)'}
# tests a change renamed: (file, original name) -> new name
RENAMED_TESTS = {
    ('bench.zig', "turboexl3 bench: fused vs separately launched steps vs Sushi's chain, K2..K8 (+ n40), Qwen geometry, "
                  "rows 1/4/8 (TURBOEXL3_BENCH=1)"):
        "turboexl3 bench: form-3 variants vs Sushi's chain, K2..K8 (+ n40), Qwen geometry, rows 1/4/8 (TURBOEXL3_BENCH=1)",
}
# DeepSeek-V4.1's trunk: the grouped linear (wo_a), the bias helper, their host reference and tests
ADDED_FILES['grouped.zig'] = ('the grouped linear at decode rows (turboexl3_grouped_prep / _gemv / _finish: dense\'s '
                              'prep_s, the routed down3 and dfin3 texts with their own template arguments)')
ADDED_FILES['topk_test.zig'] = ('root.moe below a top-k of 4 (top-1, 2, 3): forms 7, 8 and 9 give form 3\'s bytes, '
                              'form 3 equals the reference')
ADDED[('ref.zig', 'fn compareRounded')] = ('compare beyond an output store\'s rounding (half an ulp of each value): '
                                         'the chain\'s tolerance for bf16 outputs (clamped_test, topk_test)')
ADDED_FILES['deepseek_test.zig'] = ('DeepSeek-V4.1\'s GPU tests: linear, linearBias and linearGrouped against '
                                    'ref.Linear at every trunk and vision shape')
for _key in ('const grouped', 'const grouped_lock', 'const grouped_cache', 'fn groupedFor', 'fn linearGrouped',
             'fn groupedPrefill', 'fn linearBias'):
    ADDED[('root.zig', _key)] = 'the grouped linear (linearGrouped) and the bias helper (linearBias)'
ADDED[('ref.zig', 'const Linear')] = 'a dense linear\'s host reference with its W_hat decoded once (format.project\'s values)'
ADDED[('ref.zig', 'fn synthLinear')] = 'a random dense linear (or G stacked slices) for ref.Linear'
CHANGED[('root.zig', 'const dense_cache')] = '64 dense shapes: DeepSeek-V4.1\'s set beside the others in one process'
CHANGED[('root.zig', 'const shared_state')] = 'the grouped linears\' cache and its lock'


def metal_strings(fn, toks, corpus):
    """(line, kind, value) of every literal in a .zig file that holds Metal"""
    out = []
    for k, x in enumerate(toks):
        if x.kind == 'mlstr':
            out.append((x.line, 'mlstr', x.value))
            continue
        if x.kind != 'str':
            continue
        prev = [y for y in toks[max(0, k - 2):k] if y.kind != 'comment']
        if prev and prev[-1].text == 'test':
            continue  # a test's name
        if len(prev) == 2 and prev[0].text in ('@embedFile', '@import') and prev[1].text == '(':
            continue  # a path
        v = x.value
        if v and (METAL_TOK.search(v) or (len(v) >= 3 and re.search(r'[ ;(){}\[\]=<>+*&|!?]', v) and v in corpus)):
            out.append((x.line, 'str', v))
    return out


# --------------------------------------------------------------------------------------------------------------- main

def sha(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:12]


# what marks HIP device code in a string (a Zig string never needs these)
HIP_TOKENS = ('__global__', '__device__', '__shared__', '__launch_bounds__', 'threadIdx', 'blockIdx', '__shfl',
              '__syncthreads', '__builtin_amdgcn', 'hip_runtime')


def hip_files(src):
    root = os.path.join(src, 'kernels', 'hip')
    out = {}
    if not os.path.isdir(root):
        return out
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.endswith('.hip') or f.endswith('.hpp'):
                rel = os.path.relpath(os.path.join(d, f), root)
                out[rel] = read_bytes(os.path.join(d, f))
    return out


def hip_entries(text):
    t = text.decode('utf-8', 'replace')
    return sorted(set(re.findall(r'\bt3_[A-Za-z0-9_]+(?=\s*\()', t)) | set(re.findall(r'^\s*(T3_[A-Z0-9_]+)\(', t, re.M)))


def hip_report(orig_dir, new_dir, problems):
    new = hip_files(new_dir)
    old = hip_files(orig_dir)
    flags_path = os.path.join(new_dir, 'hip', 'flags.zig')
    listed = set()
    if os.path.exists(flags_path):
        with open(flags_path, encoding='utf-8') as fh:
            listed = set(re.findall(r'"([^"]+\.(?:hip|hpp))"', fh.read()))
    for f in sorted(set(new) - listed):
        problems.append(f'src/kernels/hip/{f}: not a group source or header in src/hip/flags.zig')
    for f in sorted(listed - set(new)):
        problems.append(f'src/hip/flags.zig: lists {f}, which src/kernels/hip does not hold')
    same = cw = 0
    for f in sorted(new):
        if f in old and old[f] == new[f]:
            same += 1
        elif f in old and same_but_cw(old[f].decode('utf-8', 'replace'), new[f].decode('utf-8', 'replace'), comments=True):
            cw += 1
            print(f'   equal but for comments and whitespace: {f}')
        else:
            print(f'   {"changed" if f in old else "added"}: {f}: {", ".join(hip_entries(new[f])) or "(helpers)"}')
    for f in sorted(set(old) - set(new)):
        print(f'   removed: {f}: {", ".join(hip_entries(old[f])) or "(helpers)"}')
    zig_hits = 0
    for d, _, fs in os.walk(new_dir):
        for f in fs:
            if not f.endswith('.zig'):
                continue
            path = os.path.join(d, f)
            with open(path, encoding='utf-8', newline='') as fh:
                toks = tokenize(fh.read(), f)
            for tk in toks:
                if tk.kind in ('str', 'mlstr') and any(h in (tk.value or '') for h in HIP_TOKENS):
                    zig_hits += 1
                    problems.append(f'{os.path.relpath(path, new_dir)}:{tk.line}: a string holds HIP device code: {(tk.value or "")[:60]!r}')
    print(f'   {len(new)} HIP files ({same} the original, {cw} equal but for comments and whitespace); '
          f'{len(listed)} listed in flags.zig; {zig_hits} Zig strings holding HIP\n')


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    verbose = '-v' in sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(2)
    orig_dir = os.path.abspath(args[0])
    new_dir = os.path.abspath(args[1]) if len(args) > 1 else os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
    print(f'ORIG {orig_dir}\nNEW  {new_dir}\n')
    problems = []

    def load(d):
        units = {}
        for f in sorted(os.listdir(d)):
            if f.endswith('.zig'):
                with open(os.path.join(d, f), encoding='utf-8', newline='') as fh:
                    units[f] = Unit(f, tokenize(fh.read(), f))
        return units

    ounits, nunits = load(orig_dir), load(new_dir)
    oprog, nprog = Program(orig_dir, ounits), Program(new_dir, nunits)
    added_files = []
    for f in sorted(set(ounits) ^ set(nunits)):
        if f in nunits and f in ADDED_FILES:
            added_files.append(f)
        else:
            problems.append(f'{f}: only in {"ORIG" if f in ounits else "NEW"}')

    # 1. kernels
    print('1. kernels: every newKernel call\'s source and header, original vs new')
    n_calls = n_same = 0
    names, cw, gone, added_k = set(), [], [], []
    orig_texts = []
    new_calls = {f: kernel_calls(nprog, u) for f, u in nunits.items()}
    new_names = {c[0] for cs in new_calls.values() for c in cs}
    for f in sorted(ounits):
        if f not in nunits:
            continue
        co, cn = kernel_calls(oprog, ounits[f]), new_calls[f]
        orig_texts += [(f, c[0], c[2], c[3]) for c in co]
        kept = []
        for c in co:
            dk = enclosing(ounits[f], c[4])
            if (f, dk) in REMOVED:
                gone.append((c[0], f, dk))
            else:
                kept.append(c)
        # a kernel the change added (ADDED_KERNELS: a name the original file does not call) is set aside and listed
        onames = {c[0] for c in co}
        same_calls = []
        for c in cn:
            if (f, c[0]) in ADDED_KERNELS and c[0] not in onames:
                added_k.append((f, c))
            else:
                same_calls.append(c)
        cn = same_calls
        if [c[0] for c in kept] != [c[0] for c in cn]:
            problems.append(f'{f}: newKernel calls differ in names or order (past the calls of removed declarations)')
            continue
        for (name, line, so, ho, _), (_, nline, sn, hn, _) in zip(kept, cn):
            n_calls += 1
            names.add(name)
            bad = [x for x in (so, ho, sn, hn) if isinstance(x, EvalError)]
            if bad:
                problems.append(f'{f}:{nline} {name}: not evaluable ({bad[0]})')
                continue
            ok = so == sn and ho == hn
            n_same += ok
            if ok:
                status = 'identical'
            elif (so == sn or same_but_cw(so, sn, comments=True)) and (ho == hn or same_but_cw(ho, hn)):
                status = ', '.join(x for x in ('source: whitespace' if so != sn else '', 'header: comments / whitespace' if ho != hn else '') if x)
                cw.append((name, so != sn, ho != hn))
            else:
                status = 'DIFFERS'
                problems.append(f'{f}:{nline} {name}: {"source" if so != sn else "header"} differs '
                                f'{"beyond whitespace" if so != sn else "beyond comments and whitespace"}')
            if verbose:
                print(f'   {name:34s} {f + ":" + str(nline):18s} source {len(sn.encode()):6d} B {sha(sn)}  header {len(hn.encode()):6d} B {sha(hn)}  '
                      f'{status}')
    print(f'   {n_calls} calls ({len(names)} kernel names): {n_same} identical, {len(cw)} equal but for comments and whitespace, '
          f'{n_calls - n_same - len(cw)} not')
    if cw:
        src_ws = [k for (k, s, _) in cw if s]
        hdr_cw = [k for (k, _, h) in cw if h]
        if src_ws:
            print(f'   source differs in whitespace alone ({len(src_ws)}): {", ".join(src_ws)}')
        if hdr_cw:
            print(f'   header differs in comments and whitespace alone ({len(hdr_cw)}): {", ".join(hdr_cw)}')
    if gone:
        by_decl = {}
        for (name, f, dk) in gone:
            by_decl.setdefault((f, dk), []).append(name)
        print(f'   removed with their declarations ({len(gone)} calls):')
        for (f, dk), ns in by_decl.items():
            print(f'     {f} {dk}: {", ".join(ns)}')
        dropped = sorted({n for (n, _, _) in gone} - new_names)
        print(f'   kernels removed ({len(dropped)}): {", ".join(dropped)}')
        for n in sorted({n for (n, _, _) in gone} & new_names):
            left = sum(1 for cs in new_calls.values() for c in cs if c[0] == n)
            print(f'   kept: {n} ({sum(1 for g in gone if g[0] == n)} of its calls removed, {left} left)')
    if added_k:
        def of_orig(f, text, i, comments):
            """the original kernel (file f's first) whose source (i 2) or header (i 3) is `text`: (name, exact) or None"""
            rows = [r for r in orig_texts if r[0] == f] + [r for r in orig_texts if r[0] != f]
            for row in rows:
                if row[i] == text:
                    return row[1], True
            for row in rows:
                if not isinstance(row[i], EvalError) and same_but_cw(row[i], text, comments=comments):
                    return row[1], False
            return None

        print(f'   added on purpose ({len(added_k)} calls):')
        for (f, (name, nline, sn, hn, _)) in added_k:
            bad = [x for x in (sn, hn) if isinstance(x, EvalError)]
            if bad:
                problems.append(f'{f}:{nline} {name}: not evaluable ({bad[0]})')
                continue
            print(f'     {name} ({f}:{nline}): {ADDED_KERNELS[(f, name)]}')
            for (what, text, i, comments) in (('source', sn, 2, True), ('header', hn, 3, False)):
                o = of_orig(f, text, i, comments)
                near = ' but for whitespace' if comments else ' but for comments and whitespace'
                kind = 'a new text' if o is None else (f"{o[0]}'s {what}" + ('' if o[1] else near))
                print(f'       {what} {len(text.encode()):6d} B {sha(text)}: {kind}')
    print()

    # 2. text consts
    print('2. text consts: every top-level const both trees evaluate to a string')
    same = diff = 0
    otext, ntext = {}, {}
    for units, prog, store in ((ounits, oprog, otext), (nunits, nprog, ntext)):
        for f, u in units.items():
            for c in u.consts:
                try:
                    v = prog.const(u, c)
                except EvalError:
                    continue
                if isinstance(v, str):
                    store[(f, c)] = v
    cw_consts = []
    for key in sorted(set(otext) & set(ntext)):
        if otext[key] == ntext[key]:
            same += 1
        elif key in RECUT:
            print(f'   {key[0]} {key[1]}: recut piece, {RECUT[key]}')
        elif same_but_cw(otext[key], ntext[key]):
            cw_consts.append(f'{key[0]} {key[1]}')
        else:
            diff += 1
            problems.append(f'{key[0]} {key[1]}: values differ beyond comments and whitespace')
        if verbose:
            print(f'   {key[0]:13s} {key[1]:24s} {len(ntext[key].encode()):6d} B {sha(ntext[key])}  '
                  f'{"identical" if otext[key] == ntext[key] else ("comments / whitespace" if f"{key[0]} {key[1]}" in cw_consts else "DIFFERS")}')
    print(f'   {same} identical, {len(cw_consts)} equal but for comments and whitespace, {diff} differ; {len(set(otext) - set(ntext))} only in '
          f'the original, {len(set(ntext) - set(otext))} only in the new (listed in 3)')
    if cw_consts:
        print(f'   comments / whitespace ({len(cw_consts)}): {", ".join(cw_consts)}')
    print()

    # 3. declarations
    print('3. declarations: every other top-level declaration token for token the original')
    helpers = {'fn spliceOnce', 'fn textSlice', 'fn textWithSecond', 'fn textWithSecondOf', 'fn replaceOnce'}
    composers = {'fn gemmText', 'fn gemmSmallText', 'fn gemmMixedText', 'fn pins'}
    n_same = 0
    changed, removed, added = [], [], []
    for f in sorted(set(ounits) & set(nunits)):
        ou, nu = ounits[f], nunits[f]
        od = {k: (a, b) for (k, a, b) in ou.decls}
        back = {f'test {new}': f'test {old}' for ((rf, old), new) in RENAMED_TESTS.items() if rf == f and old in ou.tests and new in nu.tests}
        nd = {back.get(k, k): (a, b) for (k, a, b) in nu.decls}
        for key in od:
            if key not in nd:
                if (f, key) in REMOVED:
                    kind = REMOVED[(f, key)]
                elif (f, key.split(' ', 1)[1]) in otext and key.startswith('const '):
                    kind = 'text'
                else:
                    kind = 'splice helper' if key in helpers else ('dump harness (5)' if (f, key) == ('prefill.zig', 'const MetalArg') else None)
                removed.append((f, key, kind))
                if kind is None:
                    problems.append(f'{f}: {key} removed')
                continue
            ot = [x.key() for x in ou.t[od[key][0]:od[key][1]]]
            nt = [x.key() for x in nu.t[nd[key][0]:nd[key][1]]]
            if ot == nt:
                n_same += 1
                continue
            name = key.split(' ', 1)[1]
            rw, n_rw = free_rewritten(ot, f)
            if key.startswith('const ') and (f, name) in RECUT:
                changed.append((f, key, 'recut piece (2)'))
            elif key.startswith('const ') and (f, name) in otext and (f, name) in ntext:
                changed.append((f, key, 'text const, same value' if otext[(f, name)] == ntext[(f, name)] else
                                ('text const, comments / whitespace (2)' if same_but_cw(otext[(f, name)], ntext[(f, name)]) else 'TEXT DIFFERS')))
            elif (f, key) in RECOMPOSED:
                changed.append((f, key, RECOMPOSED[(f, key)]))
            elif (f, key) in CHANGED:
                changed.append((f, key, CHANGED[(f, key)]))
            elif key.startswith('test '):
                changed.append((f, key, 'test (4, 5)' + (', renamed' if key in back.values() else '')))
            elif key == 'fn printMetal' and f == 'prefill.zig':
                changed.append((f, key, 'dump printer (5)'))
            elif n_rw and rw == nt:
                changed.append((f, key, f"kernel free through the consumer's API ({n_rw}: freeKernel)"))
            else:
                changed.append((f, key, 'UNEXPECTED'))
                problems.append(f'{f}: {key} changed')
        for key in nd:
            if key not in od:
                if (f, key) in ADDED:
                    kind = ADDED[(f, key)]
                else:
                    kind = 'piece / text' if key.startswith('const ') and (f, key.split(' ', 1)[1]) in ntext else ('composition' if key in composers else None)
                added.append((f, key, kind))
                if kind is None:
                    problems.append(f'{f}: {key} added')
    print(f'   {n_same} identical; {len(changed)} changed, {len(removed)} removed, {len(added)} added:')
    for f in added_files:
        print(f'     added    {f:13s} file: {ADDED_FILES[f]}')
    for (f, key, why) in changed:
        if why != 'text const, same value' or verbose:
            print(f'     changed  {f:13s} {key[:90]:90s} {why}')
    print(f'     changed  {sum(1 for c in changed if c[2] == "text const, same value")} text consts, same value (-v lists them)')
    by_why = {}
    for (f, key, kind) in removed:
        if kind in ('text', 'splice helper', 'dump harness (5)', None):
            print(f'     removed  {f:13s} {key:40s} {kind or "UNEXPECTED"}')
        else:
            by_why.setdefault((f, kind), []).append(key.split(' ', 1)[1])
    for (f, why), keys in by_why.items():
        print(f'     removed  {f:13s} {", ".join(keys)}: {why}')
    by_kind = {}
    for (f, key, kind) in added:
        by_kind.setdefault((f, kind), []).append(key.split(' ', 1)[1])
    for (f, kind), keys in sorted(by_kind.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        print(f'     added    {f:13s} {kind or "UNEXPECTED"}: {", ".join(keys)}')
    print()

    # 4 and 5. tests and dumps
    print('4. tests: each test compares the original\'s strings, now read from .metal files')
    n_tests = n_moved = n_conv = n_free = n_needle = 0
    for f in sorted(set(ounits) & set(nunits)):
        ou, nu = ounits[f], nunits[f]
        renamed_to = {old: new for ((rf, old), new) in RENAMED_TESTS.items() if rf == f and old in ou.tests and new in nu.tests}
        for name in ou.tests:
            nname = renamed_to.get(name, name)
            if nname not in nu.tests:
                problems.append(f'{f}: test "{name[:60]}" missing')
                continue
            if nname != name:
                print(f'   {f:13s} renamed "{name[:70]}" -> "{nname[:70]}"')
            if name in DUMP_TESTS:
                continue
            n_tests += 1
            moved, frees, err = align_test(name, nname, f, ou, nu, oprog, nprog)
            if err:
                problems.append(f'{f}: test "{name[:60]}": {err}')
            if moved:
                n_conv += 1
                n_moved += moved
                print(f'   {f:13s} {moved:3d} strings  "{name[:80]}"')
            if frees:
                n_free += frees
                print(f'   {f:13s} {frees:3d} free(s) through freeKernel  "{nname[:70]}"')
        for name in nu.tests:
            if name not in ou.tests and name not in renamed_to.values():
                problems.append(f'{f}: test "{name[:60]}" added')
            named = test_named(f, nu, name, otext, ntext)
            lits, pin_files = test_literals(nu, name)
            needles = [(o, n, 'count', f'{k[0]} {k[1]}') for k, (o, n) in named.items()]
            for pf in pin_files:
                needles += [(v, v, 'count', f'{pf}: {v[:50]}') for v in n_pins(nprog.embed(pf)) if v]
            needles += [(v, v, 'presence', repr(v[:50])) for v in sorted(lits)]
            for (on, nn, mode, what) in needles:
                for key, (oh, nh) in named.items():
                    if on == nn and oh == nh:
                        continue
                    n_needle += 1
                    co, cn = oh.count(on), nh.count(nn)
                    if (co != cn) if mode == 'count' else ((co > 0) != (cn > 0)):
                        problems.append(f'{f}: test "{name[:40]}": {what} in {key[0]} {key[1]}: found {co} times before, {cn} after')
    print(f'   {n_tests} tests aligned; {n_conv} read {n_moved} strings from files, each the original string; {n_free} kernel frees '
          'through freeKernel')
    print(f'   needles: in every text const a test names, its pin lines and the text consts it names are found exactly as '
          f'often as before, its string literals where they were before ({n_needle} counts where a needle or a text changed)\n')

    print('5. dumps: every TURBOEXL3_METAL_DUMP unit printMetal prints, original vs new')
    n_units = n_same = 0
    cw_units = []
    for name in DUMP_TESTS:
        try:
            oo = dump_outputs(oprog, ounits['prefill.zig'], name)
            no = dump_outputs(nprog, nunits['prefill.zig'], name)
        except (EvalError, AssertionError, KeyError) as e:
            problems.append(f'dump test "{name[:50]}": {e}')
            continue
        if len(oo) != len(no):
            problems.append(f'dump test "{name[:50]}": {len(oo)} units before, {len(no)} after')
        for a, b in zip(oo, no):
            n_units += 1
            n_same += a == b
            unit = a.split(chr(10), 1)[0].replace('@@METAL_BEGIN ', '')
            if a != b:
                if same_but_cw(a, b):
                    cw_units.append(unit)
                else:
                    problems.append(f'dump unit {unit} differs beyond comments and whitespace')
    print(f'   {n_units} units: {n_same} identical, {len(cw_units)} equal but for comments and whitespace'
          + (f' ({", ".join(cw_units)})' if cw_units else '') + '\n')

    # 6. files
    print('6. files: every @embedFile resolves, every .metal file is embedded')
    embedded = {}
    for f, u in nunits.items():
        for k in range(len(u.t) - 3):
            if u.t[k].text == '@embedFile' and u.t[k + 2].kind == 'str':
                full = os.path.normpath(os.path.join(new_dir, u.t[k + 2].value))
                embedded.setdefault(full, []).append(f'{f}:{u.t[k].line}')
                if not os.path.isfile(full):
                    problems.append(f'{f}:{u.t[k].line}: @embedFile("{u.t[k + 2].value}"): no such file')
    metal, corpus = [], []
    for root, _, files in os.walk(new_dir):
        for x in sorted(files):
            if x.endswith('.metal'):
                full = os.path.normpath(os.path.join(root, x))
                metal.append(full)
                corpus.append(read_bytes(full).decode('utf-8'))
                if full not in embedded:
                    problems.append(f'{os.path.relpath(full, new_dir)} is embedded nowhere')
                if not os.path.relpath(full, new_dir).startswith('kernels' + os.sep):
                    problems.append(f'{os.path.relpath(full, new_dir)} lies outside src/kernels')
    dirs = {}
    for m in metal:
        d = os.path.dirname(os.path.relpath(m, new_dir))
        dirs[d] = dirs.get(d, 0) + 1
    print(f'   {len(metal)} .metal files, all embedded: ' + ', '.join(f'{d}/ {n}' for d, n in sorted(dirs.items())) + '\n')

    # 7. no Metal in Zig
    print('7. no Metal in Zig: string literals holding Metal in src/*.zig')
    corpus_s = '\x00'.join(corpus)
    n_exc = 0
    for f in sorted(nunits):
        with open(os.path.join(new_dir, f), encoding='utf-8', newline='') as fh:
            toks = tokenize(fh.read(), f)
        hits = metal_strings(f, toks, corpus_s)
        for (line, kind, v) in hits:
            reason = EXCEPTIONS.get((f, v))
            if reason:
                n_exc += 1
                print(f'   exception {f}:{line} ({"multiline" if kind == "mlstr" else "string"}): {reason}')
            else:
                problems.append(f'{f}:{line}: Metal in a Zig string: {v[:80]!r}')
    print(f'   {n_exc} exception(s); {sum(1 for p in problems if "Metal in a Zig string" in p)} Metal string(s)\n')

    # 8. kernel frees
    print('8. kernel frees: every kernel turboexl3 drops goes through decode.freeKernel (the consumer\'s kernel API)')
    raw = []
    for f in sorted(nunits):
        t = nunits[f].t
        for k in range(len(t) - 2):
            if t[k].text == 'mlx' and t[k + 1].text == '.' and t[k + 2].text == 'mlx_fast_metal_kernel_free':
                raw.append(f'{f}:{t[k].line}')
                problems.append(f'{f}:{t[k].line}: mlx.mlx_fast_metal_kernel_free (mlx-c\'s free, past the consumer\'s launch cache): '
                                'call decode.freeKernel')
    n_calls = sum(1 for u in nunits.values() for k in range(len(u.t) - 1) if u.t[k].text == 'freeKernel' and u.t[k + 1].text == '(')
    print(f'   {n_calls} freeKernel calls; {len(raw)} mlx.mlx_fast_metal_kernel_free calls\n')

    # 9. HIP
    print('9. HIP: the AMD kernels only in src/kernels/hip, listed in src/hip/flags.zig, no HIP text in Zig')
    hip_report(orig_dir, new_dir, problems)

    if problems:
        print('FAIL')
        for p in problems:
            print('  ' + p)
        sys.exit(1)
    print('PASS: every kernel source and header is the original, byte for byte or but for comments and whitespace (listed), '
          'every removal and addition is a listed one, no Metal is left inside Zig, and every kernel free is freeKernel\'s')


main()
