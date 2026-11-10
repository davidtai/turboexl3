#!/usr/bin/env python3
"""Check that turboexl3's Metal lives only in kernels/**.metal files and that a change to it is only what was intended.

usage: python3 tools/check_texts.py ORIG_SRC [NEW_SRC] [-v] [--split]

  ORIG_SRC  the sources before the change (a checkout's src/; for the move out of Zig, the tree with the Metal still
            inside it, e.g. afb52be: `git worktree add /tmp/t3-orig afb52be`, then /tmp/t3-orig/src)
  NEW_SRC   the sources after it (default: this repository's src/)
  -v        every kernel and text const, one line each
  --split   a change that moves declarations between files: checks 1 to 4 over the whole tree (`split_report`)

All on the host, no Zig needed. Both trees are tokenized and their texts evaluated: string literals, `@embedFile`, `++`,
consts, `module.const`, `if` (a comptime condition: only the taken branch, as Zig analyzes it), `&.{ ... }`, labeled blocks
and the files' own functions, with sentinel / replace / replaceOnce / spliceOnce / textSlice / textJoin / pins emulated as
the Zig computes them, and `@hasDecl(sdk, name)` answered as the test build's sdk declares (SDK_DECLS).

Both trees are read whole (every .zig under src/, each @import / @embedFile path relative to its file). Where NEW holds
src/core and ORIG does not, the change is the move into src/core and src/models/<model> (MOVES): ORIG's files are read
under their NEW paths, and an @import / @embedFile path compares by the file it names, so a file that names a moved file
by its new relative path is the original token for token.

0. Moves (the move only). Every ORIG file is at its NEW path (MOVES), every non-Zig file (the .metal, .hip and .hpp
   texts) byte for byte; every NEW file is an ORIG file's image or listed in MOVE_ADDED.

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
6. Files. Every @embedFile path exists, every .metal file is embedded and lies in a kernels/ directory (beside the Zig
   that embeds it: src/core/kernels/<module>/, src/models/<model>/kernels/<module>/).
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

    def __init__(self, name, toks, real=None):
        self.name = name  # the file's path in the NEW tree, relative to src/ (ORIG's files under their moved path, MOVES)
        self.real = real or name  # its path in its own tree (what its relative @import / @embedFile paths start from)
        self.ck = None  # its tokens' keys, an @import / @embedFile path as the NEW file it names (Program.canon_keys)
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
    def __init__(self, src_dir, units, mapper=None):
        self.dir, self.units, self.cache = src_dir, units, {}
        self.mapper = mapper or (lambda rel: rel)  # this tree's path -> the NEW tree's (MOVES for ORIG)
        for u in units.values():
            u.ck = self.canon_keys(u)

    def real_path(self, unit, path):
        """`path` as `unit` names it (@import / @embedFile: relative to the unit's own directory), from src/"""
        return os.path.normpath(os.path.join(os.path.dirname(unit.real), path)).replace(os.sep, '/')

    def canon(self, unit, path):
        """the NEW tree's path of the file `unit` names by `path`"""
        return self.mapper(self.real_path(unit, path))

    def canon_keys(self, unit):
        """the unit's token keys, each @import("x.zig") / @embedFile("...") path as the NEW file it names, so a declaration
        that names a moved file by its new relative path is the original's token for token"""
        t, out = unit.t, []
        for k, x in enumerate(t):
            if x.kind == 'str' and k >= 2 and t[k - 1].text == '(' and (t[k - 2].text == '@embedFile' or
                                                                       (t[k - 2].text == '@import' and x.value.endswith('.zig'))):
                out.append(('path', self.canon(unit, x.value)))
            else:
                out.append(x.key())
        return out

    def unit_of(self, unit, path):
        """the unit `unit` imports by `path`"""
        f = self.canon(unit, path)
        if f not in self.units:
            raise EvalError(f'no unit {f}')
        return self.units[f]

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

    def embed(self, unit, path):
        full = os.path.normpath(os.path.join(self.dir, self.real_path(unit, path)))
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
            return self.p.embed(self.u, p.value)
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
            return self.member(self.p.unit_of(self.u, f), member)
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
                return self.member(self.p.unit_of(self.u, self.u.imports[x.text]), member)
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


def variant_entry(unit, table):
    """{field: (a, b)}: the token span of each field of the first entry of the array const `table` in `unit`
    (`[_]T{ .{ .field = value, ... }, ... }`); None without that const"""
    t = unit.t
    for (key, a, b) in unit.decls:
        if key != f'const {table}':
            continue
        for k in range(a, b - 1):
            if t[k].text == '.' and t[k + 1].text == '{':
                out = {}
                for (s, e) in unit.split_commas(k + 2, unit.match(k + 1)):
                    if e - s >= 4 and t[s].text == '.' and t[s + 2].text == '=':
                        out[t[s + 1].text] = (s + 3, e)
                return out
    return None


def kernel_calls(prog, unit):
    """(name, line, source, header, token index) of every newKernel call in the unit (an EvalError where not evaluable)"""
    t, out = unit.t, []
    # a call's `<v>.<field>` arguments where <v> is an entry of a variant table (VARIANT_TABLES): the first entry's values
    variants = {v: variant_entry(unit, tbl) for ((f, v), tbl) in VARIANT_TABLES.items() if f == os.path.basename(unit.name)}

    def span(a, b):
        if b - a == 3 and t[a].kind == 'id' and t[a + 1].text == '.' and variants.get(t[a].text):
            return variants[t[a].text].get(t[a + 2].text, (a, b))
        return (a, b)
    for k in range(len(t) - 1):
        if t[k].text == 'newKernel' and t[k + 1].text == '(':
            close = unit.match(k + 1)
            args = [span(a, b) for (a, b) in unit.split_commas(k + 2, close)]
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
    return [('id', 'freeKernel')] if os.path.basename(f) == 'decode.zig' else [('id', 'decode'), ('op', '.'), ('id', 'freeKernel')]


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
        if i <= ob and j <= nb and ou.ck[i] == nu.ck[j]:
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


def test_named(f, u, tname, otext, ntext, prog):
    """{key: (ORIG value, NEW value)} of every text const test `tname` of NEW's unit u names (`x` or `module.x`)"""
    a, b = u.tests[tname]
    t, out = u.t, {}
    for k in range(a, b):
        x = t[k]
        if x.kind == 'id' and t[k - 1].text != '.':
            if t[k + 1].text == '.' and x.text in u.imports and t[k + 2].kind == 'id':
                key = (prog.canon(u, u.imports[x.text]), t[k + 2].text)
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
            if t[k - 2].text == '@embedFile' and 'kernels/test/pins/' in t[k].value:
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
                                             'threads) and sharedAdmitted declines; gu8Admitted checks gu8\'s threadgroup memory against Metal\'s 32 KB')
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
# tests a change edited on purpose, by file and test name, with the reason: check 4 lists them in place of aligning
# their bodies token for token (their strings are still checked against the kernels they name)
EDITED_TESTS = {}
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
CHANGED[('root.zig', 'const pf_cache')] = ('16 prefill geometries: DeepSeek-V4.1 alone serves 8 (K3 and K2 routed banks per '
                                          'bank kind, the DSpark stages, the shared expert as a one-expert bank)')
CHANGED[('root.zig', 'const shared_state')] = 'the grouped linears\' cache and its lock'
# the dense decode GEMV at DeepSeek-V4.1's trunk widths: the grouped linear's GEMV on dense's multi-row GEMV, the launch
# shapes per row count, and their tests and sweep (deepseek_gemv_test.zig)
ADDED_KERNELS[('grouped.zig', 'turboexl3_grouped_gemv_d')] = ('the grouped linear\'s GEMV on dense\'s multi-row GEMV '
                                                             '(Grouped.dense_gemv): gemv_head, dense\'s dgemv_x / _2 / _m / '
                                                             '_fold verbatim, gemv_store into the routed planes; one decode '
                                                             'per row chunk where the routed GEMV decodes per row pair')
ADDED_KERNELS[('grouped.zig', 'turboexl3_grouped_gemv_dl')] = 'the same on dense\'s lean codeword loop (dgemv_2, dgemv_j)'
for _key in ('const gemv_d_in', 'const gemv_d_out'):
    ADDED[('grouped.zig', _key)] = 'the dense-GEMV launch\'s operand names'
CHANGED[('grouped.zig', 'const Grouped')] = ('dense_gemv (step 2 on text_gemv_d, the default; the routed GEMV the tests\' arm), '
                                           'bt_set (its tiles per threadgroup), resetCfgs; chunks of 5 rows on the lean text '
                                           'at BT 2 where dense.Dense.wideLean takes the launch')
CHANGED[('dense.zig', 'const Dense')] = ('bt_set / btAt: the GEMV\'s tiles per threadgroup per decode row count (launch shape '
                                       'only), btLaunch (BT 2 on lean chunks in launches of 2048 threadgroups or more at '
                                       'BT 1, btFor\'s elsewhere), resetCfgs; wideLean / leanAt (with Config.dense_lean5, '
                                       'chunks of 5 rows in such launches on the lean text at BT 2), btDec (DEC-ONE\'s '
                                       'tiles: bt_set\'s or btFor\'s, as before)')
CHANGED[('config.zig', 'const Config')] = ('dense_lean5: chunks of 5 rows in GEMV launches of 2048 threadgroups or more at BT 1 '
                                         'on the lean text at BT 2 (launch shape only)')
ADDED[('root.zig', 'fn denseCached')] = 'test seam: the Dense linear runs a shape on, to set its launch shape'
ADDED[('root.zig', 'fn groupedCached')] = 'test seam: the Grouped linearGrouped runs a shape on, to set its launch'
ADDED_FILES['deepseek_gemv_test.zig'] = ('the grouped dense GEMV\'s texts (CPU) and its bytes against the routed GEMV, and the '
                                         'dense decode GEMV\'s launch-shape sweep on the 40-layer trunk chain (GPU)')
for _key in ('const N', 'const wo_a', 'fn ci', 'fn bf16Exact', 'fn nowNs', 'const HostProj', 'fn randProj', 'fn randBf16',
             'const comp_layers', 'const index_layers', 'const ChainLayer', 'const ChainArm', 'fn chainForward'):
    CHANGED[('deepseek_test.zig', _key)] = 'pub: deepseek_gemv_test.zig\'s chains reuse it'
# the banked decode (root.moeClampedBanked): a call's routed pairs over an expert streamer's three slot banks, form 7's
# launches at top-1 with each line that reads a bank's arrays swapped for one that reads the row-slot's bank
# (banked_test.zig)
RECUT[('decode.zig', 'fin7_2')] = ('cut after its expert line (expert_line, then fin7_2): the banked fin7 reads a bank there; '
                                   'text_fin7, text_fin7c and text_gu3sf keep their values (2)')
RECUT[('decode.zig', 'gu3_3')] = ('cut after its weight base (gu3_w, then gu3_3): the banked gu3 reads its bank\'s trellis '
                                  'there; text_gu3 and text_gu3sf keep their values (2)')
for _name, _why in (('turboexl3_prep2b', 'prep2 with its expert line swapped for the slot row and the bank\'s suh '
                                         '(prep2b_expert)'),
                    ('turboexl3_gu3b', 'gu3 with its weight base swapped for the bank\'s trellis at the slot row (gu3b_w); '
                                       'the members compare the packed ids'),
                    ('turboexl3_fin7cb', 'fin7c with its expert line swapped for the slot row and the bank\'s svh / down suh '
                                         '(fin7b_expert)'),
                    ('turboexl3_down3b', 'down3 with its weight base swapped for the bank\'s trellis at the slot row '
                                         '(down3b_w); the members compare the packed ids'),
                    ('turboexl3_dfin3b', 'dfin3 with its expert line swapped for the slot row and the bank\'s down svh '
                                         '(dfin3b_expert)')):
    ADDED_KERNELS[('decode.zig', _name)] = _why
for _key in ('const prep2b_inputs', 'const gu3b_inputs', 'const fin7b_inputs', 'const down3b_inputs', 'const dfin3b_inputs'):
    ADDED[('decode.zig', _key)] = 'the banked launches\' operand names (each banked array once per bank)'
for _key in ('fn validateProjectionAs', 'fn bankGeometryAs'):
    ADDED[('root.zig', _key)] = 'validateClampedProjection\'s and bankGeometry\'s checks with the trellis dtype a parameter'
for _key in ('const banked_rows_max', 'const BankedHalf', 'fn moeClampedBanked', 'fn moeClampedBankedGateUp',
             'fn moeClampedBankedDown', 'const BankedCall', 'fn bankedCall', 'fn bankedGeometry'):
    ADDED[('root.zig', _key)] = 'the banked decode\'s API and its operands\' checks'
CHANGED[('root.zig', 'fn validateClampedProjection')] = 'its checks moved to validateProjectionAs at U16 (the same refusals)'
CHANGED[('root.zig', 'fn bankGeometry')] = 'its body moved to bankGeometryAs at U16 (the same refusals)'
CHANGED[('decode.zig', 'const Kernels3')] += ('; the banked decode: bankedGateUp / bankedDown, their kernels made on first use, '
                                              'their configs form 9\'s launch shape at up to banked_rows_max row-slots '
                                              '(cfgsBanked), dfin3\'s unit scores')
ADDED_FILES['banked_test.zig'] = ('the banked decode\'s tests: its texts\' compositions and its refusals (CPU), each pair '
                                  'against moeClamped on its bank byte for byte and sampled pairs against '
                                  'ref.moeRowClamped (GPU), its timing against moeClamped\'s launches '
                                  '(TURBOEXL3_BANKED_TIME)')
# Ricursive Ripple-65B (src/models/ricursive): two linears of one input in one DEC-ONE launch, the value residual and the
# clamped SwiGLU in dense's prepares, and their tests
ADDED_FILES['ricursive.zig'] = ('Ripple-65B\'s joined DEC-ONE launches (turboexl3_ric_jdec1, _jdec1_blend: dense\'s DEC-ONE '
                                'text with the member\'s suh chosen after its head) and the prepares that form v\'s value '
                                'residual and the MLP\'s clamped SwiGLU (turboexl3_ric_blend_prep, _swiglu_prep, '
                                '_swiglu_prep_sc: dense\'s prep and prep_sc reading x through an accessor)')
ADDED_FILES['ricursive_test.zig'] = ('Ripple-65B\'s tests: linear against format.project at every shape (K3, the head K6); the '
                                     'joined launches, valueLinear and mlp against the host\'s chains word for word')
ADDED[('root.zig', 'const ricursive')] = 'Ripple-65B\'s module (models/ricursive/ricursive.zig)'
CHANGED[('root.zig', 'const shared_state')] = 'registers ricursive.shared_state'

# the split of the model variants out of the core (split-model-variants; check_texts --split)
ADDED_FILES['f3.zig'] = 'F3 (DeepSeek-V4.1\'s shared expert at decode rows), out of dense.zig and root.zig'
ADDED_FILES['banked.zig'] = 'DeepSeek-V4.1\'s banked decode, out of decode.zig\'s Kernels3 (`Banked` beside a geometry\'s Kernels3)'
ADDED_FILES['shared.zig'] = 'Qwen3.8\'s gated shared expert and router in the routed decode launches, out of decode.zig\'s Kernels3 (`SharedHeads` beside a geometry\'s Kernels3)'
ADDED_FILES['models/qwen38/api.zig'] = 'Qwen3.8\'s entry points, out of root.zig (root re-exports each by name)'
ADDED_FILES['models/deepseek_v41/api.zig'] = 'DeepSeek-V4.1\'s entry points, out of root.zig (root re-exports each by name)'
ADDED_FILES['chains_test.zig'] = ('byte tests of the core paths no other test pinned: pg2 against dequantize + matmul + '
                                  'finish, the pieces\' finish against the one-call finish, the unfused chain against the '
                                  'reference')
# GLM-5.3-Flash's fused paths (glm53f-fused): the routed prefill chain over gathered rows (root.moeClampedRows: the prepare
# with a row map, turboexl3_pf_prep_r) and the banked decode with its y stored in a chosen dtype (moeClampedBankedAs)
ADDED_KERNELS[('prefill.zig', 'turboexl3_pf_prep_r')] = ('the prepare with its row line swapped for the row map\'s '
                                                        '(text_prep_rows: prep_1, prep_rmap, prep_2): each slot\'s x row '
                                                        'read through rmap, an expert streamer\'s prompt rows gathered '
                                                        'in the prepare')
ADDED[('prefill.zig', 'const prep_rows_in')] = 'the row-map prepare\'s operand names (the prepare\'s, then rmap)'
CHANGED[('prefill.zig', 'const Prefill')] += ('; prep_r (made on first use, prepRowsKernel), prepWith / moeWith / moeT128With: '
                                              'the chain with each slot\'s x row from a row map (prep / moe / moeT128 call '
                                              'them with none, the same launches)')
ADDED[('root.zig', 'fn moeClampedRows')] = ('moeClamped\'s prefill chain at top-1 over gathered rows (the row-map prepare), '
                                          'y in a chosen dtype: GLM-5.3-Flash\'s prompt calls')
ADDED[('root.zig', 'const moeClampedBankedAs')] = 'the re-export of models/deepseek_v41/api.zig\'s moeClampedBankedAs'
ADDED[('models/deepseek_v41/api.zig', 'fn moeClampedBankedAs')] = ('moeClampedBanked on the banked Kernels3 of a chosen '
                                                                 'output dtype: dfin3b\'s store the one change')
ADDED_FILES['fused_test.zig'] = ('GLM-5.3-Flash\'s fused paths held to the chains they replace byte for byte: the row-map '
                                 'prepare\'s text (CPU), moeClampedRows and moeClampedBankedAs at bf16 (GPU); the plugin\'s '
                                 'schedules held to the default\'s (the banked texts and GEMV variants, the prompt call '
                                 'sizes, the routed prefill\'s bk128 and half_rows; GPU); every pg2 candidate tile at '
                                 'every trunk linear held to the chain\'s words (GPU); their timing (TURBOEXL3_GLM_TIME)')
for _key in ('const bank_experts', 'fn bfExact', 'fn rootBank', 'fn envList', 'fn drawPairs'):
    CHANGED[('routed_test.zig', _key)] = 'pub: fused_test.zig reuses it'
# GLM-5.3-Flash's trunk at prefill rows on pg2 (a host's own tiles, pgemm.host_table from pgemm.candidates by the codes of
# models/glm53f/api.zig) and the routed prefill's bk128 / half_rows options reachable from Config (the plugin's
# TURBOEXL3_GLM53F_PG2 and _PF_OPTS)
ADDED_FILES['models/glm53f/api.zig'] = ('GLM-5.3-Flash\'s trunk linears at prefill rows and the pg2 table a host builds from '
                                        'one tile code a linear (pg2Table), with its CPU tests')
ADDED[('root.zig', 'const glm53f')] = 'GLM-5.3-Flash\'s host helpers (models/glm53f/api.zig)'
ADDED[('pgemm.zig', 'const candidates')] = 'the tiles a host times on its machine to fill host_table (named by index)'
ADDED[('pgemm.zig', 'const host_table')] = 'a host\'s own tiles a (K, N), looked up before table'
ADDED[('pgemm.zig', 'fn tileOf')] = 'the tile a (K, N): the host\'s table first, then table (tileFor\'s lookup)'
CHANGED[('pgemm.zig', 'fn tileFor')] = 'its table lookup moved to tileOf (the host\'s table first)'
CHANGED[('pgemm.zig', 'fn warm')] = 'compiles the host\'s table\'s tiles too'
CHANGED[('config.zig', 'const Config')] += ('; pf_bk128 and pf_half_rows: the routed prefill\'s bk128 and half_rows '
                                            '(prefill.Options, the same bytes) from the configuration')
CHANGED[('root.zig', 'fn pfOptions')] = 'takes bk128 and half_rows from Config.pf_bk128 and pf_half_rows'
# the routed prefill's 128-row tile launches in 128-deep K stages (prefill.Options.t128_bk128, Config.pf_t128_bk128: the
# plugin's TURBOEXL3_GLM53F_PF_OPTS tk and bk128.tk): the same kernels, BKS 128 on those launches (pf_stage_test.zig)
CHANGED[('config.zig', 'const Config')] += ('; pf_t128_bk128: both routed prefill GEMMs\' 128-row tile launches in '
                                            '128-deep K stages (prefill.Options.t128_bk128, the same bytes)')
CHANGED[('prefill.zig', 'const Options')] = ('t128_bk128 (taken beside t128 within 32 KB) and t128Stage, the 128-row tile '
                                             'launch\'s K stage depth')
CHANGED[('prefill.zig', 'const Prefill')] += ('; t128Gemm runs the 128-row tile launch at Options.t128Stage of the mixed '
                                              'launch\'s depth')
CHANGED[('root.zig', 'fn pfOptions')] += '; t128_bk128 from Config.pf_t128_bk128'
CHANGED[('root.zig', 'fn prefillFor')] = ('t128_bk128 off where the 128-row tiles do not run or their 128-deep stages '
                                          'exceed 32 KB; the engage line prints it')
CHANGED[('fused_test.zig', 'const PfOpts')] = 't128_bk128 beside bk128 and half_rows (default off)'
CHANGED[('fused_test.zig', 'const pf_opts')] = 'tk and bk128.tk appended (the plugin\'s TURBOEXL3_GLM53F_PF_OPTS values)'
CHANGED[('fused_test.zig', 'const pf_sets')] = 'tk and bk128.tk appended: the prefill options\' byte and timing tests run them'
CHANGED[('fused_test.zig', 'fn pfOptsConfigure')] = 'sets Config.pf_t128_bk128 too'
ADDED_FILES['pf_stage_test.zig'] = ('where the routed prefill takes its 128-row tile launches\' 128-deep stages and the depth '
                                    'each launch then runs (CPU); their bytes are fused_test.zig\'s prefill options test')
# the mixed launch's outputs at the rows they hold (prefill.Options.t128_zfit, Config.pf_t128_zfit: the plugin's
# TURBOEXL3_GLM53F_PF_ZFIT): the route's compact bases (zmb, the tables' fourth column), the mixed GEMMs storing there, mid
# and finish reading there, root.moeClampedRowsFit naming the call's experts (zfit_test.zig)
CHANGED[('config.zig', 'const Config')] += ('; pf_t128_zfit: beside the 128-row tiles, the mixed launch\'s outputs at the rows '
                                            'they hold (prefill.Options.t128_zfit, the same bytes)')
CHANGED[('prefill.zig', 'const Options')] += '; t128_zfit (taken beside t128)'
CHANGED[('prefill.zig', 'const T128Kernels')] += ('; z (ZKernels): the compact mixed outputs\' route, mixed launches, mid and '
                                                  'finish, made after the stock set with t128_zfit')
CHANGED[('prefill.zig', 'const RouteT128')] = 'zmb (the compact bases, t128_zfit) and zcap (the mixed launch\'s output rows)'
CHANGED[('prefill.zig', 'const ZT128')] = 'zmb, the route\'s compact bases borrowed for mid and finish'
CHANGED[('prefill.zig', 'const Prefill')] += ('; t128_zfit: routeT128 returns the compact bases, t128Gemm sizes the mixed '
                                              'launch\'s outputs at zcap rows, midT128 and finishT128 read through zmb, '
                                              'moeT128With takes call_experts')
for _key in ('fn gemmText', 'fn gemmSmallText', 'fn gemmMixedText'):
    RECOMPOSED[('prefill.zig', _key)] = ('zb: the mixed launches\' stores at the table\'s fourth column '
                                         '(gemm_*op*_xsz_zb.metal in place of gemm_*op*_xsz.metal); every text without it '
                                         'keeps its value (2)')
ADDED[('prefill.zig', 'fn t128ZCap')] = 'the compact mixed outputs\' rows for an S-slot call naming its experts or not'
for _key in ('froute_t128z_out', 'mid_t128z_in', 'fin_t128z_in'):
    ADDED[('prefill.zig', f'const {_key}')] = 'the compact mixed outputs\' kernels\' operand names (zmb)'
CHANGED[('root.zig', 'fn pfOptions')] += '; t128_zfit from Config.pf_t128_zfit'
CHANGED[('root.zig', 'fn prefillFor')] += '; t128_zfit off without the 128-row tiles, printed in the engage line'
CHANGED[('root.zig', 'fn moeClampedRows')] = 'its body moved to clampedRows (no experts named)'
ADDED[('root.zig', 'fn moeClampedRowsFit')] = 'moeClampedRows naming the call\'s experts (t128_zfit\'s compact outputs\' rows)'
ADDED[('root.zig', 'fn clampedRows')] = 'moeClampedRows\' body with the named experts'
for _key in ('const sigma', 'const prompt_slots', 'const act_rows', 'fn bitsOf', 'fn drawCall', 'fn countDiff', 'fn sortBySlot'):
    CHANGED[('fused_test.zig', _key)] = 'pub: zfit_test.zig draws, sorts and compares its calls with them'
for _name, _why in (('turboexl3_pf_t128z_route', 'text_froute_t128 with the compact bases (zmb, the tables\' fourth column) '
                                                  'between its tables and their spare rows (froute_t128z_2)'),
                    ('turboexl3_pf_t128z_gum', 'turboexl3_pf_t128_gum storing at the tables\' fourth column (gemm2mz_text)'),
                    ('turboexl3_pf_t128z_dnm', 'turboexl3_pf_t128_dnm storing at the tables\' fourth column (gemm1mz_text)'),
                    ('turboexl3_pf_t128z_mid', 'turboexl3_pf_t128_mid reading a remainder position at its compact row '
                                               '(mid_t128z_pick, mid_t128z_loads)'),
                    ('turboexl3_pf_t128z_midc', 'turboexl3_pf_t128_midc reading a remainder position at its compact row'),
                    ('turboexl3_pf_t128z_fin', 'turboexl3_pf_t128_fin reading a remainder slot at its compact row '
                                               '(finish_t128z_pick, finish_t128z_load)')):
    ADDED_KERNELS[('prefill.zig', _name)] = 'the compact mixed outputs (t128_zfit): ' + _why
ADDED_FILES['zfit_test.zig'] = ('the compact mixed outputs\' rows hold every remainder the route leaves (CPU), and moeClampedRows '
                                'with t128_zfit, the experts named or not, gives the words it gives without under the served '
                                'options, bk128 and bk128.tk (GPU)')

# the timing tests' 455-row runs: each run's slot computed in u32. @min with a comptime bound narrows its result type (u4
# for 8, u6 for 35), so the run index times the slot stride overflowed it (undefined in the ReleaseFast gpu-test): the
# routing the tests name, 9 experts 11 slots apart and 36 experts 2 apart
for _t in ('turboexl3 gpu glm53f fused timing: each fused path against the chain it replaces at the served shapes '
           '(TURBOEXL3_GLM_TIME=1)',
           'turboexl3 gpu glm53f schedule timing: the banked decode\'s schedules at a verify\'s waves, and a prompt bank call '
           'in calls of 2048 to 16384 rows (TURBOEXL3_GLM_TIME=1)',
           'turboexl3 gpu glm53f prefill options timing: a 16384-row prompt bank call under the served options, bk128, '
           'half_rows and both, in calls of 4096 and 16384 rows on each path, over 36 experts\' 455-row runs and over a '
           'drawn routing (TURBOEXL3_GLM_TIME=1)'):
    EDITED_TESTS[('fused_test.zig', _t)] = 'the 455-row runs\' slots in u32 (the narrowed @min overflowed their product)'

# DEC-ONE's GEMV variants (Config.dense_variant, Config.dec_bt): dense.zig's Dense.init makes dec0 and dec1 from the variant
# the configuration names (`dv`, an entry of `dec_variants`: its kernels' names, texts and header). Check 1 reads those
# calls' `dv.<field>` arguments as the table's first entry's (name "", the default), so the default calls are held to the
# original calls' names, texts and headers; the other entries are kernels of their own names, built only when the
# configuration names them (t4 / t1 / t0: an EXL3_T1V define file before header5; ku / kut4: the ku texts)
VARIANT_TABLES = {('dense.zig', 'dv'): 'dec_variants'}
ADDED[('config.zig', 'const dense_variants')] = 'DEC-ONE\'s GEMV variant count (dense.dec_variants.len, checked there)'
ADDED[('dense.zig', 'const DecVariant')] = 'a DEC-ONE GEMV variant: its name, its two kernels\' names, their texts and header'
ADDED[('dense.zig', 'const dec_variants')] = ('DEC-ONE\'s GEMV variants: the first today\'s dec0 and dec1 (check 1 reads the '
                                             'calls as it), then t4, t1, t0, ku and kut4; g1 and g2 at GLM-5.3-Flash\'s trunk '
                                             'shapes only')
ADDED[('dense.zig', 'comptime #62')] = 'dec_variants.len checked against config.dense_variants'
ADDED[('dense.zig', 'fn decVariantOf')] = 'dec_variants\' index by name (the plugin\'s TURBOEXL3_GLM53F_DEC)'
ADDED_FILES['dense_dec_test.zig'] = ('DEC-ONE\'s schedules at sc_3.10\'s 20 dense shapes, rows 1 to 4, bf16 and f32: each variant '
                                    'and output-tile count against the served configuration\'s bytes (GPU), and a verify\'s '
                                    'trunk timed per class and schedule (TURBOEXL3_GLM_TIME=1)')
EDITED_TESTS[('config.zig', 'turboexl3 config: the defaults are valid and served; an unserved form, slice count or layout is '
              'refused')] = 'DEC-ONE\'s dense_variant and dec_bt: the defaults, every variant and tile count served, others refused'

# wave3 (Config.banked_wave3, banked_wave3.zig; the plugin's TURBOEXL3_GLM53F_BANKED_WAVE3): the whole grouped banked call in
# three launches where five run: prep2bg zeroing the arrival counters (w3_zero before its skip line), then each grouped GEMV
# with its finish run by the last threadgroup to arrive at each finish unit (fin7cbg's statements after the gate|up GEMV,
# dfin3bg's after the down), their grid lines (decode.fin7_1b, dfin3_1b, cut out of fin7_1 and dfin3_1: the same bytes)
# swapped for the unit's. Every other kernel and text is unchanged.
ADDED_FILES['banked_wave3.zig'] = ('wave3\'s texts (turboexl3_prep2bgw3, turboexl3_gu3bgw3 and turboexl3_gu3bgm3w3, '
                                   'turboexl3_down3bgw3 and turboexl3_down3bgm3w3, each grouped GEMV variant\'s too: the '
                                   'grouped GEMVs\' and finishes\' own pieces with the w3_* pieces around them), its launch '
                                   'shape, kernels and configs (Wave3), and the texts\' CPU test')
ADDED_FILES['wave3_test.zig'] = ('wave3\'s option and its exact arrival counts (CPU), and its call against the five launches, '
                                 'word for word, at every SC rate, 1 to 4 rows of top-8 and other waves with skipped '
                                 'row-slots, selected and configured, y f32 and bf16 (GPU)')
for _k, _why in (('fin7_1a', 'fin7_1 to its grid lines: its constants'),
                 ('fin7_1b', 'fin7_1\'s grid lines and lane (banked_wave3.zig\'s gate|up launch takes its unit\'s there)'),
                 ('dfin3_1a', 'dfin3_1 to its grid lines: its constants'),
                 ('dfin3_1b', 'dfin3_1\'s grid lines, lane, thread and exchange (banked_wave3.zig\'s down launch takes its '
                              'unit\'s there, the exchange at kernel scope)'),
                 ('dfin3_1c', 'dfin3_1 past its exchange: the output, the MUL1 constants and the slot loop\'s head')):
    ADDED[('decode.zig', f'const {_k}')] = _why
ADDED[('banked.zig', 'const banked_wave3')] = 'banked_wave3.zig, the whole grouped call in three launches (Banked.wave3)'
CHANGED[('banked.zig', 'const Banked')] = (CHANGED.get(('banked.zig', 'const Banked'), 'Banked') +
                                           '; wave3 (its kernels and configs, freed first in deinit) and bankedMoeSelV, the '
                                           'whole call as the five launches or wave3\'s three (wave3On, wave3Texts)')
CHANGED[('config.zig', 'const Config')] += ('; banked_wave3: the whole grouped banked call in three launches (needs '
                                            'banked_group 4; the same words)')
ADDED[('models/deepseek_v41/api.zig', 'const BankedOuts')] = 'the whole banked call\'s hd, hs and y (moeClampedBankedWave)'
ADDED[('models/deepseek_v41/api.zig', 'fn moeClampedBankedWave')] = ('the whole banked call on a selection and variants as '
                                                                    'five launches or wave3\'s three, or as configured')
ADDED[('models/deepseek_v41/api.zig', 'fn bankedY')] = 'moeClampedBankedWave\'s y as configured (hd and hs freed)'
for _k in ('fn moeClampedBankedAs', 'fn moeClampedBankedSelV'):
    CHANGED[('models/deepseek_v41/api.zig', _k)] = 'the whole call through bankedY: wave3 as configured (Config.banked_wave3)'

# GLM-5.3-Flash's shared expert gate | up in one DEC-ONE launch (glm53f/shared_gu.zig, the plugin's
# TURBOEXL3_GLM53F_SHARED_GU): dense's text_dec1 with the member chosen over two trellises after its head (jsel_pair) and
# ricursive's tile address in the member's trellis (jtile), each member at its own K split, suh and svh
ADDED_FILES['shared_gu.zig'] = ('the shared expert\'s gate | up in one DEC-ONE launch (turboexl3_glm53f_shared_gu: dense\'s '
                                'DEC-ONE pieces with jsel_pair, ricursive\'s jtile and jstore_pair, over the two trellises as '
                                'they lie, into two outputs), its plan, the svh join and its CPU tests')
ADDED_FILES['shared_gu_test.zig'] = ('shared_gu.gateUp against root.linear of each member, word for word, at K6 and K4, decode rows '
                                     '1 to 8 and 16, bf16 and f32 (GPU)')
ADDED_KERNELS[('shared_gu.zig', 'turboexl3_glm53f_shared_gu')] = ('dense\'s text_dec1 with the member chosen over two trellises '
                                                                  'after its head (jsel_pair), the tile read in the member\'s '
                                                                  'trellis (ricursive\'s jtile) and the finish\'s value stored in '
                                                                  'the member\'s own output (jstore_pair)')
CHANGED[('ricursive.zig', 'const jsel')] = 'pub: glm53f/shared_gu.zig\'s text test holds its own member choice beside it'
CHANGED[('ricursive.zig', 'const jtile')] = 'pub: glm53f/shared_gu.zig\'s text reads the member\'s tile through it'
ADDED[('models/glm53f/api.zig', 'const shared_gu')] = 'shared_gu.zig, the shared expert\'s gate | up launch (root\'s glm53f.shared_gu)'
CHANGED[('root.zig', 'const shared_state')] = 'registers glm53f.shared_gu.shared_state'

# the persistent grouped banked GEMVs (Config.banked_ptg, banked_ptg.zig; the plugin's TURBOEXL3_GLM53F_BANKED g4p and
# g4m3p): a threadgroup builds the call's lead list once (each row-slot's turboexl3_members_g), then runs the grouped
# GEMV's work items at its index, its index + NPT, ... on the grouped texts' own bodies (gu3g_3, down3g_3, hw4b's m3 ones)
ADDED_FILES['banked_ptg.zig'] = ('the persistent grouped GEMVs (turboexl3_gu3bgp, turboexl3_down3bgp, hw4b\'s '
                                 'turboexl3_gu3bgm3p and turboexl3_down3bgm3p, each grouped GEMV variant\'s too: gu3_1 and '
                                 'down3_1, the lead list and work item loop (ptg_1 .. ptg_3) around the grouped bodies), '
                                 'their configs at NPT threadgroups and the texts\' CPU test')
ADDED_FILES['ptg_test.zig'] = ('the persistent grouped GEMVs\' option and names (CPU), and each persistent text set at every '
                               'grouped GEMV variant the plugin takes, selected and configured (g4m3p and g4p at t4.t4), '
                               'against the served banked call\'s rows, byte for byte, at every SC rate (GPU)')
ADDED[('banked.zig', 'const banked_ptg')] = 'banked_ptg.zig, the persistent grouped GEMVs (Banked.ptg)'
CHANGED[('banked.zig', 'const Banked')] = ('banked_ptg and ptg: at banked_group 4 the grouped GEMVs as persistent '
                                           'threadgroups (gpSet; the BankedSel and set texts g4p and g4m3p, bankedSetSelV '
                                           'takes their variants from ptg); a set\'s kernels held by value')
CHANGED[('config.zig', 'const Config')] += ('; banked_ptg: at banked_group 4, the grouped banked GEMVs as persistent '
                                            'threadgroups over the call\'s lead list (the same work items, so the same bytes)')
for _key in ('fn drawWave', 'fn varName', 'fn refusedVariant'):
    CHANGED[('fused_test.zig', _key)] = 'pub: ptg_test.zig draws its waves, names its variants and skips the refused ones with them'
EDITED_TESTS[('fused_test.zig', 'turboexl3 gpu glm53f schedule timing: the banked decode\'s schedules at a verify\'s waves, '
              'and a prompt bank call in calls of 2048 to 16384 rows (TURBOEXL3_GLM_TIME=1)')] += (
    '; the persistent texts\' selections (g4p, g4m3p) timed beside g4 and g4m3')

# Two linears of one input at their own K splits in one DEC-ONE launch (glm53f/pair_ks.zig, a DSA layer's q_a and kv_a: the
# plugin's TURBOEXL3_GLM53F_SMALLOP_FUSE): shared_gu's text with its head laying the members out flat at their own K
# splits (ks_head) and dense's prepare's store at the member's slice (ks_prep1_2, dec_prep1 cut at its store)
ADDED_FILES['pair_ks.zig'] = ('two linears of one input at their own K splits in one DEC-ONE launch (turboexl3_glm53f_pair_ks: '
                              'shared_gu\'s pieces with ks_head and dense\'s prepare with ks_prep1_2 for its store), its plan '
                              'and its CPU tests')
ADDED_FILES['pair_ks_test.zig'] = ('pair_ks.linear2 against root.linear of each member, word for word, a DSA layer\'s q_a and '
                                   'kv_a at K6 and K4, decode rows 1 to 8 and 16, bf16 and f32 (GPU)')
ADDED_KERNELS[('pair_ks.zig', 'turboexl3_glm53f_pair_ks')] = ('shared_gu\'s text with the head laying the members out flat '
                                                              'along x at their own K splits (ks_head) and the prepare\'s store '
                                                              'at the member\'s slice (ks_prep1_2)')
CHANGED[('dense.zig', 'const dec_prep1')] = ('cut into dec_prep1_1, dec_prep1_2 (the store of a prepared input at its offset in '
                                            'the slice) and dec_prep1_3: the same bytes')
ADDED[('dense.zig', 'const dec_prep1_1')] = 'dec_prep1 to its store'
ADDED[('dense.zig', 'const dec_prep1_2')] = 'dec_prep1\'s store (glm53f/pair_ks.zig stores at its member\'s slice instead)'
ADDED[('dense.zig', 'const dec_prep1_3')] = 'dec_prep1 after its store'
ADDED[('models/glm53f/api.zig', 'const pair_ks')] = 'pair_ks.zig, two linears at their own K splits in one launch (root\'s glm53f.pair_ks)'
CHANGED[('root.zig', 'const shared_state')] = 'registers glm53f.shared_gu.shared_state and glm53f.pair_ks.shared_state'

# The routed prefill's size classes (Config.pf_size_class, prefill.Options.size_class; the plugin's
# TURBOEXL3_GLM53F_PF_REUSE): moeT128With's row-slot-sized arrays at classRows of the call's row-slots, every launch over
# the call's own rows, y handed back as its first rows' view: the same bytes
ADDED[('prefill.zig', 'const size_class_rows')] = 'the size classes\' step (256 row-slots)'
ADDED[('prefill.zig', 'fn classRows')] = 'a call\'s rows rounded up to a multiple of size_class_rows'
ADDED_FILES['size_class_test.zig'] = ('classRows and the size classes\' rule (CPU), and prompt calls whose row-slot counts are '
                                      'not multiples of 256 with the size classes against the bytes without them, on both '
                                      'paths, over a drawn routing and 455-row runs, at every SC rate (GPU)')
CHANGED[('config.zig', 'const Config')] += ('; pf_size_class: the routed prefill\'s row-slot arrays at size classes (the same '
                                            'bytes)')
CHANGED[('prefill.zig', 'const Options')] += ('; size_class: moeT128With\'s row-slot arrays at classRows (with t128, without '
                                              't128_zfit)')
CHANGED[('prefill.zig', 'const Prefill')] += ('; prepCap, gemmGUT128Cap, midT128Cap, gemmDT128Cap and finishT128Cap: their '
                                              'outputs at a cap of rows, each launch over the call\'s; moeT128With at classRows '
                                              'with size_class, y its first rows\' view')
CHANGED[('root.zig', 'fn pfOptions')] += '; size_class from Config.pf_size_class'
CHANGED[('root.zig', 'fn prefillFor')] += '; size_class off without the 128-row tiles or with t128_zfit, its engage line once'
CHANGED[('root.zig', 'const shared_state')] += ', and pf_class_logged'
ADDED[('root.zig', 'const pf_class_logged')] = 'the size classes\' engage line, once a process'

# The routed prefill chain in the plugin's prompt calls (glm53f/chain_time_test.zig): the tile-aligned cut (the plugin's
# TURBOEXL3_GLM53F_PF_ALIGN) and the chain's timing arms (TURBOEXL3_GLM_TIME)
ADDED_FILES['chain_time_test.zig'] = ('the tile-aligned cut\'s rule (CPU) and its calls against fixed 4096-row calls byte for byte '
                                      '(GPU); the control chain over 455-row runs and a lognormal routing at both cuts, its '
                                      'stages cumulatively and the GEMM probes (TURBOEXL3_GLM_TIME)')


# GLM-5.3-Flash's DEC-ONE GEMV at its trunk shapes (dense.zig's `g1` and `g2` variants, the plugin's
# TURBOEXL3_GLM53F_DEC=g1 / g2; Dense.glmShape: K6 and the MTP layer's K4, MUL1, the sc_3.10 dense shapes, any other shape
# today's texts): text_dec0 with each lane's tile word offsets and shift taken once and the tile and input pointers stepped a
# constant per input tile (dec_g_1, dec_g_2, dec_g_3), its codewords, decodes, FMAs, folds and finish today's pieces; g2's
# GEMV reads the prepared rows as f32 (dec_g_xf_1, dec_g_xf_2), which prep_scf writes (prep_sc's text into an f32 xp: the
# same halves, widened as stored)
ADDED_KERNELS[('dense.zig', 'turboexl3_dense_prep_scf')] = ('prep_sc\'s text and header with its xp an f32 array (DecVariant.xf: '
                                                            'g2), made only for that variant: the same halves, stored widened')
CHANGED[('dense.zig', 'const DecVariant')] = ('glm (the variant serves GLM-5.3-Flash\'s trunk shapes only, Dense.glmShape) and xf '
                                             '(its GEMV reads f32 prepared rows, prep_scf\'s)')
CHANGED[('dense.zig', 'const dec_variants')] = ('g1 and g2: text_dec0_g1 / text_dec0_g2 (text_dec1 where the prepare runs inside), '
                                               'their own kernel names, header5')
CHANGED[('dense.zig', 'const Dense')] += ('; xf / prep_scf: a variant\'s f32 prepared rows (init makes prep_scf, decCfgs stores '
                                        'xp as f32, decLinear launches it in prep_sc\'s place, deinit frees it through '
                                        'freeKernel); glm_shapes / glmShape / variantFor: a glm variant at GLM-5.3-Flash\'s trunk '
                                        'shapes only, today\'s texts at any other (init takes variantFor\'s and prints the '
                                        'engage line once a process)')
CHANGED[('config.zig', 'const dense_variants')] = '8: g1, g2'
ADDED[('dense.zig', 'const dec_g_said')] = 'a glm variant\'s engage line, once a process'
CHANGED[('dense.zig', 'const shared_state')] = 'registers dec_g_said'
EDITED_TESTS[('dense_dec_test.zig', 'turboexl3 glm53f dense decode texts: each variant is today\'s texts with ku\'s tile loop '
              'in place of today\'s and its own kernel names; today\'s texts are today\'s pieces')] = (
    'g1 and g2: their compositions (their own head, loop and tile words, then text_dec0\'s pieces from the codewords on), '
    'g1\'s widening statement today\'s, and their reach (every served shape at its K; not K5, MCG or another model\'s shape)')

# DEC-ONE's arrival counters made on the host (dense.zig's dec_host_zero, the plugin's TURBOEXL3_GLM53F_HOST_ZERO): a call
# whose prepare runs inside its launch, and glm53f's joined launches (pair_ks.linear2, shared_gu.gateUp), take a new array
# of zeros copied from the host where they launch dec_zero over their counters: the same zeros, so the same bytes
ADDED[('dense.zig', 'const dec_host_zero')] = 'host-made arrival counters in place of dec_zero\'s launch (the plugin sets it)'
ADDED[('dense.zig', 'const dec_host_zeroed')] = 'the calls that took host-made counters'
ADDED[('dense.zig', 'const host_counters_max')] = 'the most counters hostCounters makes (a call past it launches dec_zero)'
ADDED[('dense.zig', 'const host_zero_words')] = 'the zeros hostCounters copies'
ADDED[('dense.zig', 'fn hostCounters')] = 'n zeroed counters copied from the host, with dec_host_zero set'
CHANGED[('dense.zig', 'const Dense')] += ('; DecCfg.ncnt (the counters\' length, decCfgs stores it) and decLinear\'s counters '
                                        'from hostCounters where it launched dec_zero')
CHANGED[('dense.zig', 'const shared_state')] += ', dec_host_zero and dec_host_zeroed'
CHANGED[('pair_ks.zig', 'fn linear2')] = 'its counters from dense.hostCounters where it launched dec_zero'
CHANGED[('shared_gu.zig', 'fn gateUp')] = 'its counters from dense.hostCounters where it launched dec_zero'

# Dense.decPrepared: DEC-ONE's GEMV launch alone over rows a caller prepared in a launch of its own (the GLM-5.3-Flash
# plugin's producer texts, which write prep_sc's outputs), and decPrepSpec, what they write
CHANGED[('dense.zig', 'const Dense')] += ('; DecPrep / decPrepSpec (what prep_sc hands DEC-ONE\'s GEMV at a row count) and '
                                        'decPrepared (dec0 at decCfgs\' config over a caller\'s prepared rows)')
ADDED[('root.zig', 'fn decPrepSpec')] = 'Dense.decPrepSpec on the Dense linear takes'
ADDED[('root.zig', 'fn decPrepared')] = 'Dense.decPrepared on the Dense linear takes, its operands checked against the spec'

# int8 gate|up and down (prefill.Options.int8, Config.pf_int8; lossy, off by default): the routed prefill's gate, up and
# down GEMMs on int8 beside the 128-row tiles, 1 a scale each activation 128-block, 2 one scale a row. prep_2 is cut after
# its two H128s (prep_2_1, prep_2_2) and mid_5 after its H128 (mid_5_1, mid_5_2): the same bytes, so the int8 prepare
# (prep_1, the row line, prep_2_1, then prep_i8 or prep_i8t) and mid (the 128-row tiles' mid to mid_5_1, then mid_i8 or
# mid_i8t) are compositions; the GEMM (gemm_i8_1, its op's operands gemm_i8_sel2 or gemm_i8_sel1, gemm_i8_2 on header_i8:
# an mpp matmul2d of int8 per 128-deep block over the trellis decoded to int8 in threadgroup memory, each block folded at
# its rows' scales, or with ROWSCALE one fold) writes ZT128's outputs, and finish is the exact chain's
ADDED_KERNELS[('prefill.zig', 'turboexl3_pf_prep_i8')] = ('the int8 prepare: text_prep to its f16 inputs (prep_2_1), then each '
                                                          '128-block at its own scale as int8 with its block sums (prep_i8)')
ADDED_KERNELS[('prefill.zig', 'turboexl3_pf_prep_i8r')] = 'the int8 prepare over a row map (text_prep_rows\' row line)'
ADDED_KERNELS[('prefill.zig', 'turboexl3_pf_prep_i8t')] = ('the int8 prepare with one scale a row (prep_i8t: the row\'s amax over '
                                                           'its blocks, one threadgroup a row)')
ADDED_KERNELS[('prefill.zig', 'turboexl3_pf_prep_i8tr')] = 'turboexl3_pf_prep_i8t over a row map'
ADDED_KERNELS[('prefill.zig', 'turboexl3_pf_t128_mid_i8')] = ('int8\'s mid: the 128-row tiles\' mid to the down input\'s f16 '
                                                              'values (mid_5_1), then each 128-block at its own scale as int8 '
                                                              'with its block sums (mid_i8)')
ADDED_KERNELS[('prefill.zig', 'turboexl3_pf_t128_mid_i8c')] = 'turboexl3_pf_t128_mid_i8 with the clamped SwiGLU'
ADDED_KERNELS[('prefill.zig', 'turboexl3_pf_t128_mid_i8t')] = ('int8\'s mid with one scale a row (mid_i8t: the row\'s amax over '
                                                               'its blocks, one threadgroup a row)')
ADDED_KERNELS[('prefill.zig', 'turboexl3_pf_t128_mid_i8tc')] = 'turboexl3_pf_t128_mid_i8t with the clamped SwiGLU'
ADDED_KERNELS[('prefill.zig', 'turboexl3_pf_gemm_i8')] = ('the int8 gate|up GEMMs (gemm_i8_1, gemm_i8_sel2, gemm_i8_2 on '
                                                          'header_i8: the register path\'s header and the int8 decode, '
                                                          'i8_helpers)')
ADDED_KERNELS[('prefill.zig', 'turboexl3_pf_gemm_i8d')] = 'the int8 down GEMM (gemm_i8_1, gemm_i8_sel1, gemm_i8_2 on header_i8)'
ADDED[('prefill.zig', 'const prep_i8_out')] = 'the int8 prepare\'s outputs (xqg, xqu, inv, xsg, xsu, sxg, sxu)'
ADDED[('prefill.zig', 'const gemm_i8_in')] = 'the int8 gate|up GEMMs\' inputs (tiles, xq0, xq1, w0, w1, sx0, sx1, xs0, xs1)'
ADDED[('prefill.zig', 'const gemm_i8d_in')] = 'the int8 down GEMM\'s inputs (tiles, xq0, w0, sx0, xs0)'
ADDED[('prefill.zig', 'const mid_i8_out')] = 'int8\'s mid\'s outputs (xdq, xsd, sxd)'
ADDED[('prefill.zig', 'const I8Kernels')] = ('the int8 chain\'s kernels (both schemes\' prepares, row-map prepares and mids, the '
                                             'gate|up and down GEMMs)')
CHANGED[('prefill.zig', 'const Options')] += '; int8 (1 a scale each activation 128-block, 2 one scale a row), taken beside t128'
CHANGED[('prefill.zig', 'const Prefill')] += ('; i8_k (made with Options.int8, freed in deinit; init refuses int8 2 past 32 blocks a '
                                            'row), prepI8 / prepI8Cap, gemmGUI8 / gemmGUI8Cap, midI8 / midI8Cap, gemmDI8 / '
                                            'gemmDI8Cap, i8Cfg and moeI8Cap; moeT128With runs moeI8Cap with int8')
CHANGED[('config.zig', 'const Config')] += '; pf_int8: the routed prefill\'s int8 gate|up and down (prefill.Options.int8), 0 by default'
CHANGED[('root.zig', 'fn pfOptions')] += '; int8 from Config.pf_int8'
CHANGED[('root.zig', 'fn prefillFor')] += '; int8 off without the 128-row tiles, the deferred map or with t128_zfit; its engage line'
ADDED[('root.zig', 'const pf_i8_logged')] = 'the int8 chain\'s engage line, once a process'
CHANGED[('root.zig', 'const shared_state')] += ' and pf_i8_logged'
ADDED[('fused_test.zig', 'fn pfI8Configure')] = 'configure with Config.pf_int8 set and the size classes, every engine released first'
ADDED[('fused_test.zig', 'fn pfI8Restore')] = 'the saved configuration back, every engine released first'
ADDED[('fused_test.zig', 'fn guStage')] = 'the route, prepare and gate|up launches alone (the int8 timing test\'s stage)'
ADDED[('fused_test.zig', 'fn dnStage')] = 'the launches through mid and down (the int8 timing test\'s second stage)'
ADDED[('fused_test.zig', 'test turboexl3 glm53f prefill int8 options: int8 1 or 2 rides the 128-row tiles on the deferred MUL1 map, the mixed outputs at their sorted positions, beside the size classes')] = 'int8\'s option rule (CPU)'
ADDED[('fused_test.zig', 'test turboexl3 gpu glm53f prefill int8: the routed gate|up and down on int8 (a scale each activation 128-block; one scale a row) agree with the exact chain (cosine >= 0.99, error rms <= 0.1 of y\'s, finite), 16384 row-slots in 4096-row calls with the size classes, on the plugin\'s default path and on moeClampedRows, at every SC rate (TURBOEXL3_GPU=1)')] = 'int8\'s agreement with the exact chain, both schemes (GPU)'
ADDED[('fused_test.zig', 'test turboexl3 gpu glm53f prefill int8 timing: a 16384-row prompt bank call with the routed gate|up and down on int8 (1: a scale each activation 128-block; 2: one scale a row) against the exact chain, in calls of 4096 and 16384 rows on each path with the size classes, and its route, prepare and gate|up alone and through mid and down, over 36 experts\' 455-row runs and over a drawn routing (TURBOEXL3_GLM_TIME=1)')] = 'int8\'s timing against the exact chain (TURBOEXL3_GLM_TIME)'


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


def hip_root(src):
    """(flags.zig, kernels/hip) of a tree"""
    return os.path.join(src, 'hip', 'flags.zig'), os.path.join(src, 'kernels', 'hip')


def hip_files(src):
    root = hip_root(src)[1]
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
    flags_path = hip_root(new_dir)[0]
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

# ------------------------------------------------------------------------------------------------------------- moves

# The move of src/ into src/core and src/models/<model> (organize-by-model): each ORIG path's NEW path, the first rule
# whose ORIG prefix matches (a rule without a trailing / is one file). Each Zig file's Metal sits in a kernels/ directory
# beside it, so a moved file's own @embedFile paths are unchanged; an @import or @embedFile across folders names the
# same file by a new relative path, which Program.canon_keys compares by the file it names.
MOVES = [
    ('root.zig', 'root.zig'),
    ('tests.zig', 'tests.zig'),
    # Qwen3.8-Flash-Next: the gated attention's projections (ATTN-IN / ATTN-OUT), the decode-row merges, the GDN tail
    ('attn_io.zig', 'models/qwen38/attn_io.zig'),
    ('attn_io_test.zig', 'models/qwen38/attn_io_test.zig'),
    ('decb.zig', 'models/qwen38/decb.zig'),
    ('decb_test.zig', 'models/qwen38/decb_test.zig'),
    ('gdn_out.zig', 'models/qwen38/gdn_out.zig'),
    ('gdn_out_test.zig', 'models/qwen38/gdn_out_test.zig'),
    ('kernels/decb/had128.metal', 'core/kernels/sources/had128.metal'),
    ('kernels/attn_io/', 'models/qwen38/kernels/attn_io/'),
    ('kernels/decb/', 'models/qwen38/kernels/decb/'),
    ('kernels/gdn_out/', 'models/qwen38/kernels/gdn_out/'),
    ('kernels/test/pins/attn_io_', 'models/qwen38/kernels/test/pins/attn_io_'),
    ('kernels/test/pins/decb_', 'models/qwen38/kernels/test/pins/decb_'),
    ('kernels/test/pins/gdn_out.metal', 'models/qwen38/kernels/test/pins/gdn_out.metal'),
    # DeepSeek-V4.1: the grouped wo_a, the trunk, F3, the clamped SwiGLU at its geometry, the banked decode
    ('grouped.zig', 'models/deepseek_v41/grouped.zig'),
    ('deepseek_test.zig', 'models/deepseek_v41/deepseek_test.zig'),
    ('deepseek_gemv_test.zig', 'models/deepseek_v41/deepseek_gemv_test.zig'),
    ('f3_test.zig', 'models/deepseek_v41/f3_test.zig'),
    ('clamped_test.zig', 'models/deepseek_v41/clamped_test.zig'),
    ('banked_test.zig', 'models/deepseek_v41/banked_test.zig'),
    ('banked_grp_test.zig', 'models/deepseek_v41/banked_grp_test.zig'),
    ('banked_ilp_test.zig', 'models/deepseek_v41/banked_ilp_test.zig'),
    ('kernels/grouped/', 'models/deepseek_v41/kernels/grouped/'),
    ('kernels/test/pins/grouped_gemv_d.metal', 'models/deepseek_v41/kernels/test/pins/grouped_gemv_d.metal'),
    ('kernels/test/pins/decode_banked.metal', 'models/deepseek_v41/kernels/test/pins/decode_banked.metal'),
    # sources.zig's texts (more than one kernel splices them)
    ('kernels/decode/gate_gemv.metal', 'core/kernels/sources/gate_gemv.metal'),
    ('kernels/decode/gate_sum.metal', 'core/kernels/sources/gate_sum.metal'),
    # the AMD backend stays where it is (an outside consumer vendors these paths)
    ('hip/', 'hip/'),
    ('hip_tests.zig', 'hip_tests.zig'),
    ('kernels/hip/', 'kernels/hip/'),
    # the shared core: every other file
    ('kernels/', 'core/kernels/'),
    ('', 'core/'),
]
# files the move adds that are no ORIG file's image (not code: checked by nothing else)
MOVE_ADDED = ('models/qwen38/README.md', 'models/deepseek_v41/README.md')


def moved_path(rel):
    """ORIG path `rel` (from src/) -> its NEW path, by MOVES"""
    for (a, b) in MOVES:
        if (a.endswith('/') or a == '') and rel.startswith(a):
            return b + rel[len(a):]
        if rel == a:
            return b
    return rel


def tree_files(d, exts=None):
    """every file under d (relative, '/'-separated, sorted), those ending in one of exts when given"""
    out = []
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if not x.startswith('.') and x not in ('zig-out', '.zig-cache')]
        for f in files:
            if f.startswith('.') or (exts and not f.endswith(exts)):
                continue
            out.append(os.path.relpath(os.path.join(root, f), d).replace(os.sep, '/'))
    return sorted(out)


def moves_report(orig_dir, new_dir, problems):
    """every ORIG file exists at its NEW path, every non-Zig file byte for byte (a Zig file is checked by 1 to 9); every
    NEW file is an ORIG file's image or listed in MOVE_ADDED"""
    ofiles, nfiles = tree_files(orig_dir), set(tree_files(new_dir))
    images, by_ext, same, kept = set(), {}, 0, 0
    for rel in ofiles:
        new = moved_path(rel)
        images.add(new)
        if new not in nfiles:
            problems.append(f'{rel}: not at its NEW path {new}')
            continue
        kept += new == rel
        if not rel.endswith('.zig'):
            if read_bytes(os.path.join(orig_dir, rel)) == read_bytes(os.path.join(new_dir, new)):
                same += 1
            else:
                problems.append(f'{rel} -> {new}: bytes differ')
        top = new.split('/')[0] if new.split('/')[0] != 'models' else '/'.join(new.split('/')[:2])
        by_ext.setdefault((top, os.path.splitext(rel)[1]), 0)
        by_ext[(top, os.path.splitext(rel)[1])] += 1
    extra = sorted(nfiles - images)
    for f in extra:
        if f not in MOVE_ADDED:
            problems.append(f'{f}: in NEW, no ORIG file\'s image and not in MOVE_ADDED')
    print(f'   {len(ofiles)} ORIG files: {kept} at the same path, {len(ofiles) - kept} moved; '
          f'{same} non-Zig files byte for byte at their new path')
    print('   ' + ', '.join(f'{top}/ {n} {ext}' for (top, ext), n in sorted(by_ext.items())))
    print(f'   added: {", ".join(f for f in extra if f in MOVE_ADDED) or "none"}\n')


def path_of(base, units):
    """the unit path whose file name is `base` (basenames are unique in src/), else `base`"""
    hits = [f for f in units if os.path.basename(f) == base]
    return hits[0] if len(hits) == 1 else base


def rekey_tables(units):
    """the tables of earlier changes name files by their name in src/: key them by the file's path"""
    for tbl in (RECUT, RECOMPOSED, REMOVED, ADDED, CHANGED, ADDED_KERNELS, RENAMED_TESTS, EDITED_TESTS):
        items = list(tbl.items())
        tbl.clear()
        tbl.update({(path_of(f, units), x): v for ((f, x), v) in items})
    items = list(ADDED_FILES.items())
    ADDED_FILES.clear()
    ADDED_FILES.update({path_of(f, units): v for (f, v) in items})


# ------------------------------------------------------------------------------------------------------------- split

# --split against a base: the kernels whose texts a change means to change, the kernels it adds, and the text consts
# whose values it changes, each with what changed (anything else that differs fails). They hold one change's entries
# while it is checked against its base and are emptied after it lands, so a later run allows nothing by them (aa0222f's
# joint layout used them: turboexl3_ric_jdec1 and _blend changed, the two interleaved test references added)
SPLIT_CHANGED_KERNELS = {}
SPLIT_ADDED_KERNELS = {}
SPLIT_CHANGED_CONSTS = {}

def split_report(oprog, nprog, ounits, nunits, orig_dir, new_dir, problems, verbose):
    """--split (declarations moved between files, not only files): checks 1 to 4 over the whole tree instead of per file.
    1. Every kernel name either tree can make (tools/kernel_docs.py's list: the computed-name families included) is in
       both, with the same set of source and header texts. 2. Every text const keeps its value under its name, in
       whatever file it now lives. 3. Declarations and 4. tests are reported (moved, changed), not compared: the proof
       of a split is 1 and 2, the suite and the doc check."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('kernel_docs', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernel_docs.py'))
    kd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kd)
    print('1. kernels (split): every kernel name, its source and header texts, across the whole tree')
    ok_ = kd.kernels(orig_dir)
    nk = kd.kernels(new_dir)

    def texts(calls):
        return {(str(c['source']), str(c['header'])) for c in calls}

    same = 0
    for name in sorted(set(ok_) | set(nk)):
        if name not in nk:
            problems.append(f'kernel {name}: removed')
            continue
        if name not in ok_:
            if name in SPLIT_ADDED_KERNELS:
                print(f'   added (intended) {name}: {SPLIT_ADDED_KERNELS[name]}')
            else:
                problems.append(f'kernel {name}: added')
            continue
        to, tn = texts(ok_[name]), texts(nk[name])
        fo = sorted({c['file'] for c in ok_[name]})
        fn = sorted({c['file'] for c in nk[name]})
        if to != tn and name in SPLIT_CHANGED_KERNELS:
            print(f'   changed (intended) {name}: {SPLIT_CHANGED_KERNELS[name]}')
        elif to != tn:
            problems.append(f'kernel {name}: its texts differ ({len(to)} before, {len(tn)} after)')
        else:
            same += 1
        if verbose or fo != fn:
            print(f'   {name:36s} {",".join(fo)} -> {",".join(fn)}{"" if to == tn else "  DIFFERS"}')
    print(f'   {len(ok_)} kernel names before, {len(nk)} after: {same} with every source and header identical\n')

    print('2. text consts (split): every text const keeps its value under its name, wherever it lives')

    def consts(units, prog):
        out = {}
        for f, u in units.items():
            for c in u.consts:
                try:
                    v = prog.const(u, c)
                except EvalError:
                    continue
                if isinstance(v, str):
                    out.setdefault(c, {})[f] = str(v)
        return out

    oc, nc = consts(ounits, oprog), consts(nunits, nprog)
    moved = kept = 0
    for name, where in sorted(oc.items()):
        for f, v in where.items():
            if name in nc and f in nc[name] and nc[name][f] == v:
                kept += 1
            elif name in nc and v in nc[name].values():
                moved += 1
                g = next(g for g, w in nc[name].items() if w == v)
                if verbose:
                    print(f'   moved {name}: {f} -> {g}')
            elif name in nc and f in nc[name] and name in SPLIT_CHANGED_CONSTS:
                print(f'   changed (intended) {f} {name}: {SPLIT_CHANGED_CONSTS[name]}')
            elif name in nc and f in nc[name]:
                problems.append(f'{f} {name}: value differs')
            else:
                print(f'   gone  {f} {name} (its value no NEW const of that name holds: checked through the kernels in 1)')
    print(f'   {kept} in place, {moved} moved with their values\n')

    print('3. declarations (split): reported')
    od = {(f, k): [x for x in ounits[f].ck[a:b]] for f in ounits for (k, a, b) in ounits[f].decls}
    nd = {(f, k): [x for x in nunits[f].ck[a:b]] for f in nunits for (k, a, b) in nunits[f].decls}
    same = sum(1 for k in od if k in nd and od[k] == nd[k])
    print(f'   {len(od)} before, {len(nd)} after: {same} token for token in place; the rest moved or changed\n')

    print('4. tests (split): every test name kept')
    ot = {t for u in ounits.values() for t in u.tests}
    nt = {t for u in nunits.values() for t in u.tests}
    for t in sorted(ot - nt):
        problems.append(f'test "{t[:70]}" missing')
    print(f'   {len(ot)} tests before, {len(nt)} after, {len(nt - ot)} added\n')


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    verbose = '-v' in sys.argv[1:]
    split = '--split' in sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(2)
    orig_dir = os.path.abspath(args[0])
    new_dir = os.path.abspath(args[1]) if len(args) > 1 else os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
    print(f'ORIG {orig_dir}\nNEW  {new_dir}\n')
    problems = []

    def load(d, mapper):
        """every .zig under d, keyed by its path in the NEW tree"""
        units = {}
        for rel in tree_files(d, ('.zig',)):
            with open(os.path.join(d, rel), encoding='utf-8', newline='') as fh:
                units[mapper(rel)] = Unit(mapper(rel), tokenize(fh.read(), rel), real=rel)
        return units

    # 0. moves: ORIG's files at their NEW paths (MOVES), when the change is the move into src/core and src/models
    moving = os.path.isdir(os.path.join(new_dir, 'core')) and not os.path.isdir(os.path.join(orig_dir, 'core'))
    omap = moved_path if moving else (lambda rel: rel)
    if moving:
        print('0. moves: every file of ORIG at its NEW path (MOVES), every kernel file byte for byte')
        moves_report(orig_dir, new_dir, problems)
    ounits, nunits = load(orig_dir, omap), load(new_dir, lambda rel: rel)
    oprog, nprog = Program(orig_dir, ounits, omap), Program(new_dir, nunits)
    rekey_tables(nunits)
    added_files = []
    for f in sorted(set(ounits) ^ set(nunits)):
        if f in nunits and f in ADDED_FILES:
            added_files.append(f)
        else:
            problems.append(f'{f}: only in {"ORIG" if f in ounits else "NEW"}')

    if split:
        split_report(oprog, nprog, ounits, nunits, orig_dir, new_dir, problems, verbose)
    else:
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
                        kind = 'splice helper' if key in helpers else ('dump harness (5)' if (os.path.basename(f), key) == ('prefill.zig', 'const MetalArg') else None)
                    removed.append((f, key, kind))
                    if kind is None:
                        problems.append(f'{f}: {key} removed')
                    continue
                ot = ou.ck[od[key][0]:od[key][1]]
                nt = nu.ck[nd[key][0]:nd[key][1]]
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
                elif key == 'fn printMetal' and os.path.basename(f) == 'prefill.zig':
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
                if (f, nname) in EDITED_TESTS:
                    print(f'   {f:13s} edited   "{nname[:70]}": {EDITED_TESTS[(f, nname)]}')
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
                named = test_named(f, nu, name, otext, ntext, nprog)
                lits, pin_files = test_literals(nu, name)
                needles = [(o, n, 'count', f'{k[0]} {k[1]}') for k, (o, n) in named.items()]
                for pf in pin_files:
                    needles += [(v, v, 'count', f'{pf}: {v[:50]}') for v in n_pins(nprog.embed(nu, pf)) if v]
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
            oo = dump_outputs(oprog, ounits[path_of('prefill.zig', nunits)], name)
            no = dump_outputs(nprog, nunits[path_of('prefill.zig', nunits)], name)
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
                full = os.path.normpath(os.path.join(new_dir, nprog.real_path(u, u.t[k + 2].value)))
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
                if 'kernels' not in os.path.relpath(full, new_dir).split(os.sep)[:-1]:
                    problems.append(f'{os.path.relpath(full, new_dir)} lies outside a kernels/ directory')
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
            reason = EXCEPTIONS.get((os.path.basename(f), v))
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


if __name__ == '__main__':
    main()
