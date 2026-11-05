#!/usr/bin/env python3
"""Every kernel turboexl3 can build, from the code, and the check that each has its entry in the docs.

usage: python3 tools/kernel_docs.py [--list | --write | --check] [SRC]

  --list   every kernel name `ks.newKernel` can make: its file, line, operands and the .metal pieces of its source and
           header (the default)
  --write  rewrite the generated block of every entry in the docs from the code, and add a skeleton entry (prose fields
           TODO) for each kernel that has none
  --check  fail when a kernel has no entry, an entry names no kernel, a generated block is not the code's, or a prose
           field is missing or still TODO
  SRC      the sources (default: this repository's src/)

All on the host, no Zig: check_texts.py's evaluator computes each newKernel call's source and header, here with
provenance, so each text is the list of .metal files (and computed strings) it is the concatenation of.

Where the entries live: a kernel made in src/core/** is documented in docs/core/kernels.md, one made in
src/models/<model>/** in docs/models/<model>/kernels.md. An entry is a heading `### `<name>`` (a family of computed
names: `### `<prefix>_<variant>`` with `family:` in its block), then the generated block between
`<!-- kernel-facts: <name> -->` and `<!-- /kernel-facts -->` (never edited by hand), then the prose fields, one bullet
each: PROSE_FIELDS.
"""
import hashlib
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..'))
_spec = importlib.util.spec_from_file_location('check_texts', os.path.join(HERE, 'check_texts.py'))
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)

PROSE_FIELDS = ('Used by', 'Shapes and dtypes', 'Grid and threadgroup', 'Template args', 'Replaces', 'Bits', 'Pinned by')


# ------------------------------------------------------------------------------------------------- texts with provenance

class P(str):
    """a string with the pieces it is the concatenation of: ('file', path from src/) or ('text', a computed string)"""
    parts = ()


def pstr(s, parts):
    p = P(s)
    p.parts = tuple(parts)
    return p


def parts_of(s):
    if isinstance(s, P):
        return s.parts
    return (('text', s),) if s else ()


def pjoin(a, b):
    return pstr(str(a) + str(b), parts_of(a) + parts_of(b))


_orig_embed = ct.Program.embed


def _embed(self, unit, path):
    return pstr(_orig_embed(self, unit, path), (('file', self.canon(unit, path)),))


def _expr(self):
    """check_texts' Eval.expr, with `++` keeping the pieces"""
    if self.peek().text == 'comptime':
        self.take()
        return self.expr()
    if self.peek().text == 'if':
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
        if isinstance(v, str) and isinstance(w, str):
            v = pjoin(v, w)
        elif type(v) is type(w):
            v = v + w
        else:
            raise ct.EvalError('++ of mixed types')
    return v


ct.Program.embed = _embed
ct.Eval.expr = _expr
ct.NATIVES['sentinel'] = lambda s: s
_join = ct.NATIVES['textJoin']


def _text_join(parts):
    out = pstr('', ())
    for p in parts:
        out = pjoin(out, p)
    return out


ct.NATIVES['textJoin'] = _text_join


def piece_list(s):
    """the pieces of a text, adjacent computed strings merged: [('file', path) | ('text', s)]"""
    out = []
    for kind, v in parts_of(s):
        if kind == 'text' and out and out[-1][0] == 'text':
            out[-1] = ('text', out[-1][1] + v)
        else:
            out.append((kind, v))
    return out


# -------------------------------------------------------------------------------------------------------- the kernels

def sha(s):
    return hashlib.sha256(str(s).encode('utf-8')).hexdigest()[:12]


def load(src):
    units = {}
    for rel in ct.tree_files(src, ('.zig',)):
        with open(os.path.join(src, rel), encoding='utf-8', newline='') as fh:
            units[rel] = ct.Unit(rel, ct.tokenize(fh.read(), rel), real=rel)
    return units, ct.Program(src, units)


def names_of(prog, unit, a, b, seen=None):
    """the strings an operand-name list expression names: literals, and the consts (arrays, `++` of arrays) it refers to"""
    t, out = unit.t, []
    seen = seen or set()
    k = a
    while k < b:
        x = t[k]
        if x.kind == 'str':
            out.append(x.value)
        elif x.kind == 'id' and t[k + 1].text == '.' and k + 2 < b and t[k + 2].kind == 'id' and x.text in unit.imports:
            other = prog.unit_of(unit, unit.imports[x.text])
            if t[k + 2].text in other.consts and (other.name, t[k + 2].text) not in seen:
                seen.add((other.name, t[k + 2].text))
                ca, cb = other.consts[t[k + 2].text]
                out += names_of(prog, other, ca, cb, seen)
            k += 3
            continue
        elif x.kind == 'id' and (x.text in unit.consts or x.text in unit.locals) and t[k - 1].text != '.' \
                and (unit.name, x.text) not in seen and x.text not in ('u8', 'const'):
            seen.add((unit.name, x.text))
            ca, cb = unit.consts.get(x.text) or unit.locals[x.text]
            out += names_of(prog, unit, ca, cb, seen)
        k += 1
    return out


def expr_text(unit, a, b):
    out = ''
    for x in unit.t[a:b]:
        if out and (out[-1].isalnum() or out[-1] == '_') and (x.text[:1].isalnum() or x.text[:1] in '_@'):
            out += ' '
        out += x.text
    return out


# Kernels whose names are computed (a comptime table indexed at run time): each family's names and headers, read from
# the tables the code builds them from. Key: (file name, the name argument as written).
def _group_family(prog, unit, prefix):
    """decode.zig's grouped-GEMV variants: `<prefix>` at variant 0, `<prefix>_<name>` at each other variant, each on
    `group_variants[i].defines ++ header_group`"""
    a, b = unit.consts['group_variants']
    t = unit.t
    variants = []
    for k in range(a, b):
        if t[k].text == '.' and t[k + 1].text == 'name' and t[k + 2].text == '=':
            name = t[k + 3].value
            d = next(m for m in range(k, b) if t[m].text == 'defines')
            variants.append((name, t[d + 2].value))
    header_group = prog.const(unit, 'header_group')
    out = []
    for (name, defines) in variants:
        out.append((prefix if not name else f'{prefix}_{name}', pjoin(pstr(defines, (('text', defines),)), header_group) if defines else header_group))
    return out


def _probe_family(prog, unit):
    """banked_ilp_test.zig's t-route probes: turboexl3_t1v_probe_<route>_<dtype>, each on `#define EXL3_T1V <route>`
    ++ decode.header"""
    t = unit.t
    a, b = unit.consts['t_routes']
    routes = [x.text for x in t[a:b] if x.kind == 'num']
    a, b = unit.consts['t_dtypes']
    dts = [x.value for x in t[a:b] if x.kind == 'str']
    header = prog.unit_of(unit, unit.imports['decode']).name
    hdr = prog.const(prog.units[header], 'header')
    out = []
    for r in routes:
        d = f'#define EXL3_T1V {r}\n'
        for dt in dts:
            out.append((f'turboexl3_t1v_probe_{r}_{dt}', pjoin(pstr(d, (('text', d),)), hdr)))
    return out


# value: (the entry's heading, the names and headers)
FAMILIES = {
    ('decode.zig', 'group_gu_m3_names[v].ptr'): ('turboexl3_gu3bgm3_<variant>', lambda p, u: _group_family(p, u, 'turboexl3_gu3bgm3')),
    ('decode.zig', 'group_gu_names[v].ptr'): ('turboexl3_gu3bg_<variant>', lambda p, u: _group_family(p, u, 'turboexl3_gu3bg')),
    ('decode.zig', 'group_down_m3_names[v].ptr'): ('turboexl3_down3bgm3_<variant>', lambda p, u: _group_family(p, u, 'turboexl3_down3bgm3')),
    ('decode.zig', 'group_down_names[v].ptr'): ('turboexl3_down3bg_<variant>', lambda p, u: _group_family(p, u, 'turboexl3_down3bg')),
    ('banked_ilp_test.zig', 't_probe_names[ri][j].ptr'): ('turboexl3_t1v_probe_<route>_<dtype>', lambda p, u: _probe_family(p, u)),
}


def kernels(src):
    """{name: [call, ...]}: each call {file, line, decl, inputs, outputs, source (P), header (P), source_expr,
    header_expr, family}"""
    units, prog = load(src)
    out = {}
    for f in sorted(units):
        u = units[f]
        t = u.t
        for k in range(len(t) - 1):
            if not (t[k].text == 'newKernel' and t[k + 1].text == '('):
                continue
            close = u.match(k + 1)
            args = u.split_commas(k + 2, close)
            vals = []
            for (a, b) in (args[3], args[4]):
                if b - a >= 3 and t[b - 2].text == '.' and t[b - 1].text == 'ptr':
                    b -= 2  # `text.ptr`: the text
                if b - a == 1 and t[a].kind == 'id' and t[a].text not in u.consts:
                    try:
                        vals.append(prog.const(u, t[a].text, local_ok=True))
                    except ct.EvalError as e:
                        vals.append(e)
                else:
                    vals.append(ct.value_of(prog, u, a, b))
            ins = names_of(prog, u, *args[1])
            outs = names_of(prog, u, *args[2])
            base = dict(file=f, line=t[k].line, decl=ct.enclosing(u, k), inputs=ins, outputs=outs,
                        source_expr=expr_text(u, *args[3]), header_expr=expr_text(u, *args[4]), family=None)
            name_tok = t[args[0][0]]
            if name_tok.kind == 'str' and args[0][1] - args[0][0] == 1:
                out.setdefault(name_tok.value, []).append(dict(base, source=vals[0], header=vals[1]))
                continue
            key = (os.path.basename(f), expr_text(u, *args[0]))
            if key not in FAMILIES:
                raise SystemExit(f'{f}:{t[k].line}: a computed kernel name ({key[1]}) that FAMILIES does not list')
            heading, make = FAMILIES[key]
            for (name, header) in make(prog, u):
                out.setdefault(name, []).append(dict(base, source=vals[0], header=header, family=heading))
    return out


# ---------------------------------------------------------------------------------------------------------------- docs

def doc_of(file):
    """the docs file a kernel made in `file` (path from src/) is documented in"""
    parts = file.split('/')
    if parts[0] == 'models' and len(parts) > 2:
        return f'docs/models/{parts[1]}/kernels.md'
    return 'docs/core/kernels.md'


def fmt_pieces(s):
    if isinstance(s, ct.EvalError) or s is None:
        return f'(not evaluable: {s})'
    ps = piece_list(s)
    shown = []
    for kind, v in ps:
        if kind == 'file':
            p = v
            for pre in ('core/kernels/', 'kernels/'):
                if p.startswith(pre):
                    p = p[len(pre):]
                    break
            p = re.sub(r'^models/([^/]+)/kernels/', r'\1:', p)
            shown.append(f'`{p[:-6] if p.endswith(".metal") else p}`')
        else:
            shown.append(f'text {v!r}' if len(v) <= 40 else f'text ({len(v.encode())} B)')
    return ' ++ '.join(shown) if shown else '(empty)'


def facts_block(name, calls):
    """the generated block of one kernel (or one family: every name)"""
    lines = [f'<!-- kernel-facts: {name} -->', '', '| fact | from the code |', '|---|---|']
    fam = calls[0].get('family_names')
    if fam:
        lines.append(f'| names | {", ".join(f"`{n}`" for n in fam)} |')
    seen = set()
    for c in calls:
        site = f'`src/{c["file"]}:{c["line"]}`' + (f' (`{c["decl"]}`)' if c['decl'] else '')
        key = (c['source_expr'], str(c['source']), c['header_expr'], str(c['header']) if not fam else '', tuple(c['inputs']), tuple(c['outputs']))
        if key in seen:
            lines.append(f'| made at | {site} (the same text) |')
            continue
        seen.add(key)
        lines.append(f'| made at | {site} |')
        lines.append(f'| inputs | {", ".join(f"`{x}`" for x in c["inputs"]) or "none"} |')
        lines.append(f'| outputs | {", ".join(f"`{x}`" for x in c["outputs"]) or "none"} |')
        src = c['source']
        lines.append(f'| source | `{c["source_expr"]}` = {fmt_pieces(src)}'
                     + ('' if isinstance(src, ct.EvalError) else f' ({len(str(src).encode())} B, `{sha(src)}`)') + ' |')
        hdr = c['header']
        if fam:
            lines.append(f'| header | `{c["header_expr"]}`: a variant\'s defines ++ the base header, one per name |')
            for (n, h) in c['family_headers']:
                lines.append(f'| header of `{n}` | {fmt_pieces(h)} ({len(str(h).encode())} B, `{sha(h)}`) |')
        else:
            lines.append(f'| header | `{c["header_expr"]}` = {fmt_pieces(hdr)}'
                         + ('' if isinstance(hdr, ct.EvalError) else f' ({len(str(hdr).encode())} B, `{sha(hdr)}`)') + ' |')
    lines += ['', f'<!-- /kernel-facts -->']
    return '\n'.join(lines)


def entries(src):
    """{heading name: (doc, [calls])}: one per static name, one per family (its calls carry family_names and
    family_headers)"""
    ks = kernels(src)
    out = {}
    fams = {}
    for name, calls in ks.items():
        fam = calls[0]['family']
        if fam:
            fams.setdefault(fam, []).append((name, calls))
        else:
            out[name] = (doc_of(calls[0]['file']), calls)
    for fam, members in fams.items():
        names = [n for (n, _) in members]
        heading = fam
        # one call site per family call: the members share it, each with its header
        sites = {}
        for (n, calls) in members:
            for c in calls:
                sites.setdefault((c['file'], c['line']), dict(c, family_names=names, family_headers=[]))['family_headers'].append((n, c['header']))
        out[heading] = (doc_of(members[0][1][0]['file']), list(sites.values()))
    return out, ks


ENTRY = re.compile(r'^### `([^`]+)`\s*$', re.M)
BLOCK = re.compile(r'<!-- kernel-facts: ([^ ]+) -->.*?<!-- /kernel-facts -->', re.S)


def parse_doc(text):
    """{name: (start, end)} of each entry: from its heading to the next heading of level <= 3 or the end"""
    heads = [(m.group(1), m.start()) for m in ENTRY.finditer(text)]
    bounds = [m.start() for m in re.finditer(r'^#{1,3} ', text, re.M)] + [len(text)]
    out = {}
    for (name, start) in heads:
        end = next(b for b in bounds if b > start)
        out[name] = (start, end)
    return out


def skeleton(name, calls):
    body = [f'### `{name}`', '', facts_block(name, calls), '']
    body += [f'- **{f}:** TODO' for f in PROSE_FIELDS]
    return '\n'.join(body) + '\n\n'


def write(src):
    ents, _ = entries(src)
    by_doc = {}
    for name, (doc, calls) in ents.items():
        by_doc.setdefault(doc, []).append((name, calls))
    for doc, items in sorted(by_doc.items()):
        path = os.path.join(REPO, doc)
        text = open(path, encoding='utf-8').read() if os.path.exists(path) else f'# Kernels\n\n'
        have = parse_doc(text)
        for (name, calls) in sorted(items):
            block = facts_block(name, calls)
            if name in have:
                a, b = have[name]
                seg = text[a:b]
                seg2 = BLOCK.sub(lambda m: block, seg, count=1) if BLOCK.search(seg) else seg.replace('\n', '\n\n' + block + '\n', 1)
                text = text[:a] + seg2 + text[b:]
                have = parse_doc(text)
            else:
                # into the section of the file that makes it (`## `src/<file>``), else at the end
                sec = re.search(r'^## `src/' + re.escape(calls[0]['file']) + r'`[^\n]*\n', text, re.M)
                if sec:
                    nxt = re.search(r'^#{1,2} ', text[sec.end():], re.M)
                    at = sec.end() + (nxt.start() if nxt else len(text) - sec.end())
                    text = text[:at].rstrip('\n') + '\n\n' + skeleton(name, calls) + text[at:]
                else:
                    text = text.rstrip('\n') + '\n\n' + skeleton(name, calls)
                have = parse_doc(text)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'w', encoding='utf-8').write(text.rstrip('\n') + '\n')
        print(f'{doc}: {len(items)} entries')


def check(src):
    ents, ks = entries(src)
    problems = []
    docs = sorted({doc for (doc, _) in ents.values()} | {
        os.path.relpath(os.path.join(r, f), REPO) for r, _, fs in os.walk(os.path.join(REPO, 'docs')) for f in fs if f == 'kernels.md'})
    found = {}
    for doc in docs:
        path = os.path.join(REPO, doc)
        if not os.path.exists(path):
            continue
        text = open(path, encoding='utf-8').read()
        for name, (a, b) in parse_doc(text).items():
            found[name] = (doc, text[a:b])
    for name, (doc, calls) in sorted(ents.items()):
        if name not in found:
            problems.append(f'{name}: no entry (expected in {doc})')
            continue
        fdoc, seg = found[name]
        if fdoc != doc:
            problems.append(f'{name}: documented in {fdoc}, made in {calls[0]["file"]} (its doc: {doc})')
        m = BLOCK.search(seg)
        if not m or m.group(0) != facts_block(name, calls):
            problems.append(f'{name}: the generated block is not the code\'s (run tools/kernel_docs.py --write)')
        for fld in PROSE_FIELDS:
            fm = re.search(r'^- \*\*' + re.escape(fld) + r':\*\*(.*)$', seg, re.M)
            if not fm or not fm.group(1).strip() or fm.group(1).strip().startswith('TODO'):
                problems.append(f'{name}: prose field "{fld}" missing or TODO')
    for name, (fdoc, _) in sorted(found.items()):
        if name not in ents:
            problems.append(f'{fdoc}: entry `{name}` names no kernel the code makes')
    lp, n_links = check_links()
    problems += lp
    n_names = len(ks)
    if problems:
        print(f'FAIL: {len(problems)} problem(s) over {n_names} kernel names ({len(ents)} entries)')
        for p in problems:
            print('  ' + p)
        return 1
    print(f'PASS: {n_names} kernel names, {len(ents)} entries, every one documented, every generated block the code\'s; '
          f'{n_links} links, every one resolves')
    return 0


# ---------------------------------------------------------------------------------------------------------------- links

def slug(title):
    """GitHub's anchor of a heading"""
    s = re.sub(r'`', '', title.strip().lower())
    s = re.sub(r'[^\w\- ]', '', s)
    return s.replace(' ', '-')


def anchors(text):
    out, fence = set(), False
    for line in text.splitlines():
        if line.startswith('```'):
            fence = not fence
        elif not fence:
            m = re.match(r'^#{1,6} (.*)$', line)
            if m:
                base = slug(m.group(1))
                n, a = 0, base
                while a in out:
                    n += 1
                    a = f'{base}-{n}'
                out.add(a)
    out |= set(re.findall(r'<a id="([^"]+)"></a>', text))
    return out


def doc_files():
    out = ['README.md', 'AGENTS.md']
    for root, _, fs in os.walk(os.path.join(REPO, 'docs')):
        out += [os.path.relpath(os.path.join(root, f), REPO) for f in fs if f.endswith('.md')]
    for root, _, fs in os.walk(os.path.join(REPO, 'src')):
        out += [os.path.relpath(os.path.join(root, f), REPO) for f in fs if f.endswith('.md')]
    return sorted(out)


def check_links():
    """every relative link and image of every doc: its file exists, and its anchor is a heading or `<a id>` there"""
    problems, n = [], 0
    cache = {}
    for doc in doc_files():
        text = open(os.path.join(REPO, doc), encoding='utf-8').read()
        body = re.sub(r'```.*?```', '', text, flags=re.S)
        for m in re.finditer(r'!?\[[^\]]*\]\(([^)\s]+)\)', body):
            url = m.group(1)
            if re.match(r'^[a-z]+:', url):
                continue
            n += 1
            path, _, anchor = url.partition('#')
            tgt = os.path.normpath(os.path.join(os.path.dirname(doc), path)) if path else doc
            full = os.path.join(REPO, tgt)
            if not os.path.exists(full):
                problems.append(f'{doc}: link {url}: no file {tgt}')
                continue
            if anchor and tgt.endswith('.md'):
                if tgt not in cache:
                    cache[tgt] = anchors(open(full, encoding='utf-8').read())
                if anchor not in cache[tgt]:
                    problems.append(f'{doc}: link {url}: no anchor #{anchor} in {tgt}')
    return problems, n


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    src = os.path.abspath(args[0]) if args else os.path.join(REPO, 'src')
    if '--write' in sys.argv:
        write(src)
    elif '--check' in sys.argv:
        sys.exit(check(src))
    else:
        ks = kernels(src)
        for name in sorted(ks):
            for c in ks[name]:
                print(f'{name:36s} src/{c["file"]}:{c["line"]:<5d} in {", ".join(c["inputs"])} | out {", ".join(c["outputs"])}')
                print(f'{"":36s}   source {c["source_expr"]} = {fmt_pieces(c["source"])}')
                print(f'{"":36s}   header {c["header_expr"]} = {fmt_pieces(c["header"])}')
        print(f'{len(ks)} kernel names, {sum(len(v) for v in ks.values())} calls')


if __name__ == '__main__':
    main()
