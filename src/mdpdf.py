"""Markdown 文档 -> PDF（各项研究共用）：md -> 独立 HTML -> 本机 Chrome / Edge 无头打印成 PDF。不需要联网。

做法
  1. 行间公式 $$…$$ 先换成占位符，免得 markdown 把下标、星号当成强调；markdown_it（commonmark + 表格）渲染正文，md 里内嵌的
     HTML（表格、<sub>、<i> 等）原样保留；再把占位符换成 MathML。
  2. 公式由下面的 latex_to_mathml 转成 MathML，浏览器自己排版（Chrome / Edge 109 起支持 MathML Core），字体用 Windows 自带的
     Cambria Math，所以不联网、不依赖 MathJax / KaTeX。只支持文档里用到的 LaTeX 子集：字母、数字、运算符、上下标、\\frac、
     \\sum、\\left…\\right、\\boldsymbol、\\mathbf、\\mathrm、\\text、希腊字母、\\in、\\neq、\\times 等和 \\, \\; \\quad \\qquad
     等空白；遇到不认识的命令直接报错（指出是哪条命令），不会悄悄输出错的公式。行内的 $…$ 不处理。
  3. HTML 带 <base href="md 所在目录">，图片等相对路径照 md 的位置解析，所以中间 HTML 可以写在任何地方（不在文档所在的目录里留文件）。
  4. 无头浏览器 --print-to-pdf，用临时的用户目录（不碰本机浏览器的配置），不打印页眉页脚；先写到临时目录，成功后再移到目标位置。

版式：A4，页边距 20 mm；正文 Times New Roman + 宋体，标题微软雅黑；三线表；图片不超出版心；标题、表题、图和表尽量不跨页断开。
"""
import re
import shutil
import subprocess
import tempfile
from html import escape
from pathlib import Path

BROWSERS = [Path(r'C:\Program Files\Google\Chrome\Application\chrome.exe'),
            Path(r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'),
            Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'),
            Path(r'C:\Program Files\Microsoft\Edge\Application\msedge.exe')]

CSS = """
@page { size: A4; margin: 20mm 20mm 20mm 20mm; }
html { font-size: 10.5pt; }
body { margin: 0; background: #fff; color: #111; line-height: 1.75;
       font-family: 'Times New Roman', 'SimSun', '宋体', serif; }
h1, h2, h3, strong, th { font-family: 'Times New Roman', 'Microsoft YaHei', '微软雅黑', sans-serif; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 1.2em; }
h2 { font-size: 13.5pt; margin: 1.5em 0 .6em; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 1.2em 0 .5em; break-after: avoid; }
p, li { text-align: justify; }
p { margin: .45em 0; }
ol, ul { padding-left: 1.6em; }
li { margin: .3em 0; }
img { display: block; max-width: 100%; height: auto; margin: .6em auto .2em; break-inside: avoid; }
p:has(> img) { break-inside: avoid; break-after: avoid; margin: 0; }
p.caption { text-align: center; margin-top: .9em; }
p.caption:has(+ table) { break-after: avoid; }
p:has(> img) + p.caption, table + p { break-before: avoid; }
table { border-collapse: collapse; margin: .4em auto .5em; font-size: 9pt; line-height: 1.45; break-inside: avoid;
        border-top: 1.4px solid #000; border-bottom: 1.4px solid #000; }
thead th, table tr:first-child th { border-bottom: .7px solid #000; }
th { font-weight: normal; padding: 4px 7px; vertical-align: bottom; text-align: center; }
th, td[align="right"], td[align="center"] { white-space: nowrap; }
td { padding: 2.5px 7px; vertical-align: top; }
math { font-family: 'Cambria Math', 'STIX Two Math', math; }
math[display="block"] { display: block math; margin: .8em 0; font-size: 1.12em; }
"""

# ---------------------------------------------------------------- LaTeX 子集 -> MathML
GREEK = dict(zip(('alpha beta gamma delta epsilon varepsilon zeta eta theta iota kappa lambda mu nu xi pi rho sigma tau phi '
                   'varphi chi psi omega Gamma Delta Theta Lambda Pi Sigma Phi Omega').split(), 'αβγδϵεζηθικλμνξπρστϕφχψωΓΔΘΛΠΣΦΩ'))
OPS = {'in': '∈', 'notin': '∉', 'neq': '≠', 'ne': '≠', 'times': '×', 'cdot': '⋅', 'le': '≤', 'leq': '≤', 'ge': '≥',
       'geq': '≥', 'pm': '±', 'approx': '≈', 'to': '→', 'ldots': '…', 'cdots': '⋯', 'prime': '′'}
BIG = {'sum': '∑', 'prod': '∏'}
SPACE = {',': .1667, ':': .2222, '>': .2222, ';': .2778, ' ': .25, 'quad': 1, 'qquad': 2, '!': -.1667}
DELIM = {'(': '(', ')': ')', '[': '[', ']': ']', '|': '|', '.': '', '\\{': '{', '\\}': '}', '\\|': '‖'}


def styled(ch, bold, italic):
    """拉丁、希腊字母 -> Unicode 数学粗体（italic=True 为粗斜体）。"""
    if 'A' <= ch <= 'Z' or 'a' <= ch <= 'z':
        base = (0x1D468 if italic else 0x1D400) + (ord(ch) - 65 if ch <= 'Z' else 26 + ord(ch) - 97)
        return chr(base)
    if 'α' <= ch <= 'ω':
        return chr((0x1D736 if italic else 0x1D6C2) + ord(ch) - ord('α'))
    if 'Α' <= ch <= 'Ω':
        return chr((0x1D71C if italic else 0x1D6A8) + ord(ch) - ord('Α'))
    return ch


class TexParser:
    TOKEN = re.compile(r'\\[A-Za-z]+|\\.|[A-Za-z]|\d+(?:\.\d+)?|\S')

    def __init__(self, tex):
        self.toks = [t for t in self.TOKEN.findall(tex)]
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self):
        t = self.peek()
        if t is None:
            raise ValueError('公式意外结束')
        self.i += 1
        return t

    def arg(self):
        """一个参数：{…} 或单个记号。"""
        if self.peek() == '{':
            self.take()
            nodes = self.seq('}')
            self.take()
            return mrow(nodes)
        return mrow([self.atom()])

    def raw_group(self):
        """{…} 里的原始记号（\\text、\\boldsymbol 等用）。"""
        if self.take() != '{':
            raise ValueError('这里需要 {')
        depth, out = 1, []
        while True:
            t = self.take()
            depth += (t == '{') - (t == '}')
            if depth == 0:
                return out
            out.append(t)

    def seq(self, stop=None):
        nodes = []
        while self.peek() is not None and self.peek() != stop:
            t = self.peek()
            if t in ('_', '^'):
                if not nodes:
                    raise ValueError(f'{t} 前面没有底数')
                nodes[-1] = self.script(nodes[-1])
            elif t == "'":
                self.take()
                if not nodes:
                    raise ValueError("' 前面没有底数")
                nodes[-1] = f'<msup>{nodes[-1]}<mo>′</mo></msup>'
            else:
                nodes.append(self.atom())
        return nodes

    def script(self, base):
        sub = sup = None
        while self.peek() in ('_', '^'):
            if self.take() == '_':
                sub = self.arg()
            else:
                sup = self.arg()
        big = base.startswith('<mo largeop')
        if sub and sup:
            return f'<{"munderover" if big else "msubsup"}>{base}{sub}{sup}</{"munderover" if big else "msubsup"}>'
        if sub:
            return f'<{"munder" if big else "msub"}>{base}{sub}</{"munder" if big else "msub"}>'
        return f'<{"mover" if big else "msup"}>{base}{sup}</{"mover" if big else "msup"}>'

    def atom(self):
        t = self.take()
        if t == '{':
            nodes = self.seq('}')
            self.take()
            return mrow(nodes)
        if re.fullmatch(r'[A-Za-z]', t):
            return f'<mi>{t}</mi>'
        if re.fullmatch(r'\d+(?:\.\d+)?', t):
            return f'<mn>{t}</mn>'
        if not t.startswith('\\'):
            return f'<mo>{escape({"-": "−", "*": "∗"}.get(t, t))}</mo>'
        name = t[1:]
        if name in SPACE:
            return f'<mspace width="{SPACE[name]}em"/>'
        if name in GREEK:
            return f'<mi>{GREEK[name]}</mi>'
        if name in OPS:
            return f'<mo>{OPS[name]}</mo>'
        if name in BIG:
            return f'<mo largeop="true" movablelimits="false">{BIG[name]}</mo>'
        if name == 'frac':
            return f'<mfrac>{self.arg()}{self.arg()}</mfrac>'
        if name == 'sqrt':
            return f'<msqrt>{self.arg()}</msqrt>'
        if name == 'left':
            open_ = self.delim()
            nodes = self.seq(r'\right')
            if self.take() != r'\right':
                raise ValueError(r'\left 没有对应的 \right')
            close = self.delim()
            fence = lambda d: f'<mo stretchy="true" fence="true">{escape(d)}</mo>' if d else ''  # noqa: E731
            return mrow([fence(open_)] + nodes + [fence(close)])
        if name in ('boldsymbol', 'mathbf', 'bm'):
            chars = [GREEK.get(x[1:], x) if x.startswith('\\') else x for x in self.raw_group()]
            return mrow([f'<mi>{styled(ch, True, name != "mathbf")}</mi>' for ch in chars])
        if name in ('mathrm', 'text', 'textrm', 'operatorname'):
            s = ''.join(x if not x.startswith('\\') else x[1:] for x in self.raw_group())
            return f'<mtext>{escape(s)}</mtext>' if name.startswith('text') else f'<mi mathvariant="normal">{escape(s)}</mi>'
        if name in ('{', '}', '|', '%', '&', '#', '_'):
            return f'<mo>{escape(name)}</mo>'
        raise ValueError(f'不支持的 LaTeX 命令 {t}（src/mdpdf.py 的 latex_to_mathml 只支持文档里用到的子集）')

    def delim(self):
        t = self.take()
        if t not in DELIM:
            raise ValueError(rf'\left / \right 后面的定界符 {t} 不支持')
        return DELIM[t]


def mrow(nodes):
    return nodes[0] if len(nodes) == 1 else f'<mrow>{"".join(nodes)}</mrow>'


def latex_to_mathml(tex, display=True):
    p = TexParser(tex)
    nodes = p.seq()
    if p.peek() is not None:
        raise ValueError(f'公式里有多余的 {p.peek()}：{tex.strip()}')
    attr = ' display="block"' if display else ''
    return f'<math xmlns="http://www.w3.org/1998/Math/MathML"{attr}><mrow>{"".join(nodes)}</mrow></math>'


# ---------------------------------------------------------------- md -> HTML -> PDF
def md_to_html(md_path):
    """md 文件 -> 独立的 HTML 文本（图片等相对路径照 md 所在目录解析）。"""
    from markdown_it import MarkdownIt
    md_path = Path(md_path).resolve()
    text = md_path.read_text(encoding='utf-8')
    maths = []

    def hold(m):
        maths.append(m.group(1))
        return f'\n\nMATHBLOCK{len(maths) - 1}END\n\n'
    text = re.sub(r'\$\$(.+?)\$\$', hold, text, flags=re.S)
    body = MarkdownIt('commonmark', {'html': True}).enable('table').render(text)
    # 整段只有一个加粗的“表 N …”“图 N …”是表题、图题：居中，表题不与下面的表分页
    body = re.sub(r'<p>(<strong>[表图] ?\d+[ 　].*?</strong>)</p>', r'<p class="caption">\1</p>', body)
    for i, tex in enumerate(maths):
        try:
            mml = latex_to_mathml(tex)
        except ValueError as e:
            raise SystemExit(f'{md_path.name} 第 {i + 1} 个行间公式转换失败：{e}') from None
        body, n = re.subn(rf'<p>MATHBLOCK{i}END</p>', lambda m: mml, body)
        assert n == 1, f'第 {i + 1} 个公式的占位符没有找到'
    m = re.search(r'^#\s+(.+)$', text, flags=re.M)
    title = escape(m.group(1).strip() if m else md_path.stem)
    return (f'<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n<title>{title}</title>\n'
            f'<base href="{md_path.parent.as_uri()}/">\n<style>{CSS}</style>\n</head>\n<body>\n{body}</body>\n</html>\n')


def find_browser():
    for p in BROWSERS:
        if p.exists():
            return p
    raise SystemExit('找不到 Chrome 或 Edge，无法打印 PDF；找过：\n' + '\n'.join(f'  {p}' for p in BROWSERS))


def headless(html_path, *flags, timeout=180):
    """无头浏览器打开 html_path，附加 flags（如 --print-to-pdf=…、--screenshot=…）；用临时用户目录。"""
    with tempfile.TemporaryDirectory() as prof:
        cmd = [str(find_browser()), '--headless', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
               '--disable-extensions', f'--user-data-dir={prof}', *flags, Path(html_path).resolve().as_uri()]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
        if r.returncode != 0:
            raise SystemExit(f'浏览器退出码 {r.returncode}：{" ".join(cmd)}\n{r.stderr[-2000:]}')


def html_to_pdf(html_path, pdf_path):
    pdf_path = Path(pdf_path)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pdf = Path(tmp) / 'out.pdf'
        headless(html_path, '--no-pdf-header-footer', f'--print-to-pdf={tmp_pdf}')
        if not tmp_pdf.exists() or tmp_pdf.stat().st_size == 0:
            raise SystemExit(f'浏览器没有写出 PDF（{html_path}）')
        shutil.move(str(tmp_pdf), str(pdf_path))
    return pdf_path


def convert(md_path, html_path, pdf_path=None):
    """md -> html_path（中间文件）-> PDF（默认在 md 旁边、同名 .pdf）。返回 PDF 路径。"""
    md_path = Path(md_path).resolve()
    if not md_path.exists():
        raise SystemExit(f'找不到文档 {md_path}')
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(md_to_html(md_path), encoding='utf-8')
    return html_to_pdf(html_path, pdf_path or md_path.with_suffix('.pdf'))
