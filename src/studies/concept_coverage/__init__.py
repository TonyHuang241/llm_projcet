"""专题研究 concept_coverage：一个样本里有多少篇研报抽到了某个指定概念，以及它在图里的位置（独立的检查工具）。

只读 graph 步骤的成品（src/graph/load.py 的 read_table），不调用大模型，不被其他步骤调用，也不改其他步骤的输出；
可以反复运行，每次覆盖自己的输出目录。默认概念是“投资评级”（src/measure/variables.py 的 RATING_NODE）。

口径（每篇研报一行，图只由进 DAG 的边构成，同 measure 步骤）
  extracted      库里有这篇的抽取结果；没有的只计数，不进后面各比例的分母
  in_table       概念表里有名字完全等于该概念的概念
  in_graph       该概念是图里的节点（至少有一条 in_dag 的边连着它）
  n_in、n_out    它在图里的入度、出度：按边表里 in_dag 为真的边重算，同一对（原因, 结果）只算一次、自环不算
                 （与 build_graph 建出的图相同）；每次运行都和概念表的 n_in、n_out、in_dag 逐个概念比对，结果打印并写进
                 summary.md 和 manifest.json，以边表重算的为准
  strict_sink    入度 > 0、出度 = 0（严格汇点，measure 的 RATING_STRICT_SINK 同一口径）
  has_out        出度 > 0（它还指向别的概念）；source_only：入度 = 0、出度 > 0
  alias_only     没有完全同名的概念，但某个概念的 aliases（按“、”拆开）里有一项完全等于该概念名
  variant_names  没有完全同名的概念时，名字里含 --contains 子串的概念名（用“、”连接）；has_variant 为是否非空
  --contains 默认“评级”；--concept 不是默认值而又没给 --contains 时不做近似名称统计。

在仓库根目录运行：
    python run.py study concept_coverage                                   默认样本 hs300_deep、概念“投资评级”
    python run.py study concept_coverage --sample hs300_full
    python run.py study concept_coverage --concept 营业收入 --tag revenue --sample hs300_full
--concept 不是默认值时必须给 --tag（输出子目录的英文名，默认 rating）。

输出在 output/studies/concept_coverage/<样本>_<版本>/<tag>/：
    reports.parquet      每篇研报一行的标记表（report_id、version、company、stock_code、date、broker、n_chars、n_chunks 和上面各标记）
    summary.csv          汇总表：样本清单 → 已抽取 → 概念表里有 → 在图里 → 是严格汇点，及孤立、有出边、近似名称、别名等几行；
                         篇数、占已抽取的比例、占上一层（parent 列）的比例
    by_length.csv        按正文字数 n_chars 分组（<1000、1000–2000、2000–5000、5000–10000、≥10000）的篇数和三个比例
    by_chunks.csv        按块数 n_chunks（1、2、3+）分组，同上
    by_broker.csv        按券商分组，同上，按篇数降序
    variant_names.csv    没有同名概念的研报里各近似名称出现的研报数（降序）
    summary.md           一页中文小结（汇总表、度数核对和各分组表，只有数字和口径）
    manifest.json        样本、版本、概念、篇数、度数核对、生成时间（src/samples 的 write_manifest）
"""
from src.config import DEFAULT_SAMPLE, study_dir
from src.measure.variables import RATING_NODE

STUDY = 'concept_coverage'
DEFAULT_CONTAINS, DEFAULT_TAG = '评级', 'rating'  # 默认概念（RATING_NODE）对应的近似名称子串和输出子目录


def add_args(ap):
    ap.description = '统计一个样本里有多少篇研报抽到了某个概念、它在图里的位置（独立的检查工具，只读库的成品）'
    ap.add_argument('--sample', default=DEFAULT_SAMPLE, help=f'样本，默认 {DEFAULT_SAMPLE}；样本见 python run.py sample')
    ap.add_argument('--version', default=None, help='抽取版本，默认样本的主版本（照清单）；给别的版本时样本的全部研报都取自它')
    ap.add_argument('--concept', default=RATING_NODE, help=f'概念名（完全匹配），默认 {RATING_NODE}')
    ap.add_argument('--contains', default=None,
                    help=f'近似名称用的子串；默认概念时默认“{DEFAULT_CONTAINS}”，其他概念不给时不做近似名称统计')
    ap.add_argument('--tag', default=None, help=f'输出子目录的英文名；默认概念时默认 {DEFAULT_TAG}，其他概念必须给')


def run(args):
    import time

    from src import samples
    from src.graph import version_spec
    from src.studies.concept_coverage import coverage
    default = args.concept == RATING_NODE
    if not default and not args.tag:
        raise SystemExit(f'--concept 不是默认的“{RATING_NODE}”时必须用 --tag 给输出子目录的英文名，'
                         f'如 --concept {args.concept} --tag <英文名>')
    tag = args.tag or DEFAULT_TAG
    contains = args.contains if args.contains is not None else (DEFAULT_CONTAINS if default else None)
    samples.spec(args.sample)
    if args.version:
        version_spec(args.version)
    version = args.version or samples.main_version(args.sample)
    out = study_dir(STUDY, args.sample, version) / tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f'样本 {args.sample}，抽取版本 {version}，概念“{args.concept}”，近似名称子串：{contains or "（不统计）"}')

    t, check = coverage.report_flags(args.sample, args.version, args.concept, contains)
    tabs = coverage.tables(t, contains)
    t.to_parquet(out / 'reports.parquet', index=False)
    for name, df in tabs.items():
        df.to_csv(out / name, index=False, encoding='utf-8-sig', float_format='%.6f', lineterminator='\n')

    ok = check['n_mismatch_concepts'] == 0 and check['n_edge_nodes_not_in_table'] == 0
    check_text = (f'概念表的 n_in、n_out、in_dag 与按边表（in_dag 的边，同一对只算一次、去自环）重算的逐个比对：'
                  f'共 {check["n_concepts"]} 个概念，{"全部一致" if ok else "不一致"}；不一致 {check["n_mismatch_concepts"]} 个概念、'
                  f'涉及 {check["n_mismatch_reports"]} 篇研报；边表里有、概念表里没有的节点 {check["n_edge_nodes_not_in_table"]} 个。'
                  f'下面的度数一律按边表重算。')
    pd_opts = dict(index=False, float_format=lambda v: f'{v:.4f}')
    print('\n[汇总]\n' + tabs['summary.csv'].to_string(**pd_opts))
    print('\n[度数核对] ' + check_text)
    for name in ('by_length.csv', 'by_chunks.csv'):
        print(f'\n[{name}]\n' + tabs[name].to_string(**pd_opts))
    big = tabs['by_broker.csv'][lambda d: d.n_reports >= 100].sort_values(['share_strict_sink', 'group'])
    print(f'\n[券商：篇数 ≥ 100 的 {len(big)} 家里“是严格汇点”比例最低的 10 家、最高的 5 家]\n'
          + (big.head(10).to_string(**pd_opts) + '\n' + big.iloc[::-1].head(5).to_string(**pd_opts) if len(big) else '（没有）'))
    if contains:
        print('\n[近似名称前 15 个]\n' + tabs['variant_names.csv'].head(15).to_string(index=False))

    write_md(out / 'summary.md', args.sample, version, args.concept, contains, tabs, check_text)
    x = t[t.extracted]
    vs = x.version.value_counts().to_dict()
    samples.write_manifest(
        out, study=STUDY, sample=args.sample, version=version,
        version_rule='照样本清单' if args.version in (None, samples.main_version(args.sample)) else '样本的全部研报都取自该版本',
        concept=args.concept, contains=contains, tag=tag, n_sample=len(t), n_extracted=len(x),
        extracted_by_version={v: int(n) for v, n in vs.items()}, n_in_table=int(x.in_table.sum()),
        n_in_graph=int(x.in_graph.sum()), n_strict_sink=int(x.strict_sink.sum()), degree_check=check)
    print(f'\n输出在 {out}（{time.perf_counter() - t0:.1f} 秒）')


def write_md(path, sample, version, concept, contains, tabs, check_text):
    """一页中文小结：只有数字和口径。"""
    import time

    from src.studies.concept_coverage.coverage import md_table
    vn = tabs['variant_names.csv']
    parts = [f'# 概念覆盖统计：{concept}', '',
             f'样本 {sample}；抽取版本 {version}；概念“{concept}”；生成时间 {time.strftime("%Y-%m-%d %H:%M:%S")}', '',
             '## 汇总', '',
             '“已抽取”是库里有结果的研报，以下比例的分母都是它（share_of_extracted），share_of_parent 是占 parent 一层的比例。'
             '“在图里”指至少有一条进 DAG 的边连着该概念；严格汇点是入度 > 0、出度 = 0；别名按“、”拆开后完全匹配。', '',
             md_table(tabs['summary.csv']), '', '## 度数核对', '', check_text, '']
    for title, name, note in [
            ('按正文字数', 'by_length.csv', '已抽取的研报按研报表的 n_chars 分组（左闭右开）；三个比例的分母是该组篇数。'),
            ('按块数', 'by_chunks.csv', '已抽取的研报按研报表的 n_chunks 分组；三个比例的分母是该组篇数。'),
            ('按券商', 'by_broker.csv', '已抽取的研报按券商分组，按篇数降序；三个比例的分母是该券商的篇数。')]:
        parts += [f'## {title}', '', note, '', md_table(tabs[name]), '']
    if contains:
        parts += ['## 近似名称', '', f'没有同名概念的研报里，名字含“{contains}”的各概念名出现的研报数（一篇里出现算一次），降序。', '',
                  md_table(vn) if len(vn) else '（没有）', '']
    path.write_text('\n'.join(parts), encoding='utf-8')
