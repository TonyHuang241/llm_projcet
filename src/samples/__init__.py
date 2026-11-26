"""样本（sample）：分析用哪些研报。步骤入口 python run.py sample（列出样本、看各样本的抽取进度、按规则重建清单）。

样本不存逻辑结构，只是一张“研报 ID → 取自哪个抽取版本的库”的清单：本目录的 <样本名>.csv，两列 report_id、version，入 git。
逻辑结构存在各版本的库里（output/graph/<版本>/，见 src/graph/__init__.py 的 VERSIONS），同一版本下每篇只存一份，
所以几个样本可以共用同一个库里的研报，补抽一个新样本时已经抽过的研报不会重复抽、重复存。

下游怎么用：
  src/graph/load.py、src/measure/load.py 按样本读：清单决定每篇取自哪个版本；各版本按 SAMPLES 里登记的 versions 顺序上下拼接，
  版本内保持库里文件的行序（回归对求和顺序敏感，行序不能变）。accuracy 的范围 = 样本清单的研报；研究的范围再按 companies 限定。
  读成品时还可以另给一个版本（assignment 的 version）：给的就是样本的主版本（versions 的第一个）时照清单取；给的是别的版本时，
  样本的全部研报都从那个版本的库里取——开新版本、在同一批研报上重抽后，用它比较新旧版本（研究输出目录名是 <样本>_<版本>）。

登记的样本（SAMPLES）：
  hs300_deep   沪深300 深度研报，现有研究用的 1019 篇：975 篇取自 v1、44 篇取自 v0。冻结的清单，不能按规则重建。
               来历：重构前分析用的是 deep 运行（全部公司的深度报告：首页有深度、首次覆盖、公司专题等类型标注
               （src/text/meta.py 的 deep_line），且正文——清理后、去掉风险提示——超过 5000 字，选中 975 篇）和 pilot 运行
               （早期试跑的 7 家公司，65 篇）两次运行的合并，同一篇取 deep 的；只在 pilot 里的 44 篇取自 v0。
               2026-11-25 从当时的输出冻结成 hs300_deep.csv，行序 = 当时下游读到的行序（deep 的文件行序，再接 pilot 独有的）。
               研究范围另限：公司在 data/hs300_2024_dag_covered.csv（241 家）里，研究代码再要求图里至少有一条边。
  hs300_full   沪深300 成分股的全部个股研报，全部取自 v1（2026-11-25 已全部抽完）。按规则建清单（build_hs300_full）：
               成分股 = data/market_data/index_stock.parquet 里 index_code == '000300' 在 20231229、20240628、20241231 三期的并集；
               研报 = 研报元数据表里 is_stock_report 为真、status == 'ok'、stock_code、company、broker 都非空，且代码前 6 位在成分股里。
               研究范围不另限公司。

建一个新样本：在 SAMPLES 里登记一项（名称用英文；写清中文说明、来历、版本顺序、研究范围是否另限公司），给出 build 函数，
再运行 python run.py sample <名字> --build 生成 <名字>.csv（之后要冻结的，把 frozen 改成 True）。

在仓库根目录运行：
    python run.py sample                       列出全部样本
    python run.py sample hs300_full            篇数、各版本各多少篇、库里已有结果多少篇、还缺多少篇
    python run.py sample hs300_full --build    按规则重建清单（冻结的样本不能重建）
"""
import json
import time
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, DEFAULT_SAMPLE, INDEX_STOCK, REPORTS, graph_dir

SAMPLES_DIR = Path(__file__).resolve().parent
HS300_DATES = ('20231229', '20240628', '20241231')  # hs300_full 的成分股：这三期的并集


def build_hs300_full():
    """hs300_full 的清单：沪深300 成分股（HS300_DATES 三期并集）的全部能识别公司的个股研报，全部取自 v1，按 report_id 排序。"""
    from src.graph.store import in_library
    idx = pd.read_parquet(INDEX_STOCK, columns=['date', 'code', 'index_code'])
    cons = set(idx[(idx.index_code == '000300') & idx.date.isin(HS300_DATES)].code)
    meta = pd.read_parquet(REPORTS, columns=['report_id', 'is_stock_report', 'status', 'stock_code', 'company', 'broker'])
    meta = meta[meta.status == 'ok']
    keep = [in_library(r) and r['stock_code'][:6] in cons for r in meta.to_dict('records')]
    ids = sorted(meta[keep].report_id)
    print(f'成分股（{"、".join(HS300_DATES)} 三期并集）{len(cons)} 家；研报 {len(ids)} 篇、'
          f'{meta[keep].stock_code.str[:6].nunique()} 家公司')
    return pd.DataFrame({'report_id': ids, 'version': 'v1'})


# 样本登记表：versions 是读成品时各版本上下拼接的顺序，第一个是主版本（研究输出目录名用它）；
# companies 是研究范围另限的公司名单（csv，stock_code 列），None 表示不另限；frozen 的清单不能按规则重建；
# noun 是研究输出（正文数字的说明文字）里称呼样本研报的词
SAMPLES = {
    'hs300_deep': dict(
        title='沪深300 深度研报（现有研究的 1019 篇）', noun='深度研报',
        versions=('v1', 'v0'), frozen=True, build=None,
        companies=DATA_DIR / 'hs300_2024_dag_covered.csv',
        origin='重构前 deep 运行（全部公司的深度报告：首页有深度报告类标注、正文超过 5000 字，975 篇）与 '
               'pilot 运行（7 家试跑公司）的合并，同一篇取 deep 的；2026-11-25 从当时的输出冻结'),
    'hs300_full': dict(
        title='沪深300 成分股的全部个股研报', noun='样本内研报',
        versions=('v1',), frozen=False, build=build_hs300_full,
        companies=None,
        origin="index_stock.parquet 里 index_code == '000300' 在 20231229、20240628、20241231 三期并集的成分股；"
               '元数据表里能识别公司的个股研报（is_stock_report、status == ok、stock_code / company / broker 非空）'),
}


def spec(name):
    """样本的登记项；名字不对时报错并列出已有的样本。"""
    if name not in SAMPLES:
        raise SystemExit(f'没有样本 {name}；已有：{"、".join(SAMPLES)}（登记在 src/samples/__init__.py 的 SAMPLES）')
    return SAMPLES[name]


def path(name):
    return SAMPLES_DIR / f'{name}.csv'


def main_version(name):
    """样本的主版本（登记的 versions 的第一个）：研究输出目录名 <样本>_<版本> 默认用它。"""
    return spec(name)['versions'][0]


def read(name=DEFAULT_SAMPLE):
    """样本清单：DataFrame（report_id、version），行序同 csv。"""
    s, p = spec(name), path(name)
    if not p.exists():
        raise SystemExit(f'样本 {name} 还没有清单 {p}' + ('' if s['frozen'] else f'：先运行 python run.py sample {name} --build'))
    t = pd.read_csv(p, dtype=str, keep_default_na=False)
    if list(t.columns) != ['report_id', 'version'] or not t.report_id.is_unique:
        raise SystemExit(f'{p} 应只有 report_id、version 两列，且 report_id 不重复')
    bad = sorted(set(t.version) - set(s['versions']))
    if bad:
        raise SystemExit(f'{p} 里有没登记的版本 {bad}（样本 {name} 登记的版本：{s["versions"]}）')
    return t


def assignment(name=DEFAULT_SAMPLE, version=None):
    """(每篇取自哪个版本的 DataFrame（report_id、version，行序同清单）, 版本的拼接顺序)。
    version 为 None 或样本的主版本时照清单；是别的版本时，样本的全部研报都取自那个版本（比较新旧版本用）。"""
    t = read(name)
    if version is None or version == main_version(name):
        return t, tuple(spec(name)['versions'])
    return t.assign(version=version), (version,)


def library_ids(version):
    """版本的库里已有结果的研报（data/reports.parquet，即上次写输出时有有效存档的研报）；库还没有结果时为空集。"""
    p = graph_dir(version) / 'data' / 'reports.parquet'
    return set(pd.read_parquet(p, columns=['report_id']).report_id) if p.exists() else set()


def by_version(versions):
    """{版本: 篇数} 的说明文字。"""
    return '、'.join(f'{v} {n} 篇' for v, n in versions.items())


def write_manifest(out, **fields):
    """在输出目录写 manifest.json（这份输出用的样本、版本、篇数等），加上生成时间。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    obj = {**fields, 'generated': time.strftime('%Y-%m-%d %H:%M:%S')}
    (out / 'manifest.json').write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def status(name):
    """打印样本的篇数、各版本篇数、库里已有结果和还缺的篇数。"""
    s, t = spec(name), read(name)
    print(f'{name}：{s["title"]}')
    print(f'  来历：{s["origin"]}')
    print(f'  清单：{path(name).relative_to(SAMPLES_DIR.parent.parent).as_posix()}（{"冻结" if s["frozen"] else "按规则生成，可用 --build 重建"}）；'
          f'研究范围：' + (f'另限 {Path(s["companies"]).relative_to(DATA_DIR.parent).as_posix()} 里的公司' if s['companies'] else '不另限公司'))
    print(f'  研报 {len(t)} 篇、{t.version.nunique()} 个版本：{by_version(t.version.value_counts().to_dict())}'
          f'（读成品时的拼接顺序 {" → ".join(s["versions"])}）')
    total_have = 0
    for v in s['versions']:
        ids = set(t.report_id[t.version == v])
        if not ids:
            continue
        have = len(ids & library_ids(v))
        total_have += have
        print(f'  {v}：{len(ids)} 篇，库（output/graph/{v}/）里已有结果 {have} 篇，还缺 {len(ids) - have} 篇')
    print(f'  合计：已有 {total_have} 篇，还缺 {len(t) - total_have} 篇'
          + ('' if total_have == len(t) else f'（准确的待抽篇数和块数用 python run.py graph --sample {name} --dry-run 看）'))


def add_args(ap):
    ap.description = '样本：分析用哪些研报（研报 ID → 取自哪个抽取版本的库）。不给名字时列出全部样本'
    ap.add_argument('name', nargs='?', default=None, help='样本名；给了时打印它的篇数、各版本篇数、库里已有和还缺的篇数')
    ap.add_argument('--build', action='store_true', help='按登记的规则重建这个样本的清单（冻结的样本不能重建）')


def run(args):
    if args.name is None:
        if args.build:
            raise SystemExit('--build 要给样本名：python run.py sample <名字> --build')
        print(f'样本（登记在 src/samples/__init__.py 的 SAMPLES；默认 {DEFAULT_SAMPLE}）：')
        for name, s in SAMPLES.items():
            n = len(read(name)) if path(name).exists() else None
            print(f'  {name:<12}{s["title"]}；' + (f'{n} 篇' if n is not None else '还没有清单')
                  + f'；版本 {" → ".join(s["versions"])}；' + ('冻结' if s['frozen'] else '按规则生成')
                  + '；研究范围' + ('另限公司名单' if s['companies'] else '不另限公司'))
        return
    s = spec(args.name)
    if args.build:
        if s['frozen'] or s['build'] is None:
            raise SystemExit(f'样本 {args.name} 的清单是冻结的（{s["origin"]}），不能按规则重建；'
                             f'要改范围请另建一个样本（见 src/samples/__init__.py 开头）')
        t = s['build']()
        t.to_csv(path(args.name), index=False, lineterminator='\n')
        print(f'写到 {path(args.name)}')
    status(args.name)
