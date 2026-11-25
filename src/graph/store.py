"""抽取库的共用部分（各抽取版本共用、读成品的 load.py 也用，所以放在版本包外面）：库目录的布局、库的“全集”。

一个抽取版本一个库 output/graph/<版本>/（src.config.graph_dir）：
  概念DAG.html     查看页面（入口）
  概念统一.html    名称统一的核对页面（跑过 unify 才有）
  data/            成品：concepts / edges / reports.parquet（库里全部有有效存档的研报），graphs/（查看页面读的每篇数据），
                   以及名称统一的 concept_unify / unify_groups / unified_edges.parquet、unify.js
  cache/           大模型调用缓存（*_cache.jsonl）、各篇存档 reports/、出错记录 failed_reports.jsonl、进度日志 progress.log
"""
from pathlib import Path


class DagOut:
    """库目录的布局。"""

    def __init__(self, root):
        self.root = Path(root)
        self.viewer = self.root / '概念DAG.html'
        self.data = self.root / 'data'
        self.graphs = self.data / 'graphs'
        self.cache = self.root / 'cache'
        self.reports = self.cache / 'reports'
        self.failed = self.cache / 'failed_reports.jsonl'
        self.log = self.cache / 'progress.log'

    def make(self):
        for d in (self.data, self.cache):
            d.mkdir(parents=True, exist_ok=True)
        return self


def in_library(r):
    """库的全集：研报元数据表（status == 'ok' 的行）里能识别公司的个股研报——是个股公司研报，股票代码、公司名、券商都识别出来了。
    r 是元数据表的一行（dict）。每个版本的库只能装这些研报；具体抽哪些由样本清单或 --report-ids 决定。"""
    return bool(r['is_stock_report'] and r['stock_code'] and r['company'] and r['broker'])
