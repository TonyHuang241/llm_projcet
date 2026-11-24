"""唯一入口：在仓库根目录运行 python run.py <步骤> [参数]，python run.py <步骤> --help 看该步骤的选项。

步骤名只定义一次（下面的 STEPS）：命令名、src/ 下的目录名、output/ 下的目录名相同。每个步骤的模块提供 add_args(parser) 和
run(args)，默认运行参数写在该步骤自己的目录里；这里只做分发，选中子命令后才导入对应的模块（跑 text 不会导入 networkx、
大模型调用等用不到的东西）。以后加步骤只需在 STEPS 里加一行。
专题研究：python run.py study <名字> [参数] → src/studies/<名字>/。
"""
import argparse
import importlib
import sys

# 命令 -> (模块路径, 一句话说明)
STEPS = {
    'text': ('src.text', '研报纯文本 -> 正文段落表和研报元数据表（不调用大模型）'),
    'graph': ('src.graph', '正文段落 -> 每篇研报的概念和因果边（概念 DAG，调用大模型）'),
    'unify': ('src.graph.unify', '概念 DAG 抽完后，跨研报统一同一家公司同一个概念的不同叫法（调用大模型）'),
    'measure': ('src.measure', '概念 DAG -> 每篇研报的逻辑结构变量（源点数、路径数、平均路径长度，原始值，不调用大模型）'),
    'accuracy': ('src.accuracy', '分析师预测表 + 实际值 + 行情 -> 每篇研报和每个公司-券商对的预测准确度、同行时滞、控制变量（原始值）'),
}
STUDY_PKG = 'src.studies'


def usage():
    lines = ['用法：python run.py <步骤> [参数]    （python run.py <步骤> --help 看该步骤的选项）', '', '步骤：']
    lines += [f'  {name:<8}{desc}' for name, (_, desc) in STEPS.items()]
    lines += [f'  {"study":<8}专题研究：python run.py study <名字> [参数]，对应 {STUDY_PKG.replace(".", "/")}/<名字>/']
    return '\n'.join(lines)


def load_module(path, label):
    """导入步骤模块；模块本身不存在时给出清楚的报错（模块内部 import 出错照常抛出，便于定位）。"""
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError as e:
        if e.name and (path == e.name or path.startswith(e.name + '.')):
            raise SystemExit(f'没有 {label}：找不到模块 {path}（{path.replace(".", "/")}/）') from None
        raise


def main(argv):
    for stream in (sys.stdout, sys.stderr):  # 输出被重定向时 Windows 默认用本地编码，写中文会出错
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    if not argv or argv[0] in ('-h', '--help'):
        print(usage())
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == 'study':
        if not rest or rest[0].startswith('-'):
            raise SystemExit('用法：python run.py study <名字> [参数]')
        name, rest = rest[0], rest[1:]
        prog, module = f'run.py study {name}', load_module(f'{STUDY_PKG}.{name}', f'专题研究 {name}')
    elif cmd in STEPS:
        prog, module = f'run.py {cmd}', load_module(STEPS[cmd][0], f'步骤 {cmd}')
    else:
        raise SystemExit(f'未知的步骤：{cmd}\n\n{usage()}')
    ap = argparse.ArgumentParser(prog=prog)
    module.add_args(ap)
    module.run(ap.parse_args(rest))


if __name__ == '__main__':  # text 步骤用多进程，子进程会重新导入本文件，入口代码只能放在这里面
    main(sys.argv[1:])
