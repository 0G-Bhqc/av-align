"""生成 Colab 运行包用的 vendored L1 单文件（av_align_l1_vendored.py）。

机械合并 av_align 的四个 L1 模块（layout / interventions / build / fixtures）为单一
命名空间文件：去掉相对导入，其余逐字保留。生成后由 tests 同目录的核验脚本比对
vendored 与包版本的布局哈希一致性（生成时即验证，运行包内记录 SHA-256）。
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO.parent / "av_align-runpkg-tmp" / "av_align_l1_vendored.py"
MODULES = ["layout", "interventions", "build", "fixtures"]

HEADER = '''"""av_align L1 vendored 快照（Colab 运行包专用，非源码真值）。

由 D:\\Develop\\av_align\\scripts\\make_vendor.py 机械生成：
合并 av_align/{layout,interventions,build,fixtures}.py，去除相对导入，其余逐字保留。
每次仓库 L1 变更后必须重新生成并更新运行包内 SHA-256；上游真值是 git 仓库。
"""

'''

future_seen = False


def strip_module(src: str, name: str) -> str:
    global future_seen
    lines = src.splitlines()
    out: list[str] = []
    i = 0
    # 去掉模块 docstring（首个三引号块）与 from __future__ 重复
    while i < len(lines):
        line = lines[i]
        if line.startswith('"""'):
            j = i
            if line.count('"""') >= 2:
                i += 1
            else:
                j += 1
                while j < len(lines) and '"""' not in lines[j]:
                    j += 1
                i = j + 1
            continue
        if line.startswith("from __future__"):
            if not future_seen:
                out.append(line)
                future_seen = True
            i += 1
            continue
        if re.match(r"^from \.\w* import", line) or re.match(r"^from \. import", line):
            # 跳过整个括号导入块
            if "(" in line:
                while i < len(lines) and ")" not in lines[i]:
                    i += 1
            i += 1
            continue
        if line.startswith("import numpy as np") and any("numpy" in l for l in out):
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    parts = [HEADER]
    for name in MODULES:
        src = (REPO / "av_align" / f"{name}.py").read_text(encoding="utf-8")
        parts.append(f"# {'=' * 20} av_align/{name}.py {'=' * 20}\n\n")
        parts.append(strip_module(src, name))
        parts.append("\n\n")
    text = "".join(parts)
    # 清理连续空行（>2 合并为 2）
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    import hashlib

    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"vendored -> {OUT}")
    print(f"sha256   -> {sha}")
    # 写出 SHA 供运行包引用
    (OUT.parent / "av_align_l1_vendored.sha256").write_text(sha + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
