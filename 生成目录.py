# -*- coding: utf-8 -*-
"""
生成目录.py
用途：一键将 MATLAB 项目的 ZIP 包镜像到当前项目下的 SSCTV_RPCA_py 目录，
并为每个 .m 生成对应的 .py 存根文件（便于你逐步把 .m 翻成 .py）。

直接运行此脚本即可。默认输入 ZIP 路径可在下方常量修改。
"""

from __future__ import annotations
import os
import re
import csv
import sys
import zipfile
import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# === 需要你确认的路径（Windows 可用原始中文路径）================================
# 例：r"C:\Users\24455\Desktop\基于光谱-空间关联全变分正则化的高光谱图像恢复模型研究\代码\SSCTV-RPCA.zip"
INPUT_ZIP = r"C:\Users\24455\Desktop\基于光谱-空间关联全变分正则化的高光谱图像恢复模型研究\代码\SSCTV-RPCA.zip"

# 输出根目录：当前脚本同级的 SSCTV_RPCA_py
OUTPUT_ROOT = Path(__file__).resolve().parent / "SSCTV_RPCA_py"

# 是否复制非 .m 文件（如数据、README、脚本等）
COPY_OTHER = True

# 项目名（写入 README）
PROJECT_NAME = "SSCTV-RPCA (Python Scaffold)"

# ============================================================================

FUNC_REGEXES = [
    re.compile(r'^\s*function\s*(?P<lhs>\[[^\]]*\]|[^\=\s]+)?\s*=\s*(?P<name>[a-zA-Z_]\w*)\s*\((?P<args>[^\)]*)\)',
               re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*function\s*(?P<name>[a-zA-Z_]\w*)\s*\((?P<args>[^\)]*)\)',
               re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*function\s*(?P<lhs>[^\=\s]+)?\s*=\s*(?P<name>[a-zA-Z_]\w*)\s*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*function\s*(?P<name>[a-zA-Z_]\w*)\s*$', re.IGNORECASE | re.MULTILINE),
]

@dataclass
class MatlabSignature:
    raw: str
    name: Optional[str]
    args: List[str]
    outputs: List[str]

def sanitize_identifier(s: str) -> str:
    s = s.strip()
    if not s:
        return "_"
    s = re.sub(r'[^0-9a-zA-Z_]', '_', s)
    if re.match(r'^\d', s):
        s = '_' + s
    if s == "~":
        return "_"
    return s

def dedup_names(names: List[str]) -> List[str]:
    seen = {}
    result = []
    for n in names:
        base = n
        if base in seen:
            seen[base] += 1
            n = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        result.append(n)
    return result

def parse_matlab_signature(text: str) -> MatlabSignature:
    for rx in FUNC_REGEXES:
        m = rx.search(text)
        if m:
            raw_line = m.group(0).strip()
            name = m.groupdict().get('name')
            lhs = m.groupdict().get('lhs') or ""
            args = m.groupdict().get('args') or ""

            args_list = []
            if args:
                args_list = [sanitize_identifier(x) for x in args.split(',') if x.strip()]
            args_list = dedup_names(args_list)

            outputs = []
            lhs = lhs.strip()
            if lhs:
                if lhs.startswith('[') and lhs.endswith(']'):
                    parts = [sanitize_identifier(x) for x in lhs[1:-1].split(',') if x.strip()]
                else:
                    parts = [sanitize_identifier(lhs)]
                outputs = dedup_names(parts)
            else:
                outputs = []

            return MatlabSignature(raw=raw_line, name=name, args=args_list, outputs=outputs)
    return MatlabSignature(raw="", name=None, args=[], outputs=[])

def py_stub_for_signature(sig: MatlabSignature, rel_path: str) -> str:
    header = [
        "# -*- coding: utf-8 -*-",
        '"""',
        f"Auto-generated stub converted from MATLAB file: {rel_path}",
        "This is a starting point for manual translation.",
        "",
        "Notes:",
        "- Review the signature and the return values; MATLAB semantics may differ.",
        "- Replace NotImplementedError with real implementation.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
    ]

    if sig.name:
        fn_name = sanitize_identifier(sig.name)
        args = ", ".join(sig.args)
        if not args:
            args = ""
        body = [
            f"def {fn_name}({args}):",
            f'    """',
            f"    Stub for MATLAB function `{sig.name}` extracted from: {rel_path}",
            f"    Original signature: {sig.raw}" if sig.raw else "    (Original signature not detected.)",
            f'    """',
            "    # TODO: translate the MATLAB implementation here",
            "    raise NotImplementedError('Function stub generated from MATLAB file')",
        ]
        return "\n".join(header + body) + "\n"
    else:
        body = [
            "def main():",
            f'    """',
            f"    Entry point stub for MATLAB script converted from: {rel_path}",
            f'    """',
            "    # TODO: translate the MATLAB script into Python here",
            "    raise NotImplementedError('Script stub generated from MATLAB file')",
            "",
            "if __name__ == '__main__':",
            "    main()",
        ]
        return "\n".join(header + body) + "\n"

def ensure_init_py(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)
    initf = dir_path / '__init__.py'
    if not initf.exists():
        initf.write_text('# Makes this directory a Python package\n', encoding='utf-8')

def write_project_scaffold(out_root: Path, project_name: Optional[str] = None):
    out_root.mkdir(parents=True, exist_ok=True)
    readme = out_root / 'README.md'
    if not readme.exists():
        readme.write_text(f"# {project_name or 'python-project'}\n\n"
                          "This project was scaffolded from a MATLAB repository.\n\n"
                          "Each `.m` file has a corresponding Python stub for manual translation.\n",
                          encoding='utf-8')
    gi = out_root / '.gitignore'
    if not gi.exists():
        gi.write_text("__pycache__/\n*.pyc\n*.pyo\n*.DS_Store\n.env\n.venv\nbuild/\ndist/\n",
                      encoding='utf-8')
    req = out_root / 'requirements.txt'
    if not req.exists():
        req.write_text("# Add libraries as you need them\nnumpy\nscipy\n", encoding='utf-8')

def common_prefix_dir(paths: List[str]) -> str:
    if not paths:
        return ""
    cp = posixpath.commonprefix(paths)
    if cp and not cp.endswith('/'):
        cp = cp.rsplit('/', 1)[0] + '/'
    return cp

def process_from_zip(zip_path: Path, out_root: Path, copy_other: bool, index_rows: List[List[str]]):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = [n for n in zf.namelist() if not n.endswith('/')]
        prefix = common_prefix_dir(names)

        for name in sorted(names):
            rel = name[len(prefix):] if name.startswith(prefix) else name
            if not rel:
                continue
            if rel.lower().endswith('.m'):
                with zf.open(name) as f:
                    raw = f.read()
                # 可能存在中文编码，尝试多种解码
                code = None
                for enc in ('utf-8', 'gb18030', 'gbk', 'latin1'):
                    try:
                        code = raw.decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
                if code is None:
                    code = raw.decode('utf-8', errors='ignore')

                sig = parse_matlab_signature(code)
                py_rel = rel[:-2] + '.py'
                py_dst = out_root / py_rel
                ensure_init_py(py_dst.parent)
                stub = py_stub_for_signature(sig, rel)
                py_dst.write_text(stub, encoding='utf-8')
                index_rows.append([rel, py_rel, 'function' if sig.name else 'script',
                                   sig.name or '', str(len(sig.outputs)), sig.raw])
            else:
                if copy_other:
                    out_path = out_root / rel
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as fsrc, open(out_path, 'wb') as fdst:
                        fdst.write(fsrc.read())
                ensure_init_py((out_root / rel).parent)

def main():
    zip_path = Path(INPUT_ZIP)
    if not zip_path.exists():
        print(f"[ERROR] 找不到 ZIP 文件：{zip_path}")
        print("请修改脚本顶部的 INPUT_ZIP 为你的实际路径（注意使用 r'...路径...' 原始字符串）。")
        sys.exit(2)

    out_root = OUTPUT_ROOT
    write_project_scaffold(out_root, PROJECT_NAME)

    index_rows: List[List[str]] = []
    process_from_zip(zip_path, out_root, COPY_OTHER, index_rows)

    idx_path = out_root / 'matlab_to_python_index.csv'
    with open(idx_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['matlab_path', 'python_path', 'kind', 'function_name', 'n_outputs', 'raw_signature'])
        w.writerows(index_rows)

    print(f"[OK] 目录及 .py 存根已生成：{out_root}")
    print(f"[OK] 映射索引已写入：{idx_path}")
    print("[TIP] 现在可以在 PyCharm 中逐个打开对应 .py 文件，把 .m 的实现迁移过去。")

if __name__ == '__main__':
    main()
