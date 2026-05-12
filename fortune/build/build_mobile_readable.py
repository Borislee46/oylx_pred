#!/usr/bin/env python3
"""
将 fortune/ 下所有命理分析 md 合并为一个手机可读文件，方便微信发送后在地铁上阅读。

输出:
  - 命理全集_手机阅读.txt   (纯文本，微信可直接预览)
  - 命理全集_手机阅读.html  (移动端HTML，浏览器打开，带目录导航)

用法:
  python build_mobile_readable.py
"""

import os
import re
from datetime import datetime

FORTUNE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.dirname(os.path.abspath(__file__))

# 阅读顺序（按这个顺序合并）
READING_ORDER = [
    "analyses/紫微斗数命盘分析_1993年4月6日.md",
    "analyses/紫微斗数命盘综合分析_完整版.md",
    "living/事实日志.md",
    "living/命理对照本.md",
    "living/人物知识图谱.md",
    "analyses/来广营风水×命盘交叉分析.md",
]

# 跳过 README（它是指南不是分析内容）


def strip_table_row_legacy(line: str) -> str:
    """Convert a single pipe-delimited table separator row like |---|---| into a plain divider."""
    if re.match(r"^\|[\s\-:|]+\|$", line):
        return ""
    return line


def table_to_text(lines):
    """Render a markdown table as readable indented text blocks."""
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)

    if len(rows) < 2:
        return "\n".join(["  ".join(r) for r in rows])

    # Drop separator row
    cleaned = []
    for r in rows:
        if all(re.fullmatch(r"[-: ]+", c) for c in r):
            continue
        cleaned.append(r)

    if not cleaned:
        return ""

    header = cleaned[0]
    body = cleaned[1:]

    out = []
    for row in body:
        for i, cell in enumerate(row):
            if i < len(header):
                out.append(f"  {header[i]}: {cell}")
        out.append("")
    return "\n".join(out)


def simplify_markdown(text: str) -> str:
    """简化 markdown 为纯文本，保留结构但去除复杂格式。"""
    lines = text.split("\n")
    result = []
    in_table = False
    table_buffer = []

    for line in lines:
        # 表格
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            if not in_table:
                in_table = True
                table_buffer = []
            table_buffer.append(line)
            continue
        else:
            if in_table:
                converted = table_to_text(table_buffer)
                if converted.strip():
                    result.append(converted)
                result.append("")
                in_table = False
                table_buffer = []

        # 标题
        if stripped.startswith("### "):
            result.append(f"\n  【{stripped[4:]}】")
        elif stripped.startswith("## "):
            result.append(f"\n▌ {stripped[3:]}")
        elif stripped.startswith("# "):
            result.append(f"\n{'='*36}")
            result.append(f"  {stripped[2:]}")
            result.append(f"{'='*36}")
        elif stripped.startswith("---"):
            result.append("  ─────────────────────────")
        elif stripped.startswith("```"):
            result.append("")
        elif stripped.startswith(">"):
            result.append(f"  {stripped.lstrip('> ')}")
        elif re.match(r"^[\-\*] ", stripped):
            result.append(f"  · {stripped.lstrip('-* ')}")
        elif re.match(r"^\d+\.", stripped):
            result.append(f"  {stripped}")
        else:
            result.append(line)

    if in_table and table_buffer:
        converted = table_to_text(table_buffer)
        if converted.strip():
            result.append(converted)

    return "\n".join(result)


def build_html(md_parts: list[tuple[str, str]], out_path: str):
    """构建移动端优化的 HTML。"""
    # 简单地把简化后的文本包装成 HTML
    html_parts = []
    for title, text in md_parts:
        html_parts.append(f'<section><h2>{title}</h2><pre>{text}</pre></section>')

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>命理全集 · 李佳鹏 · 癸酉 1993.04.06</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 15px; line-height: 1.75; color: #2c2c2c;
    background: #f8f5f0; padding: 12px;
  }}
  h1 {{
    text-align: center; font-size: 20px; padding: 20px 0 8px;
    color: #5e3c1c;
  }}
  .toc {{
    background: #fff; border-radius: 10px; padding: 16px; margin: 12px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
  }}
  .toc h3 {{ font-size: 14px; color: #8b6914; margin-bottom: 8px; }}
  .toc ol {{ padding-left: 20px; }}
  .toc li {{ margin: 4px 0; }}
  .toc a {{ color: #5e3c1c; text-decoration: none; }}
  section {{
    background: #fff; border-radius: 10px; padding: 16px; margin: 12px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
  }}
  section h2 {{
    font-size: 17px; color: #8b6914; border-bottom: 1px solid #e8dcc8;
    padding-bottom: 8px; margin-bottom: 12px;
  }}
  pre {{
    white-space: pre-wrap; word-wrap: break-word;
    font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 14px; line-height: 1.8; color: #3a3a3a;
  }}
  .footer {{
    text-align: center; font-size: 12px; color: #aaa; padding: 20px 0;
  }}
</style>
</head>
<body>
<h1>命理全集</h1>
<p style="text-align:center;color:#999;font-size:13px;">李佳鹏 · 癸酉年三月十六日子时 · 安星码 C5VUC</p>
<p style="text-align:center;color:#999;font-size:12px;">生成 {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<div class="toc">
<h3>目录</h3>
<ol>
{"".join(f'<li><a href="#s{i}">{title}</a></li>' for i, (title, _) in enumerate(md_parts))}
</ol>
</div>
{"".join(f'<section id="s{i}"><h2>{title}</h2><pre>{text}</pre></section>' for i, (title, text) in enumerate(md_parts))}
<div class="footer">命盘数据来源：文墨天机 · 安星码 C5VUC</div>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    print("=" * 50)
    print("  命理文档 → 手机可读格式")
    print("=" * 50)

    # 检查哪些文件存在
    parts = []
    found = 0
    missing = 0
    for fname in READING_ORDER:
        fpath = os.path.join(FORTUNE_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                raw = f.read()
            title = fname.replace(".md", "")
            simplified = simplify_markdown(raw)
            parts.append((title, simplified))
            found += 1
            print(f"  ✅ {fname} ({len(raw)} chars)")
        else:
            missing += 1
            print(f"  ❌ {fname} (not found)")

    if found == 0:
        print("\n没有找到任何文件，退出。")
        return

    print(f"\n共 {found} 篇，缺失 {missing} 篇\n")

    # 输出纯文本版
    txt_path = os.path.join(BUILD_DIR, "命理全集_手机阅读.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("命理全集\n")
        f.write("李佳鹏 · 癸酉年三月十六日子时 · 安星码 C5VUC\n")
        f.write(f"生成: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"共 {found} 篇\n")
        f.write("=" * 36 + "\n\n")

        for i, (title, text) in enumerate(parts, 1):
            f.write(f"\n{'█' * 36}\n")
            f.write(f"  {i}. {title}\n")
            f.write(f"{'█' * 36}\n\n")
            f.write(text)
            f.write("\n\n")

    print(f"  📄 纯文本: {txt_path}")

    # 输出 HTML 版
    html_path = os.path.join(BUILD_DIR, "命理全集_手机阅读.html")
    build_html(parts, html_path)
    print(f"  🌐 HTML:   {html_path}")

    print("\n发送到微信的方式:")
    print('  1. 纯文本: 微信 "文件传输助手" → 发送文件 → 选择 txt')
    print('  2. HTML:   微信 "文件传输助手" → 发送文件 → 选择 html')
    print('             → 手机端用浏览器打开阅读（效果更好）')
    print(f"\n完成！")


if __name__ == "__main__":
    main()
