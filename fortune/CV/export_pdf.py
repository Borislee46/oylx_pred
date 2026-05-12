"""导出简历 HTML → PDF（Chrome headless，零依赖）

使用方法:
    python export_pdf.py                  # 导出 PDF
    python export_pdf.py --watch          # 监控 HTML/CSS 变化，自动导出
    python export_pdf.py --open           # 在浏览器中预览
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
HTML = HERE / "resume.html"
CSS = HERE / "resume.css"
OUT = HERE / "简历-李佳鹏-DS-2027.pdf"

CHROME_CANDIDATES = [
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
]


def find_chrome():
    for p in CHROME_CANDIDATES:
        if Path(p).exists():
            return p
    # fallback: try PATH
    for name in ("chrome", "msedge", "chromium"):
        try:
            subprocess.run([name, "--version"], capture_output=True)
            return name
        except FileNotFoundError:
            continue
    raise FileNotFoundError("未找到 Chrome/Edge，请安装后重试。")


def export():
    chrome = find_chrome()
    url = HTML.as_uri()
    subprocess.run([
        chrome,
        "--headless=new",
        "--disable-gpu",
        f"--print-to-pdf={OUT}",
        "--no-pdf-header-footer",
        url,
    ], check=True, timeout=30)
    size_kb = OUT.stat().st_size / 1024
    print(f"✓ 导出完成: {OUT.name} ({size_kb:.0f} KB)")


def open_preview():
    chrome = find_chrome()
    url = HTML.as_uri()
    subprocess.Popen([chrome, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✓ 浏览器已打开: {url}")


def watch():
    print("监控中... (Ctrl+C 停止)")
    print(f"  HTML: {HTML}")
    print(f"  CSS:  {CSS}")
    print(f"  →    {OUT}")
    last = {}
    while True:
        for f in (HTML, CSS):
            try:
                mtime = f.stat().st_mtime
            except FileNotFoundError:
                continue
            if f.name not in last or mtime != last[f.name]:
                last[f.name] = mtime
                print(f"[{time.strftime('%H:%M:%S')}] {f.name} 已变化，重新导出...")
                try:
                    export()
                except Exception as e:
                    print(f"✗ 导出失败: {e}")
        time.sleep(2)


if __name__ == "__main__":
    if "--watch" in sys.argv or "-w" in sys.argv:
        watch()
    elif "--open" in sys.argv or "-o" in sys.argv:
        open_preview()
    else:
        export()
