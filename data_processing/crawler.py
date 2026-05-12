"""
指南者(compassedu.hk)案例爬虫 — 纯 requests，断点续爬，安全限速。

用法:
  py crawler.py                           # 从已有数据末尾续爬，自动探测最新ID
  py crawler.py --start 45000 --end 46000 # 爬指定范围
  py crawler.py --start 50000             # 从50000爬到最新
  py crawler.py --retry not_found         # 重试skip list中指定类型的失败ID
  py crawler.py --probe                   # 只探测最新ID，不爬取

输出: cases_result.xlsx, skip_list.json, crawler.log
"""

import argparse
import json
import logging
import os
import random
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

# --- 默认配置 ---
CONFIG = {
    "OUTPUT_FILE": "cases_result.xlsx",
    "SKIP_FILE": "skip_list.json",
    "LOG_FILE": "crawler.log",
    "SAVE_INTERVAL": 20,
    "DELAY_MIN": 0.3,
    "DELAY_MAX": 0.8,
    "TIMEOUT": 20,
    "MAX_RETRIES": 3,
    "CONSECUTIVE_EMPTY_LIMIT": 50,
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# --- 日志 ---
logger = logging.getLogger("compass_crawler")


def setup_logger():
    if logger.hasHandlers():
        return
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(CONFIG["LOG_FILE"], encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)


def clean_str(text):
    if not isinstance(text, str):
        return text
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def load_existing(filepath):
    """加载已有数据，返回 (id_set, records_list)"""
    ids, data = set(), []
    if os.path.exists(filepath):
        try:
            df = pd.read_excel(filepath)
            if "URL" in df.columns:
                ids = set(df["URL"].str.extract(r"/(\d+)$")[0].dropna().astype(int))
            data = df.to_dict("records")
            logger.info(f"已加载 {len(ids)} 条已有数据: {filepath}")
        except Exception as e:
            logger.error(f"读取已有文件失败: {e}")
    return ids, data


def load_skip(filepath):
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_skip(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def add_skip(skip_list, case_id, reason, filepath):
    skip_list.setdefault(reason, []).append(case_id)
    save_skip(filepath, skip_list)


def save_data(records, filepath):
    if not records:
        return
    df = pd.DataFrame(records)
    wanted = [
        "案例标题",
        "发布时间",
        "学生姓名",
        "毕业学校",
        "本科专业",
        "录取学校",
        "录取专业",
        "基本背景",
        "主要经历",
        "Offer图片链接",
        "URL",
    ]
    cols = [c for c in wanted if c in df.columns]
    others = [c for c in df.columns if c not in cols]
    df = df[cols + others]
    df.to_excel(filepath, index=False, engine="openpyxl")
    logger.info(f"已保存 {len(records)} 条 → {filepath}")


def probe_max_id(session):
    """二分探测当前网站的近似最大案例ID"""
    lo, hi = 50000, 65000
    logger.info("正在探测最新案例ID...")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            resp = session.get(f"https://m.compassedu.hk/newst/{mid}", headers=HEADERS, timeout=10)
            if resp.status_code == 200 and len(resp.text) > 10000:
                soup = BeautifulSoup(resp.text, "html.parser")
                if soup.find("div", class_="module-head"):
                    lo = mid
                else:
                    hi = mid - 1
            else:
                hi = mid - 1
        except Exception:
            hi = mid - 1
        time.sleep(0.3)
    logger.info(f"当前最新案例ID约: {lo}")
    return lo


def scrape_page(case_id, session):
    """抓取单个案例页，返回 (status, data_dict)"""
    url = f"https://m.compassedu.hk/newst/{case_id}"

    for attempt in range(CONFIG["MAX_RETRIES"]):
        try:
            resp = session.get(url, headers=HEADERS, timeout=CONFIG["TIMEOUT"])
            break
        except requests.RequestException as e:
            if attempt < CONFIG["MAX_RETRIES"] - 1:
                wait = (attempt + 1) * 3
                logger.warning(f"ID {case_id} 请求失败(尝试{attempt + 1})，{wait}s后重试: {e}")
                time.sleep(wait)
            else:
                logger.error(f"ID {case_id} 请求全部失败: {e}")
                return "network_error", None

    if resp.status_code == 404:
        return "not_found", None

    if len(resp.text) < 10000:
        if "404" in resp.text[:500] or "not found" in resp.text[:500].lower():
            return "not_found", None
        return "empty", None

    soup = BeautifulSoup(resp.text, "html.parser")

    title_box = soup.find("div", class_="module-head")
    if not title_box:
        return "empty", None
    cname = title_box.find("div", class_="cname")
    if not cname:
        return "empty", None

    # 解析
    data = {"URL": url}
    data["案例标题"] = cname.get_text(strip=True)
    time_tag = title_box.find("div", class_="time")
    if time_tag:
        data["发布时间"] = time_tag.get_text(strip=True)

    for panel in soup.find_all("div", class_="module-panel"):
        head = panel.find("div", class_="head")
        if not head:
            continue
        title_div = head.find("div", class_="title")
        if not title_div:
            continue

        section = title_div.get_text(strip=True)

        if section == "录取详情":
            info_box = panel.find("div", class_="info-box")
            if info_box:
                for row in info_box.find_all("div", class_="row"):
                    name = row.find("div", class_="name")
                    detail = row.find("div", class_="detail") or row.find("a", class_="detail")
                    if name and detail:
                        data[name.get_text(strip=True)] = detail.get_text(strip=True)

        elif section == "主要经历":
            exp_div = panel.find("div", class_="experience")
            if exp_div:
                texts = []
                for item in exp_div.find_all("div", class_="experience_box"):
                    et = item.find("div", class_="experience_text")
                    if et:
                        texts.append(et.get_text(strip=True))
                if texts:
                    data["主要经历"] = "\n".join(texts)

        elif section == "Offer展示":
            offer_box = panel.find("div", class_="offer-box")
            if offer_box:
                img = offer_box.find("img")
                if img and img.get("src"):
                    data["Offer图片链接"] = img["src"]

    # 基本背景兜底
    if "基本背景" not in data:
        text = soup.get_text()
        m = re.search(r"基本背景[：:]\s*(.+?)(?:\n|$)", text)
        if m:
            data["基本背景"] = m.group(1).strip()

    for k, v in data.items():
        if isinstance(v, str):
            data[k] = clean_str(v)

    if len(data) > 2:
        return "success", data
    return "incomplete", data


def run_crawl(to_crawl, existing_ids, all_data, skip_list):
    """核心爬取循环"""
    session = requests.Session()
    consecutive_empty = 0
    successful_since_save = 0
    total_success = 0
    total_fail = 0

    for i, case_id in enumerate(to_crawl):
        if consecutive_empty >= CONFIG["CONSECUTIVE_EMPTY_LIMIT"]:
            logger.warning(f"连续 {consecutive_empty} 次空页，触发熔断！")
            break

        logger.info(f"[{i + 1}/{len(to_crawl)}] ID {case_id} ...")

        status, case_data = scrape_page(case_id, session)

        if status == "success":
            all_data.append(case_data)
            consecutive_empty = 0
            successful_since_save += 1
            total_success += 1
            logger.info(f"  OK  {case_data.get('案例标题', '?')[:70]}")

            if successful_since_save >= CONFIG["SAVE_INTERVAL"]:
                save_data(all_data, CONFIG["OUTPUT_FILE"])
                successful_since_save = 0
        else:
            consecutive_empty += 1
            total_fail += 1
            add_skip(skip_list, case_id, status, CONFIG["SKIP_FILE"])
            logger.info(f"  {status} (连续空页: {consecutive_empty})")

        delay = random.uniform(CONFIG["DELAY_MIN"], CONFIG["DELAY_MAX"])
        if consecutive_empty > 10:
            delay *= 2.5
        time.sleep(delay)

    save_data(all_data, CONFIG["OUTPUT_FILE"])
    return total_success, total_fail


def main():
    setup_logger()

    parser = argparse.ArgumentParser(description="指南者(compassedu.hk)案例爬虫")
    parser.add_argument("--start", type=int, help="起始ID")
    parser.add_argument("--end", type=int, help="结束ID（默认自动探测）")
    parser.add_argument(
        "--retry",
        type=str,
        choices=["not_found", "empty", "incomplete", "network_error", "blocked"],
        help="重试skip list中某类失败ID",
    )
    parser.add_argument("--probe", action="store_true", help="仅探测最新ID后退出")
    parser.add_argument("--output", type=str, default=CONFIG["OUTPUT_FILE"], help="输出文件名")
    args = parser.parse_args()

    CONFIG["OUTPUT_FILE"] = args.output
    session = requests.Session()

    # 仅探测模式
    if args.probe:
        max_id = probe_max_id(session)
        print(f"最新案例ID: {max_id}")
        return

    # 加载已有数据
    existing_ids, all_data = load_existing(CONFIG["OUTPUT_FILE"])
    skip_list = load_skip(CONFIG["SKIP_FILE"])
    all_skip_ids = {id for ids in skip_list.values() for id in ids}

    # 确定待爬列表
    if args.retry:
        if args.retry in skip_list and skip_list[args.retry]:
            to_crawl = sorted(set(skip_list[args.retry]))
            skip_list[args.retry] = []
            save_skip(CONFIG["SKIP_FILE"], skip_list)
            logger.info(f"重试模式: {len(to_crawl)} 个 {args.retry} 类型ID")
        else:
            logger.info(f"没有 {args.retry} 类型的失败ID需要重试")
            return
    else:
        start = args.start
        if start is None:
            if existing_ids:
                start = max(existing_ids) + 1
                logger.info(f"从已有数据末尾 {start} 开始续爬")
            else:
                start = 1

        end = args.end if args.end else probe_max_id(session)
        logger.info(f"爬取范围: {start} - {end}")

        to_crawl = sorted(set(range(start, end + 1)) - existing_ids - all_skip_ids)

    logger.info(f"待爬: {len(to_crawl)} | 已有: {len(existing_ids)} | 已跳过: {len(all_skip_ids)}")
    logger.info(
        f"延迟: {CONFIG['DELAY_MIN']}-{CONFIG['DELAY_MAX']}s | "
        f"熔断: {CONFIG['CONSECUTIVE_EMPTY_LIMIT']}次 | "
        f"每{CONFIG['SAVE_INTERVAL']}条保存"
    )

    if not to_crawl:
        logger.info("没有需要爬取的ID，数据已是最新。")
        return

    succ, fail = run_crawl(to_crawl, existing_ids, all_data, skip_list)

    logger.info(f"完成！成功: {succ} | 失败: {fail} | 总计: {len(all_data)}")


if __name__ == "__main__":
    main()
