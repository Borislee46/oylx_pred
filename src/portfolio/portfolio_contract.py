from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.utils.contract_config import get_tier_config

_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge_base" / "contracts_master.csv"

_SUPPORTED_REGIONS = frozenset({"中国香港", "中国澳门", "新加坡", "马来西亚"})

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioContract:
    tier_id: str
    label: str
    regions: tuple[str, ...]
    max_schools: int
    max_majors: int
    price_cny: int
    is_joint: bool
    min_schools: int = 1
    reject_deduction: int = 0
    visa_denial: int = 0
    add_school_cost: int = 0
    program: str = ""

    @property
    def refund_ratio(self) -> float:
        return self.reject_deduction / self.price_cny if self.price_cny else 0.0

    @classmethod
    def from_tier(cls, tier_id: str) -> PortfolioContract:
        csv_contract = load_contracts().get(tier_id)
        if csv_contract is not None:
            return csv_contract
        t = get_tier_config(tier_id)
        regions = tuple(t.get("regions") or ())
        return cls(
            tier_id=tier_id,
            label=str(t.get("label") or tier_id),
            regions=regions,
            max_schools=int(t.get("max_schools") or 6),
            max_majors=int(t.get("max_majors") or 2),
            price_cny=int(t.get("price_cny") or 0),
            is_joint=len(regions) > 1,
        )


_SOURCE_REGIONS: dict[str, tuple[str, ...]] = {
    "单国别-香港": ("中国香港",),
    "单国别-澳门": ("中国澳门",),
    "单国别-新加坡": ("新加坡",),
    "单国别-马来西亚": ("马来西亚",),
    "单国别-泰国": ("泰国",),
    "联申-港澳": ("中国香港", "中国澳门"),
    "联申-新港澳": ("新加坡", "中国香港", "中国澳门"),
    "高端-新港澳泰": ("新加坡", "中国香港", "中国澳门", "泰国"),
    "博士/研究型硕士": ("新加坡", "中国香港", "中国澳门", "马来西亚"),
}


def _parse_school_range(raw: str) -> tuple[int, int]:
    lo_total = hi_total = 0
    for segment in raw.split("/"):
        nums = [int(n) for n in re.findall(r"\d+", segment)]
        if not nums:
            continue
        lo_total += min(nums)
        hi_total += max(nums)
    return (lo_total or 1, hi_total or 6)


def _parse_money(raw: str) -> int:
    nums = [int(n) for n in re.findall(r"\d+", (raw or "").replace(",", ""))]
    return min(nums) if nums else 0


def _contract_key(source: str, program: str, plan: str = "") -> str:
    keys = {
        ("单国别-香港", "授课式研究生"): "single_hk",
        ("单国别-澳门", "授课式研究生"): "single_mo",
        ("单国别-新加坡", "私立大学研究生"): "single_sg_private",
        ("单国别-新加坡", "公立大学授课式研究生"): "single_sg_public",
        ("联申-港澳", "港澳授课型硕士"): "joint_hkmo",
        ("联申-新港澳", "新港澳授课型硕士"): "joint_hkmo_sg",
    }
    if (source, program) in keys:
        return keys[(source, program)]
    if source == "单国别-马来西亚":
        my_keys = {
            "预科/私立大学研究生": "single_my_private",
            "本科/授课型硕士": "single_my_taught",
            "本科/授课型/混合型硕士": "single_my_taught_full",
        }
        return my_keys.get(program, re.sub(r"\W+", "_", f"{source}_{program}")[:40])
    if source == "高端-新港澳泰":
        return {"A": "premium_a", "B": "premium_b", "C": "premium_c"}.get(plan, "premium_a")
    if source == "博士/研究型硕士":
        return {"A": "research_a", "B": "research_b"}.get(plan, "research_a")
    return re.sub(r"\W+", "_", f"{source}_{program}")[:40]


def _plan_letter(text: str) -> str:
    m = re.search(r"([ABC])计划", text)
    return m.group(1) if m else ""


@lru_cache(maxsize=1)
def load_contracts() -> dict[str, PortfolioContract]:
    contracts: dict[str, PortfolioContract] = {}
    if not _CSV_PATH.exists():
        return contracts

    with _CSV_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source = (row.get("source") or "").strip()
            program = (row.get("program") or "").strip()
            regions = _SOURCE_REGIONS.get(source)
            if regions is None:
                _logger.warning("load_contracts: source %r not mapped, skipped", source)
                continue
            supported_regions = tuple(r for r in regions if r in _SUPPORTED_REGIONS)
            if not supported_regions:
                _logger.warning(
                    "load_contracts: source %r has no supported regions, skipped", source
                )
                continue
            price = _parse_money(row.get("price"))
            lo, hi = _parse_school_range(row.get("schools") or "")
            plan = _plan_letter(row.get("contract") or "")
            key = _contract_key(source, program, plan)
            label = f"{source} · {program}"
            if plan:
                label = f"{source}{plan}计划 · {program}"
            if key in contracts:
                _logger.warning(
                    "load_contracts: duplicate key %r overwritten (source=%r program=%r)",
                    key,
                    source,
                    program,
                )
            contracts[key] = PortfolioContract(
                tier_id=key,
                label=label,
                regions=supported_regions,
                max_schools=hi,
                min_schools=lo,
                max_majors=int(_parse_money(row.get("majors")) or 1),
                price_cny=price,
                reject_deduction=_parse_money(row.get("reject_deduction")),
                visa_denial=_parse_money(row.get("visa_denial")),
                add_school_cost=_parse_money(row.get("add_school")),
                is_joint=len(supported_regions) > 1,
                program=program,
            )
    return dict(contracts)


@lru_cache(maxsize=1)
def _region_map() -> dict[str, str]:
    from src.pages.prediction.page_data_loader import machine_learning_model

    return dict(machine_learning_model.resource_loader().university_country_map)


def university_region(university: str) -> str:
    return _region_map().get(university or "", "未知")


def filter_pool_by_contract(
    pool: list[dict[str, Any]], contract: PortfolioContract
) -> list[dict[str, Any]]:
    allowed = set(contract.regions)
    return [s for s in pool if university_region(s.get("university", "")) in allowed]


def portfolio_k_for_contract(contract: PortfolioContract, pool_size: int) -> int:
    return min(contract.max_schools, pool_size)


_REGION_TARGETS: dict[str, tuple[str, ...]] = {
    "中国香港": (
        "香港大学",
        "香港中文大学",
        "香港科技大学",
        "香港城市大学",
        "香港理工大学",
        "香港浸会大学",
        "香港岭南大学",
        "香港教育大学",
    ),
    "中国澳门": ("澳门大学", "澳门科技大学", "澳门理工大学", "澳门城市大学"),
    "新加坡": ("新加坡国立大学", "新加坡南洋理工大学", "新加坡管理大学"),
    "马来西亚": ("马来亚大学",),
}


def target_universities_for_contract(contract: PortfolioContract) -> list[str]:
    unis: list[str] = []
    for region in contract.regions:
        unis.extend(_REGION_TARGETS.get(region, ()))
    return unis or list(_REGION_TARGETS["中国香港"])
