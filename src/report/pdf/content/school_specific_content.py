import json
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class SchoolSpecificContentGenerator:
    def __init__(self, file_path: str | None = None):
        if file_path is None:
            from ..config import _PROJECT_ROOT

            file_path = str(_PROJECT_ROOT / "config" / "school_specific_content.json")
        self.school_db = self._load_school_content_from_file(file_path)

    def _load_school_content_from_file(self, file_path: str) -> dict:
        if not Path(file_path).exists():
            logger.warning("院校特定内容文件缺失，跳过: %s", file_path)
            return {}
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return data
