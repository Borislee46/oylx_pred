# !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files.
import json

from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class SchoolSpecificContentGenerator:
    def __init__(self, file_path: str = "config/school_specific_content.json"):
        self.school_db = self._load_school_content_from_file(file_path)

    def _load_school_content_from_file(self, file_path: str) -> dict:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return data
