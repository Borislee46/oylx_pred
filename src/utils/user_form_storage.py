import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class UserFormStorage:
    def __init__(self, storage_dir: str = "user_form_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    def _get_user_file_path(self, user_id: str) -> Path:
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ("-", "_"))
        return self.storage_dir / f"{safe_user_id}_form_data.json"

    def _retry_operation(
        self,
        operation: Callable[[], Any],
        operation_name: str,
        user_id: str,
        max_retries: int,
        retry_delay: float,
        is_save: bool = False,
    ) -> Any:
        for attempt in range(max_retries):
            try:
                return operation()
            except json.JSONDecodeError as e:
                logger.error(f"用户 {user_id} 表单数据 JSON 解析失败: {str(e)}")
                return None
            except Exception as e:
                if attempt < max_retries - 1:
                    delay_multiplier = attempt + 1 if is_save else 1
                    logger.warning(
                        f"{operation_name}用户 {user_id} 表单数据失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}, 将重试"
                    )
                    time.sleep(retry_delay * delay_multiplier)
                else:
                    logger.error(
                        f"{operation_name}用户 {user_id} 表单数据时出错: {str(e)}",
                        exc_info=True,
                    )
                    return False if operation_name in ("保存", "删除") else None
        return False if operation_name in ("保存", "删除") else None

    def save_form_data(self, user_id: str, form_data: dict[str, Any]) -> bool:
        def _save() -> bool:
            temp_file = None
            try:
                file_path = self._get_user_file_path(user_id)
                storage_data = {
                    "user_id": user_id,
                    "last_updated": datetime.now(UTC).isoformat(),
                    "form_data": self._clean_form_data(form_data),
                }
                fd, temp_file = tempfile.mkstemp(
                    suffix=".json", prefix="form_", dir=str(self.storage_dir), text=True
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(storage_data, f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())
                except Exception:
                    os.close(fd)
                    raise

                os.replace(temp_file, file_path)
                temp_file = None
                return True
            finally:
                if temp_file and os.path.exists(temp_file):
                    os.unlink(temp_file)

        return self._retry_operation(
            operation=_save,
            operation_name="保存",
            user_id=user_id,
            max_retries=3,
            retry_delay=0.1,
            is_save=True,
        )

    def load_form_data(self, user_id: str) -> dict[str, Any] | None:
        def _load() -> dict[str, Any] | None:
            file_path = self._get_user_file_path(user_id)
            if not file_path.exists():
                return None

            with open(file_path, encoding="utf-8") as f:
                storage_data = json.load(f)

            if not isinstance(storage_data, dict):
                logger.error(f"用户 {user_id} 数据格式错误: 期望 dict, 实际 {type(storage_data)}")
                return None

            return storage_data.get("form_data", {})

        return self._retry_operation(
            operation=_load,
            operation_name="加载",
            user_id=user_id,
            max_retries=2,
            retry_delay=0.1,
        )

    def _clean_form_data(self, form_data: dict[str, Any]) -> dict[str, Any]:
        fields_to_save = {
            "target_majors",
            "target_universities",
            "selected_target_countries",
            "selected_major_categories",
            "selected_target_universities",
            "selected_target_majors",
            "background_university",
            "background_major_original",
            "background_major",
            "gpa_raw",
            "gpa_scale",
            "language_type",
            "language_score_raw",
            "research_count",
            "award_count",
            "internship_count",
            "paper_count",
        }

        cleaned_data = {}
        for field in fields_to_save:
            if field in form_data and form_data[field] is not None:
                cleaned_data[field] = form_data[field]

        experience_details = form_data.get("experience_details", {})
        if isinstance(experience_details, dict):
            cleaned_experience = {}
            for detail_field in [
                "research_details",
                "award_details",
                "internship_details",
                "paper_details",
            ]:
                if detail_field in experience_details:
                    detail_value = experience_details[detail_field]
                    if detail_value and isinstance(detail_value, str) and detail_value.strip():
                        cleaned_experience[detail_field] = detail_value.strip()

            if cleaned_experience:
                cleaned_data["experience_details"] = cleaned_experience

        return cleaned_data

    def delete_user_data(self, user_id: str) -> bool:
        def _delete() -> bool:
            file_path = self._get_user_file_path(user_id)
            if not file_path.exists():
                return True
            file_path.unlink()
            logger.info(f"成功删除用户 {user_id} 的表单数据")
            return True

        return self._retry_operation(
            operation=_delete,
            operation_name="删除",
            user_id=user_id,
            max_retries=2,
            retry_delay=0.1,
        )
