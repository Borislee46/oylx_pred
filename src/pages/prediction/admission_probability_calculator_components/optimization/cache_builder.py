from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class CacheBuilder:
    @staticmethod
    def build_major_category_cache(all_schools_data: list) -> dict:
        major_category_cache = {}
        for school in all_schools_data:
            uni = school.get("university", "")
            major = school.get("major", "")
            faculty = school.get("faculty", "")
            if uni and major and faculty:
                key = f"{uni}|{major}"
                major_category_cache[key] = faculty
        return major_category_cache

    @staticmethod
    def get_bg_target_similarity_cache() -> dict:
        try:
            from src.pages.prediction.page_data_loader import (
                cached_load_bg_target_similarity_cache,
            )

            return cached_load_bg_target_similarity_cache()
        except Exception as e:
            logger.warning(f"加载bg_target_similarity_cache时出错: {e}")
            return {}
