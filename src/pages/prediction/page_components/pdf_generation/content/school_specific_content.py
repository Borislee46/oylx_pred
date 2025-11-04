import json
from pathlib import Path
from typing import Dict
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")

class SchoolSpecificContentGenerator:
    
    def __init__(self, file_path="config/school_specific_content.json"):
        self.school_db = self._load_school_content_from_file(file_path)
    
    def _load_school_content_from_file(self, file_path: str) -> Dict:
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                logger.warning(f"学校特定内容文件不存在: {file_path}，使用空字典")
                return {}
            
            if not file_path_obj.is_file():
                logger.warning(f"路径不是文件: {file_path}，使用空字典")
                return {}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    logger.warning(f"文件内容格式不正确，期望字典类型: {file_path}，使用空字典")
                    return {}
                return data
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败 {file_path}: {str(e)}，使用空字典", exc_info=True)
            return {}
        except PermissionError as e:
            logger.error(f"文件权限不足 {file_path}: {str(e)}，使用空字典", exc_info=True)
            return {}
        except IOError as e:
            logger.error(f"文件读取失败 {file_path}: {str(e)}，使用空字典", exc_info=True)
            return {}
        except Exception as e:
            logger.error(f"加载学校特定内容文件时发生未知错误 {file_path}: {str(e)}，使用空字典", exc_info=True)
            return {}

