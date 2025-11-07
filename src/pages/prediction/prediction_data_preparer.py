from src.pages.prediction.prediction_exceptions import InvalidInputError
from src.pages.prediction.prediction_utils import get_background_faculty
from src.utils.app_data_loader import load_raw_cases_data
from src.utils.logger import setup_logger
from src.utils.school_level_service import get_school_level_service

prediction_handler_logger = setup_logger("page3", "prediction")


def prepare_input_data(input_data_from_form: dict) -> dict:
    if not isinstance(input_data_from_form, dict):
        raise InvalidInputError("输入数据必须是字典类型")

    required_fields = ["background_university", "background_major"]
    missing_fields = [field for field in required_fields if not input_data_from_form.get(field)]

    if missing_fields:
        error_msg = f"缺少必需字段: {', '.join(missing_fields)}"
        prediction_handler_logger.warning(error_msg)
        raise InvalidInputError(error_msg)

    background_uni_name = input_data_from_form.get("background_university", "")
    if not background_uni_name or not isinstance(background_uni_name, str):
        raise InvalidInputError("background_university 必须是非空字符串")

    service = get_school_level_service()
    school_level = service.get_school_level(background_uni_name)

    input_data = input_data_from_form.copy()
    input_data["school_level"] = school_level

    background_major = input_data.get("background_major")
    if background_major:
        if not isinstance(background_major, str):
            prediction_handler_logger.warning(
                f"background_major 类型不正确: {type(background_major)}"
            )
        else:
            cases_df = load_raw_cases_data()
            faculty = get_background_faculty(background_major, cases_df)
            if faculty:
                input_data["faculty"] = faculty

    return input_data

