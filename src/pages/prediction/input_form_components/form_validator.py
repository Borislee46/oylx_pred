from src.pages.prediction.config.ui_messages import FORM_ERROR_MESSAGES
from src.pages.prediction.input_form_components.form_config import (
    GMAT_SCORE_RANGE,
    GPA_SCALES,
    GRE_SCORE_RANGE,
    STANDARDIZED_TEST_TYPES,
)
from src.pages.prediction.input_form_components.gpa_converter import GPAConverter
from src.pages.prediction.input_form_components.language_score_validator import (
    LanguageScoreValidator,
)
from src.pages.prediction.input_form_components.validation_errors import ValidationError
from src.utils.logger import setup_logger
from src.utils.school_level_service import get_school_level_service

logger = setup_logger("page3", "prediction")


class FormValidator:
    @staticmethod
    def validate_standardized_test_score(exam_type, score):
        if not score:
            return True, None, None

        if exam_type not in STANDARDIZED_TEST_TYPES:
            return False, FORM_ERROR_MESSAGES["exam_type_invalid"].format(exam_type=exam_type), None

        try:
            parsed_score = float(score)

            if not parsed_score.is_integer():
                return (
                    False,
                    FORM_ERROR_MESSAGES["exam_score_not_integer"].format(exam_type=exam_type),
                    None,
                )

            score_int = int(parsed_score)
            ranges = GRE_SCORE_RANGE if exam_type == "GRE" else GMAT_SCORE_RANGE

            if score_int < 0:
                return (
                    False,
                    FORM_ERROR_MESSAGES["exam_score_negative"].format(exam_type=exam_type),
                    None,
                )

            if score_int < ranges["min"] or score_int > ranges["max"]:
                return (
                    False,
                    FORM_ERROR_MESSAGES["exam_score_out_of_range"].format(
                        exam_type=exam_type, min=ranges["min"], max=ranges["max"]
                    ),
                    None,
                )

            return True, None, parsed_score

        except ValueError:
            return False, f"{exam_type}分数无效，请输入整数", None

    @staticmethod
    def normalize_gpa(raw_gpa, scale_key, background_university=None, gpa_converter=None):
        if raw_gpa is None or raw_gpa == "":
            return None

        if scale_key is None or scale_key == "":
            return None

        if background_university and gpa_converter:
            country = gpa_converter.get_university_country(background_university)

            json_result = GPAConverter.convert_gpa_by_rules(
                raw_gpa, scale_key, background_university, country
            )
            if json_result is not None:
                return json_result

        try:
            raw_gpa = float(raw_gpa)
            original_max_gpa_scale = float(GPA_SCALES[scale_key]["max"])

            if original_max_gpa_scale > 0:
                normalized_gpa = (raw_gpa / original_max_gpa_scale) * 4.0
                return round(max(0, min(normalized_gpa, 4.0)), 2)
            else:
                return 0.0
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"错误：标准GPA归一化失败: {e}")
            return None

    @staticmethod
    def validate_form_data(form_data, gpa_converter=None) -> list[ValidationError]:
        errors: list[ValidationError] = []

        if not form_data["background_university"]:
            errors.append(
                ValidationError(
                    "background_university", FORM_ERROR_MESSAGES["background_university_empty"]
                )
            )

        if not form_data["background_major_original"]:
            errors.append(
                ValidationError(
                    "background_major_original", FORM_ERROR_MESSAGES["background_major_empty"]
                )
            )
        elif not form_data["background_major"]:
            errors.append(
                ValidationError("background_major", FORM_ERROR_MESSAGES["background_major_invalid"])
            )

        if form_data["gpa_raw"] is None or form_data["gpa_raw"] == "":
            errors.append(ValidationError("gpa_raw", FORM_ERROR_MESSAGES["gpa_empty"]))
        elif form_data["gpa_raw"] == 0:
            errors.append(ValidationError("gpa_raw", FORM_ERROR_MESSAGES["gpa_zero"]))
        else:
            normalized_gpa = FormValidator.normalize_gpa(
                form_data["gpa_raw"],
                form_data["gpa_scale"],
                form_data.get("background_university"),
                gpa_converter,
            )
            if normalized_gpa is None:
                errors.append(ValidationError("gpa_raw", FORM_ERROR_MESSAGES["gpa_parse_failed"]))
            elif normalized_gpa == 0.0 and form_data["gpa_raw"] > 0:
                errors.append(
                    ValidationError("gpa_scale", FORM_ERROR_MESSAGES["gpa_scale_invalid"])
                )

        exam_type = form_data.get("exam_type")
        exam_score = form_data.get("exam_score")
        if exam_score:
            is_valid, error_msg, _ = FormValidator.validate_standardized_test_score(
                exam_type, exam_score
            )
            if not is_valid:
                errors.append(ValidationError("exam_score", error_msg))

        school_service = get_school_level_service()
        background_university = form_data.get("background_university")
        is_overseas = (
            school_service.is_overseas_school(background_university)
            if background_university
            else False
        )

        if form_data.get("language_score_input_error"):
            errors.append(
                ValidationError(
                    "language_score_input", FORM_ERROR_MESSAGES["language_score_input_error"]
                )
            )

        if form_data["language_type"] == "雅思" and form_data["language_score_raw"] is not None:
            if form_data[
                "language_score_raw"
            ] > 0 and not LanguageScoreValidator.validate_ielts_step(
                form_data["language_score_raw"]
            ):
                errors.append(
                    ValidationError("language_score_raw", FORM_ERROR_MESSAGES["ielts_step_invalid"])
                )

        if form_data["language_score_raw"] == 0 and not is_overseas:
            errors.append(
                ValidationError(
                    "language_score_raw",
                    FORM_ERROR_MESSAGES["language_score_zero"].format(
                        language_type=form_data["language_type"]
                    ),
                )
            )

        experience_fields = [
            "research_count",
            "award_count",
            "internship_count",
            "paper_count",
        ]
        field_names = [
            FORM_ERROR_MESSAGES["experience_field_research"],
            FORM_ERROR_MESSAGES["experience_field_award"],
            FORM_ERROR_MESSAGES["experience_field_internship"],
            FORM_ERROR_MESSAGES["experience_field_paper"],
        ]

        for field, name in zip(experience_fields, field_names, strict=False):
            if form_data[field] is None:
                errors.append(ValidationError(field, f"{name}不能为空"))

        if "experience_details" in form_data and isinstance(form_data["experience_details"], dict):
            experience_checks = [
                ("research_count", "research_details", "科研项目"),
                ("award_count", "award_details", "获奖情况"),
                ("internship_count", "internship_details", "实习经历"),
                ("paper_count", "paper_details", "论文发表"),
            ]

            for count_field, detail_field, field_name in experience_checks:
                count_value = form_data.get(count_field, 0)
                detail_value = form_data["experience_details"].get(detail_field, "").strip()

                if count_value == 0 and detail_value:
                    errors.append(
                        ValidationError(
                            count_field,
                            FORM_ERROR_MESSAGES["experience_detail_mismatch"].format(
                                field_name=field_name
                            ),
                        )
                    )

        return errors
