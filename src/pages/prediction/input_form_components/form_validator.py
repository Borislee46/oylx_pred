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
            return False, f"{exam_type}分数无效，请选择有效的考试类型", None

        try:
            parsed_score = float(score)

            if not parsed_score.is_integer():
                return False, f"{exam_type}分数无效，请输入整数", None

            score_int = int(parsed_score)
            ranges = GRE_SCORE_RANGE if exam_type == "GRE" else GMAT_SCORE_RANGE

            if score_int < 0:
                return False, f"{exam_type}分数无效，请输入大于0的整数", None

            if score_int < ranges["min"] or score_int > ranges["max"]:
                return (
                    False,
                    f"{exam_type}分数无效，请输入 {ranges['min']} - {ranges['max']} 之间的整数",
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
            errors.append(ValidationError("background_university", "请选择背景院校"))

        if not form_data["background_major_original"]:
            errors.append(ValidationError("background_major_original", "请选择背景专业"))
        elif not form_data["background_major"]:
            errors.append(ValidationError("background_major", "背景专业选择无效，请重新选择"))

        if form_data["gpa_raw"] is None or form_data["gpa_raw"] == "":
            errors.append(ValidationError("gpa_raw", "GPA不能为空"))
        elif form_data["gpa_raw"] == 0:
            errors.append(ValidationError("gpa_raw", "GPA不能为0"))
        else:
            normalized_gpa = FormValidator.normalize_gpa(
                form_data["gpa_raw"],
                form_data["gpa_scale"],
                form_data.get("background_university"),
                gpa_converter,
            )
            if normalized_gpa is None:
                errors.append(ValidationError("gpa_raw", "GPA无法解析或归一化，请检查输入与分制"))
            elif normalized_gpa == 0.0 and form_data["gpa_raw"] > 0:
                errors.append(ValidationError("gpa_scale", "GPA分制无效"))

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
            errors.append(ValidationError("language_score_input", "请修正语言成绩输入错误"))

        if form_data["language_type"] == "雅思" and form_data["language_score_raw"] is not None:
            if form_data[
                "language_score_raw"
            ] > 0 and not LanguageScoreValidator.validate_ielts_step(
                form_data["language_score_raw"]
            ):
                errors.append(ValidationError("language_score_raw", "雅思成绩必须是0.5的倍数"))

        if form_data["language_score_raw"] == 0 and not is_overseas:
            errors.append(
                ValidationError("language_score_raw", f"{form_data['language_type']}成绩不能为0")
            )

        experience_fields = [
            "research_count",
            "award_count",
            "internship_count",
            "paper_count",
        ]
        field_names = ["科研项目数量", "获奖数量", "实习数量", "论文数量"]

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
                            f"{field_name}数量为0，但填写了详细信息，请检查数量或清空详细信息",
                        )
                    )

        return errors
