from pydantic import BaseModel, Field, computed_field

INTENTS = ("profile", "question", "vague", "off_topic")
NEXT_ACTIONS = ("fill_and_predict", "fill_only", "ask_clarification", "answer")
CONFIDENCES = ("high", "medium", "low")


class LeadInDecision(BaseModel):
    model_config = {"extra": "allow"}

    intent: str = "profile"
    feedback: str = ""
    clarifying_question: str = ""
    missing_required: list[str] = Field(default_factory=list)
    confidence: str = "medium"

    university: str = ""
    major: str = ""
    major_2: str = ""
    degree_type: str = ""
    gpa: float | None = None
    gpa_scale: str = ""
    language_type: str = ""
    language_score: float | None = None
    standardized_test_type: str = ""
    standardized_test_score: float | None = None
    country: str = ""
    target_schools: list[str] = Field(default_factory=list)
    target_majors: list[str] = Field(default_factory=list)
    research: str = ""
    research_count: int | None = None
    internship: str = ""
    internship_count: int | None = None
    paper: str = ""
    paper_count: int | None = None
    award: str = ""
    award_count: int | None = None

    def core_missing(self) -> list[str]:
        missing: list[str] = []
        if not str(self.university or "").strip():
            missing.append("院校")
        if not str(self.major or "").strip():
            missing.append("专业")
        if self.gpa in (None, "", 0, 0.0):
            missing.append("GPA")
        if self.language_score in (None, "", 0, 0.0):
            missing.append("语言成绩")
        return missing

    @computed_field
    @property
    def next_action(self) -> str:
        if self.intent == "question":
            return "answer"
        if self.intent in ("vague", "off_topic"):
            return "ask_clarification"
        if self.intent == "profile":
            if not self.core_missing() and self.confidence in ("high", "medium"):
                return "fill_and_predict"
            return "fill_only"
        return "ask_clarification"

    def to_extracted_info(self) -> dict:
        decision_fields = {
            "intent",
            "next_action",
            "feedback",
            "clarifying_question",
            "missing_required",
            "confidence",
        }
        data = self.model_dump(
            exclude=decision_fields,
            exclude_none=True,
            mode="python",
        )
        return {
            k: v
            for k, v in data.items()
            if v not in ("", []) and not k.startswith("should_predict")
        }

    @computed_field
    @property
    def should_predict(self) -> bool:
        return (
            self.intent == "profile"
            and not self.core_missing()
            and self.confidence in ("high", "medium")
        )
