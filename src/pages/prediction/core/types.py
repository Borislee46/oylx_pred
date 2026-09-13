from typing import TypedDict


class PredictionInput(TypedDict, total=False):
    background_university: str
    background_major: str
    target_universities: list[str]
    target_majors: list[str]
    gpa: float
    gpa_model: float
    language_score: float
    language_type: str
    internship_count: int
    research_count: int
    award_count: int
    paper_count: int
    school_level: str
    experience_details: dict[str, str]
