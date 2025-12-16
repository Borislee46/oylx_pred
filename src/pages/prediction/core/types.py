from typing import TypedDict


class PredictionInput(TypedDict, total=False):
    background_university: str
    background_major: str
    target_universities: list[str]
    target_majors: list[str]
    gpa: float
    language_score: float
    internship_count: int
    research_count: int
    award_count: int
    paper_count: int
    school_level: int
    experience_details: dict[str, str]
