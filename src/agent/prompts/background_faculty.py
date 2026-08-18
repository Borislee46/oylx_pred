def build_background_faculty_prompt(
    background_major_original: str,
    base_faculty: str | None,
    valid_faculties: list[str],
    max_extra: int = 2,
) -> str:
    base = (base_faculty or "").strip()
    major = (background_major_original or "").strip()
    max_extra = int(max_extra) if max_extra is not None else 2
    if max_extra < 0:
        max_extra = 0

    return (
        f"background_major_original: {major}\n"
        f"base_faculty: {base}\n"
        f"max_extra_faculties: {max_extra}\n"
        f"valid_faculties (only pick from this list): {valid_faculties}"
    )
