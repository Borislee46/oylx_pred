from src.pages.prediction.prediction_result_display import ResultsDisplay


def display_results_section(
    input_data, sim_results, cross_results, user_specified_results, cases_df
):
    if all(x is None for x in [sim_results, cross_results, user_specified_results]):
        return

    results_display = ResultsDisplay(
        top_similarity_results=sim_results,
        top_cross_major_results=cross_results,
        user_specified_results=user_specified_results,
    )

    results_display.display(
        input_data.get("target_universities", []),
        input_data.get("target_majors", []),
        input_data.get("gpa"),
        input_data.get("language_score"),
        input_data.get("language_type"),
        background_university=input_data.get("background_university"),
    )
