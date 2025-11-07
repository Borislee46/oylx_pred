def persist_input_state(
    session_manager,
    current_input_data: dict,
    session_key_input_data: str,
    session_key_is_school_selection_submit: str,
) -> None:
    session_manager.set(
        **{
            session_key_input_data: current_input_data,
            session_key_is_school_selection_submit: False,
        }
    )

