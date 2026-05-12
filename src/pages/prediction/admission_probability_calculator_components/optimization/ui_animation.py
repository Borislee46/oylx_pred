# !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files.
import time

import streamlit as st


class OptimizationUIAnimation:
    @staticmethod
    def animate_step(
        placeholder: st.delta_generator.DeltaGenerator, base_text: str, duration: float
    ) -> None:
        interval = 0.3
        start_time = time.time()
        cycle_count = 0

        while time.time() - start_time < duration:
            dots = "." * ((cycle_count % 3) + 1)
            placeholder.markdown(
                f"<p style='font-size: 14px; font-weight: normal;'>{base_text}{dots}</p>",
                unsafe_allow_html=True,
            )
            time.sleep(interval)
            cycle_count += 1

        placeholder.markdown(
            f"<p style='font-size: 14px; font-weight: normal;'>{base_text}...</p>",
            unsafe_allow_html=True,
        )

    @staticmethod
    def update_step(
        step_placeholder: st.delta_generator.DeltaGenerator,
        text: str,
        animate: bool,
        duration: float = 0.7,
    ) -> None:
        if animate:
            OptimizationUIAnimation.animate_step(step_placeholder, text, duration)
        else:
            step_placeholder.markdown(
                f"<p style='font-size: 14px; font-weight: normal;'>{text}...</p>",
                unsafe_allow_html=True,
            )
