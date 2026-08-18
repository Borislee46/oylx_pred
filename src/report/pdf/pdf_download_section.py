from datetime import datetime
from typing import Any

import streamlit as st

from src.utils.logger import setup_logger
from src.utils.numeric import prob_round
from src.utils.session_manager import SessionManager

logger = setup_logger("page3", "prediction")

_PDF_BYTES_KEY = "_hk_pdf_bytes"
_PDF_HASH_KEY = "_hk_pdf_hash"
_PDF_NAME_KEY = "_hk_pdf_name"
_PDF_ERROR_KEY = "_hk_pdf_error"
_GENERATING_KEY = "_hk_pdf_generating"


def _pdf_results_hash(unified: list[dict], nick: str) -> str:
    import hashlib
    import json

    payload = [
        (
            str(r.get("university", "")),
            str(r.get("major", "")),
            prob_round(r.get("probability")),
        )
        for r in unified
    ]
    return hashlib.md5(
        json.dumps([payload, nick], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


_PDF_LABEL = ("生成家长版报告", "下载家长版报告", "Signals择校报告_家长版")


def render_pdf_download_button(
    session_manager: SessionManager,
    res_model: Any = None,
    input_data: dict | None = None,
    key_suffix: str = "",
) -> None:
    if res_model is None or input_data is None:
        return
    unified = getattr(res_model, "unified_results", None) or []
    if not unified:
        return

    gen_label, dl_label, name_prefix = _PDF_LABEL
    nick = (
        session_manager.get("user_nickname") or st.session_state.get("e2_user_nickname") or "用户"
    )
    cur_hash = _pdf_results_hash(unified, nick)
    key_part = f"_{key_suffix}" if key_suffix else ""
    bytes_key = _PDF_BYTES_KEY + key_part
    hash_key = _PDF_HASH_KEY + key_part
    name_key = _PDF_NAME_KEY + key_part
    error_key = _PDF_ERROR_KEY + key_part
    generating_key = _GENERATING_KEY + key_part

    if st.session_state.get(hash_key) == cur_hash and st.session_state.get(bytes_key):
        st.download_button(
            label=dl_label,
            data=st.session_state[bytes_key],
            file_name=st.session_state.get(name_key, "report.pdf"),
            mime="application/pdf",
            width='stretch',
            type="primary",
            key=f"hk_pdf_download_btn{key_part}",
        )
        return

    if st.button(gen_label, width='stretch', key=f"hk_pdf_gen_btn{key_part}", type="primary"):
        st.session_state[generating_key] = True
        st.rerun(scope="fragment")

    if st.session_state.get(generating_key):
        from src.report.pdf.generators.pdf_report_generator import (
            PDFReportGenerator,
        )

        st.markdown(
            '<div id="hk-pin-pdf"></div>'
            "<script>"
            "setTimeout(function(){"
            'var el=document.getElementById("hk-pin-pdf");'
            'if(el)el.scrollIntoView({behavior:"smooth",block:"start"});'
            "},50);"
            "</script>",
            unsafe_allow_html=True,
        )

        pdf_bytes = None
        try:
            with st.skeleton(height=120):
                st.caption("正在生成 PDF 报告，请稍候...")
                pdf_bytes = PDFReportGenerator().generate_report_from_pathfinder(
                    unified_results=unified, input_data=input_data, user_nickname=nick
                )
        except Exception:
            logger.exception("PDF generation failed")

        st.session_state.pop(generating_key, None)
        if pdf_bytes:
            date_str = datetime.now().strftime("%Y%m%d")
            st.session_state[bytes_key] = pdf_bytes
            st.session_state[hash_key] = cur_hash
            st.session_state[name_key] = f"{name_prefix}_{nick}_{date_str}.pdf"
            st.session_state.pop(error_key, None)
            try:
                from src.utils.analytics import track as _t

                _t("pdf_generated", school_count=len(unified))
            except Exception:
                pass
        else:
            st.session_state[error_key] = True
        st.rerun(scope="fragment")

    if st.session_state.get(error_key):
        st.caption("PDF 生成失败，请稍后重试。")
