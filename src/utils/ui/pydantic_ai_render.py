import json

import streamlit as st


def render_execution_trace(trace: list[dict]):
    if not trace:
        return

    icon_map = {
        "thinking": ("#7c3aed", "思考"),
        "text": ("#6b7280", "输出"),
        "tool_call": ("#2563eb", "调用工具"),
        "tool_return": ("#d97706", "工具返回"),
        "final": ("#059669", "最终结果"),
    }

    html_parts = ['<div style="font-family: system-ui, sans-serif; padding: 0.5rem 0;">']
    for _i, step in enumerate(trace):
        color, label = icon_map.get(step["type"], ("#6b7280", step["type"]))

        if step["type"] == "thinking":
            content = step["content"]
            if len(content) > 500:
                content = content[:500] + f" ...（共 {len(step['content'])} 字符）"
            html_parts.append(
                f'<div style="display:flex; align-items:flex-start; margin-bottom:10px;'
                f'border-left:3px solid {color}; padding-left:12px;">'
                ""
                f"<div>"
                f'<span style="font-weight:600; color:{color}; font-size:0.8rem;">{label}</span>'
                f'<p style="margin:2px 0; color:#d1d5db; font-size:0.85rem; white-space:pre-wrap;">{content}</p>'
                f"</div></div>"
            )

        elif step["type"] == "tool_call":
            args_preview = json.dumps(step["args"], ensure_ascii=False, indent=2)
            html_parts.append(
                f'<div style="display:flex; align-items:flex-start; margin-bottom:10px;'
                f'border-left:3px solid {color}; padding-left:12px;">'
                ""
                f'<div style="flex:1;">'
                f'<span style="font-weight:600; color:{color}; font-size:0.8rem;">{label}</span>'
                f'<code style="display:block; margin:2px 0; color:#93c5fd; font-size:0.85rem;">'
                f'{step["name"]}</code>'
                f'<pre style="margin:2px 0; color:#9ca3af; font-size:0.78rem; '
                f'background:#111827; padding:6px; border-radius:6px; overflow-x:auto;">'
                f"{args_preview}</pre>"
                f"</div></div>"
            )

        elif step["type"] == "tool_return":
            html_parts.append(
                f'<div style="display:flex; align-items:flex-start; margin-bottom:10px;'
                f'border-left:3px solid {color}; padding-left:12px;">'
                ""
                f'<div style="flex:1;">'
                f'<span style="font-weight:600; color:{color}; font-size:0.8rem;">'
                f'{label} ← {step["name"]}</span>'
                f'<pre style="margin:2px 0; color:#fbbf24; font-size:0.78rem; '
                f'background:#111827; padding:6px; border-radius:6px; overflow-x:auto;">'
                f'{step["content"]}</pre>'
                f"</div></div>"
            )

        elif step["type"] == "final":
            content = step["content"]
            if len(content) > 300:
                content = content[:300] + "..."
            html_parts.append(
                f'<div style="display:flex; align-items:flex-start; margin-bottom:10px;'
                f'border-left:3px solid {color}; padding-left:12px;">'
                ""
                f"<div>"
                f'<span style="font-weight:600; color:{color}; font-size:0.8rem;">{label}</span>'
                f'<p style="margin:2px 0; color:#d1d5db; font-size:0.78rem;">{content}</p>'
                f"</div></div>"
            )

        elif step["type"] == "text":
            content = step["content"]
            if len(content) > 200:
                content = content[:200] + f" ...（共 {len(step['content'])} 字符）"
            html_parts.append(
                f'<div style="display:flex; align-items:flex-start; margin-bottom:10px;'
                f'border-left:3px solid {color}; padding-left:12px;">'
                ""
                f"<div>"
                f'<p style="margin:2px 0; color:#d1d5db; font-size:0.85rem;">{content}</p>'
                f"</div></div>"
            )

    html_parts.append("</div>")
    st.html("".join(html_parts))


def maybe_render_thinking(out: dict, show: bool, label: str = "思考过程"):
    if not show:
        return
    render_thinking_from_dict(out, label)


def render_thinking_from_dict(out: dict, label: str = "思考过程"):
    thoughts = out.get("thinking", [])
    if not thoughts:
        return
    with st.expander(f"[思考] {label}（{len(thoughts)} 段思考）"):
        for i, t in enumerate(thoughts):
            st.caption(f"消息 #{t['msg_index']} · 第 {i + 1} 段")
            content = t["content"]
            if len(content) > 3000:
                content = content[:3000] + f"\n\n...（截断，共 {len(t['content'])} 字符）"
            st.info(content, icon=":material/info:")


def render_tool_calls(out: dict, show_trace: bool = False):
    if out.get("tool_calls"):
        for i, tc in enumerate(out["tool_calls"]):
            args_str = json.dumps(tc["args"], ensure_ascii=False, indent=2)
            st.info(f"{i + 1}. `{tc['tool']}`\n\n```json\n{args_str}\n```", icon=":material/info:")

    if show_trace and out.get("trace"):
        st.divider()
        st.caption(f"执行轨迹（{len(out['trace'])} 步）")
        render_execution_trace(out["trace"])
