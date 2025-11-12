import streamlit as st
from streamlit_elements import dashboard, elements, mui, nivo, sync


def render_draggable_nivo_charts():
    st.subheader("nivo test")

    if "nivo_dashboard_layout" not in st.session_state:
        st.session_state.nivo_dashboard_layout = [
            dashboard.Item("pie_chart", 0, 0, 6, 5),
            dashboard.Item("bar_chart", 6, 0, 6, 5),
        ]

    chart_data = [
        {"id": "A", "label": "A", "value": 35},
        {"id": "B", "label": "B", "value": 25},
        {"id": "C", "label": "C", "value": 20},
        {"id": "D", "label": "D", "value": 20},
    ]

    with elements("draggable_nivo_charts"):
        with dashboard.Grid(
            st.session_state.nivo_dashboard_layout, onLayoutChange=sync("nivo_dashboard_layout")
        ):
            with mui.Paper(
                key="pie_chart",
                sx={
                    "height": "100%",
                    "display": "flex",
                    "flexDirection": "column",
                    "border": "2px solid #1976d2",
                    "borderRadius": "8px",
                    "boxShadow": "0 4px 12px rgba(25, 118, 210, 0.15)",
                    "overflow": "hidden",
                    "backgroundColor": "#ffffff",
                },
            ):
                with mui.Box(
                    sx={
                        "backgroundColor": "#1976d2",
                        "color": "#ffffff",
                        "padding": "12px 16px",
                        "borderBottom": "2px solid #1565c0",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "space-between",
                    }
                ):
                    mui.Typography("饼图分析", variant="h6", sx={"fontWeight": 600})
                    mui.Chip(
                        label=f"总计: {sum(item['value'] for item in chart_data)}",
                        size="small",
                        sx={
                            "backgroundColor": "rgba(255, 255, 255, 0.2)",
                            "color": "#ffffff",
                            "fontWeight": 500,
                        },
                    )

                with mui.Box(
                    sx={
                        "flex": 1,
                        "padding": "16px",
                        "display": "flex",
                        "flexDirection": "column",
                        "minHeight": 0,
                        "backgroundColor": "#fafafa",
                    }
                ):
                    nivo.Pie(
                        data=chart_data,
                        margin={"top": 20, "right": 80, "bottom": 80, "left": 80},
                        innerRadius=0.5,
                        padAngle=0.7,
                        cornerRadius=3,
                        activeOuterRadiusOffset=8,
                        colors={"scheme": "nivo"},
                        borderWidth=1,
                        borderColor={"theme": "background"},
                        enableArcLinkLabels=True,
                        arcLinkLabelsSkipAngle=10,
                        arcLinkLabelsTextColor="#333333",
                        arcLinkLabelsThickness=2,
                        arcLinkLabelsColor={"from": "color"},
                        enableArcLabels=True,
                        arcLabelsSkipAngle=10,
                        arcLabelsTextColor={"from": "color", "modifiers": [["darker", 1.2]]},
                        legends=[
                            {
                                "anchor": "bottom",
                                "direction": "row",
                                "translateY": 56,
                                "itemWidth": 100,
                                "itemHeight": 18,
                                "itemTextColor": "#666",
                                "symbolSize": 18,
                                "symbolShape": "circle",
                            }
                        ],
                    )
