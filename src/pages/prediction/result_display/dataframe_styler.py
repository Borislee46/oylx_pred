import streamlit as st


class DataFrameStyler:
    @staticmethod
    def create_styled_dataframe(df):
        if df.empty:
            return df

        style_needed = False
        style_columns = []

        if "目标专业" in df.columns:
            if df["目标专业"].astype(str).str.contains("(New!)", regex=False).any():
                style_needed = True
                style_columns.append("目标专业")

        if "±%" in df.columns:
            if df["±%"].astype(str).str.strip().ne("").any():
                style_needed = True
                style_columns.append("±%")

        if not style_needed:
            return df

        def style_cells(val):
            if isinstance(val, str):
                if "(New!)" in val:
                    return "color: #FF4B4B; font-weight: bold;"
                elif val.startswith("+"):
                    return "color: #28a745; font-weight: bold;"
                elif val.startswith("-"):
                    return "color: #dc3545; font-weight: bold;"
            return ""

        return df.style.map(style_cells, subset=style_columns)

    @staticmethod
    def get_column_config(df, column_widths=None):
        column_widths = column_widths or {}

        if "专业详情" in df.columns and "专业详情" not in column_widths:
            column_widths["专业详情"] = "large"
        if "±%" in df.columns and "±%" not in column_widths:
            column_widths["±%"] = "small"

        column_config = {}
        for col_name in df.columns:
            width = column_widths.get(col_name, "small")

            if col_name == "专业详情":
                column_config[col_name] = st.column_config.TextColumn(
                    width=width, help="学校和专业的详细信息", max_chars=None
                )
            elif col_name == "录取概率":
                column_config[col_name] = st.column_config.ProgressColumn(
                    width=width, min_value=0, max_value=1, format=" "
                )
            elif col_name == "±%":
                column_config[col_name] = st.column_config.TextColumn(
                    width=width, help="相对上次的概率±%（±%）"
                )
            else:
                column_config[col_name] = st.column_config.TextColumn(width=width)

        return column_config
