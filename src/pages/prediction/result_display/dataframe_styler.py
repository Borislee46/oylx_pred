import streamlit as st


class DataFrameStyler:
    @staticmethod
    def create_styled_dataframe(df):
        if df.empty:
            return df

        style_needed = False
        style_columns = []

        major_column = None
        for col in df.columns:
            if col == "推荐专业" or col.startswith("推荐专业"):
                major_column = col
                break

        if major_column and df[major_column].astype(str).str.contains("(new!)", regex=False).any():
            style_needed = True
            style_columns.append(major_column)

        if not style_needed:
            return df

        def style_cells(val):
            if isinstance(val, str):
                if "(new!)" in val:
                    return "color: #06b6d4;"
            return ""

        return df.style.map(style_cells, subset=style_columns)

    @staticmethod
    def get_column_config(df, column_widths=None, label_map=None):
        column_widths = column_widths or {}
        label_map = label_map or {}

        if "推荐专业详情" in df.columns and "推荐专业详情" not in column_widths:
            column_widths["推荐专业详情"] = "large"

        column_config = {}
        for col_name in df.columns:
            width = column_widths.get(col_name, "small")
            label = label_map.get(col_name)

            if col_name == "推荐专业详情":
                column_config[col_name] = st.column_config.TextColumn(
                    label=label, width=width, max_chars=None
                )
            elif col_name == "录取概率":
                column_config[col_name] = st.column_config.ProgressColumn(
                    label=label,
                    width=width,
                    min_value=0,
                    max_value=1,
                    format=" ",
                    pinned=True,
                    color="#06b6d4",
                )
            else:
                column_config[col_name] = st.column_config.TextColumn(label=label, width=width)

        return column_config
