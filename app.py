import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AI Data Quality Investigator",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 AI Data Quality Investigator")
st.write(
    "Upload a CSV or Excel file to investigate data quality "
    "problems and preview cleaning."
)

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.success(f"Loaded: {uploaded_file.name}")

    st.subheader("Dataset preview")
    st.dataframe(data.head(10), use_container_width=True)

    st.subheader("Dataset overview")
    col1, col2 = st.columns(2)

    col1.metric("Rows", data.shape[0])
    col2.metric("Columns", data.shape[1])

    if st.button("Investigate dataset"):
        st.subheader("Missing values")
        missing = data.isna().sum()
        missing = missing[missing > 0]

        if missing.empty:
            st.success("No missing values detected.")
        else:
            st.dataframe(
                missing.rename("Missing count").reset_index(
                    names="Column"
                ),
                use_container_width=True
            )

        st.subheader("Duplicate rows")
        duplicates = int(data.duplicated().sum())
        st.metric("Exact duplicate rows", duplicates)

        st.subheader("Text with extra spaces")
        whitespace = []

        for column in data.select_dtypes(
            include=["object", "string"]
        ).columns:
            count = data[column].apply(
                lambda value: isinstance(value, str)
                and value != value.strip()
            ).sum()

            if count > 0:
                whitespace.append({
                    "Column": column,
                    "Affected values": int(count)
                })

        if whitespace:
            st.dataframe(
                pd.DataFrame(whitespace),
                use_container_width=True
            )
        else:
            st.success("No extra spaces detected in text values.")
