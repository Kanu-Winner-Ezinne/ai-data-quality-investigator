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
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.success(f"Loaded: {uploaded_file.name}")

        # Dataset preview
        st.subheader("Dataset preview")
        st.dataframe(data.head(10), use_container_width=True)

        # Dataset overview
        st.subheader("Dataset overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", data.shape[0])
        col2.metric("Columns", data.shape[1])
        col3.metric("Duplicate rows", int(data.duplicated().sum()))

        # Run investigation
        if st.button("Investigate dataset", type="primary"):

            st.header("Data Quality Investigation")

            # 1. Missing values
            st.subheader("1. Missing values")

            missing = data.isna().sum()
            missing = missing[missing > 0]

            if missing.empty:
                st.success("No missing values detected.")
            else:
                missing_report = (
                    missing.rename("Missing count")
                    .rename_axis("Column")
                    .reset_index()
                )

                missing_report["Percentage"] = (
                    missing_report["Missing count"]
                    / len(data)
                    * 100
                ).round(2)

                st.dataframe(
                    missing_report,
                    use_container_width=True
                )

            # 2. Duplicate rows
            st.subheader("2. Duplicate rows")

            duplicate_count = int(data.duplicated().sum())

            if duplicate_count == 0:
                st.success("No exact duplicate rows detected.")
            else:
                st.warning(
                    f"Found {duplicate_count} exact duplicate rows."
                )

                with st.expander("Preview duplicate rows"):
                    st.dataframe(
                        data[data.duplicated(keep=False)],
                        use_container_width=True
                    )

            # 3. Extra whitespace
            st.subheader("3. Text with extra spaces")

            whitespace_issues = []

            text_columns = data.select_dtypes(
                include=["object", "string"]
            ).columns

            for column in text_columns:
                values = data[column]

                has_extra_spaces = values.apply(
                    lambda value: (
                        isinstance(value, str)
                        and value != value.strip()
                    )
                )

                count = int(has_extra_spaces.sum())

                if count > 0:
                    whitespace_issues.append({
                        "Column": column,
                        "Affected values": count
                    })

            if whitespace_issues:
                st.warning(
                    "Some text values contain leading or trailing spaces."
                )

                st.dataframe(
                    pd.DataFrame(whitespace_issues),
                    use_container_width=True
                )
            else:
                st.success(
                    "No leading or trailing spaces detected."
                )

            # 4. Suspicious negative numeric values
            st.subheader("4. Suspicious negative values")

            negative_issues = []

            numeric_columns = data.select_dtypes(
                include=["number"]
            ).columns

            for column in numeric_columns:
                if any(
                    keyword in str(column).lower()
                    for keyword in [
                        "amount",
                        "price",
                        "quantity",
                        "sales",
                        "revenue",
                        "cost",
                        "profit",
                        "age"
                    ]
                ):
                    negative_count = int(
                        (data[column] < 0).sum()
                    )

                    if negative_count > 0:
                        negative_issues.append({
                            "Column": column,
                            "Negative values": negative_count
                        })

            if negative_issues:
                st.warning(
                    "Negative values were found in numeric columns "
                    "that may require investigation."
                )

                st.dataframe(
                    pd.DataFrame(negative_issues),
                    use_container_width=True
                )

                st.caption(
                    "Negative values are not automatically errors. "
                    "For example, refunds and returns can legitimately "
                    "produce negative sales amounts."
                )
            else:
                st.success(
                    "No negative values detected in the checked columns."
                )

            # 5. Invalid-looking email addresses
            st.subheader("5. Invalid-looking email addresses")

            email_columns = [
                column for column in data.columns
                if "email" in str(column).lower()
            ]

            if not email_columns:
                st.info(
                    "No columns with 'email' in the name were found."
                )
            else:
                email_issues = []

                for column in email_columns:
                    non_missing = data[column].notna()

                    email_values = (
                        data.loc[non_missing, column]
                        .astype(str)
                        .str.strip()
                    )

                    invalid_mask = ~email_values.str.match(
                        r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
                        na=False
                    )

                    invalid_count = int(invalid_mask.sum())

                    if invalid_count > 0:
                        email_issues.append({
                            "Column": column,
                            "Invalid-looking emails": invalid_count
                        })

                if email_issues:
                    st.warning(
                        "Some email values do not match a basic "
                        "email-address pattern."
                    )

                    st.dataframe(
                        pd.DataFrame(email_issues),
                        use_container_width=True
                    )
                else:
                    st.success(
                        "No invalid-looking email addresses detected."
                    )

            # 6. Cleaning preview
            st.header("Cleaning Preview")

            st.write(
                "Preview a conservative cleanup that trims leading "
                "and trailing spaces in text columns and removes "
                "exact duplicate rows. Your original uploaded data "
                "is not modified."
            )

            cleaned_data = data.copy()

            for column in cleaned_data.select_dtypes(
                include=["object", "string"]
            ).columns:
                cleaned_data[column] = cleaned_data[column].apply(
                    lambda value: (
                        value.strip()
                        if isinstance(value, str)
                        else value
                    )
                )

            rows_before = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            duplicates_removed = rows_before - len(cleaned_data)

            col1, col2 = st.columns(2)

            col1.metric("Original rows", len(data))
            col2.metric("Rows after preview", len(cleaned_data))

            st.write(
                f"Exact duplicate rows that would be removed: "
                f"{duplicates_removed}"
            )

            st.dataframe(
                cleaned_data.head(10),
                use_container_width=True
            )

            # Download cleaned preview
            csv_data = cleaned_data.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download cleaned CSV preview",
                data=csv_data,
                file_name="cleaned_data_preview.csv",
                mime="text/csv"
            )

    except Exception as error:
        st.error(
            "The dataset could not be processed. "
            "Check that the file is valid and try again."
        )

        st.exception(error)
