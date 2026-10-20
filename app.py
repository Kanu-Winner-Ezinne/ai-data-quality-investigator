import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="AI Data Quality Investigator",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 AI Data Quality Investigator")
st.write(
    "Upload a CSV or Excel dataset to investigate data quality "
    "issues, review findings, and preview conservative cleaning."
)


def suggest_columns(columns, keywords):
    """Suggest columns based on their names."""
    return [
        col for col in columns
        if any(word in str(col).lower() for word in keywords)
    ]


def display_report(report):
    """Display a report only when it contains findings."""
    if report:
        st.dataframe(pd.DataFrame(report), use_container_width=True)
    else:
        st.success("No issues detected by this check.")


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

        # --------------------------------------------------
        # 1. Dataset overview and visible column names
        # --------------------------------------------------
        st.header("1. Dataset overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", data.shape[0])
        col2.metric("Columns", data.shape[1])
        col3.metric("Exact duplicate rows", int(data.duplicated().sum()))

        st.subheader("Column names detected")

        column_information = pd.DataFrame({
            "Column name": [str(col) for col in data.columns],
            "Data type": [str(data[col].dtype) for col in data.columns],
            "Missing values": [
                int(data[col].isna().sum()) for col in data.columns
            ],
            "Unique values": [
                int(data[col].nunique(dropna=True)) for col in data.columns
            ]
        })

        st.dataframe(column_information, use_container_width=True)

        with st.expander("View all column names as a list"):
            for column in data.columns:
                st.write(f"- {column}")

        st.subheader("Dataset preview")
        st.dataframe(data.head(10), use_container_width=True)

        # --------------------------------------------------
        # 2. Investigation settings
        # --------------------------------------------------
        st.header("2. Choose columns for investigation")
        st.write(
            "Review these selections. The app suggests columns based "
            "on their names, but you can change the selections."
        )

        columns = list(data.columns)

        id_suggestions = suggest_columns(
            columns,
            ["id", "transaction", "customer", "order", "invoice"]
        )

        date_suggestions = suggest_columns(
            columns,
            ["date", "time", "timestamp", "dob", "birth"]
        )

        numeric_name_suggestions = suggest_columns(
            columns,
            [
                "amount", "price", "sales", "revenue", "quantity",
                "cost", "profit", "total", "age"
            ]
        )

        category_suggestions = [
            col for col in columns
            if (
                pd.api.types.is_object_dtype(data[col])
                or pd.api.types.is_string_dtype(data[col])
                or pd.api.types.is_categorical_dtype(data[col])
            )
            and not any(
                word in str(col).lower()
                for word in ["description", "comment", "address"]
            )
        ]

        numeric_suggestions = list(
            data.select_dtypes(include="number").columns
        )

        id_columns = st.multiselect(
            "Business ID columns",
            options=columns,
            default=id_suggestions,
            help="Examples: Transaction_ID, Customer_ID, Order_ID."
        )

        date_columns = st.multiselect(
            "Columns that should contain dates",
            options=columns,
            default=date_suggestions
        )

        numeric_text_columns = st.multiselect(
            "Columns that should contain numbers",
            options=columns,
            default=numeric_name_suggestions
        )

        category_columns = st.multiselect(
            "Category columns to check",
            options=columns,
            default=category_suggestions
        )

        outlier_columns = st.multiselect(
            "Numeric columns to check for outliers",
            options=numeric_suggestions,
            default=numeric_suggestions
        )

        # --------------------------------------------------
        # 3. Run checks
        # --------------------------------------------------
        if st.button("Investigate dataset", type="primary"):

            st.header("3. Investigation results")
            total_findings = 0

            # Missing values
            st.subheader("Missing values")

            missing = data.isna().sum()
            missing = missing[missing > 0]

            if missing.empty:
                st.success("No missing values detected.")
            else:
                total_findings += int(missing.sum())

                missing_report = (
                    missing.rename("Missing count")
                    .rename_axis("Column")
                    .reset_index()
                )

                missing_report["Percentage"] = (
                    missing_report["Missing count"] / len(data) * 100
                ).round(2)

                st.dataframe(
                    missing_report,
                    use_container_width=True
                )

            # Exact duplicate rows
            st.subheader("Exact duplicate rows")

            duplicate_mask = data.duplicated(keep=False)
            duplicate_count = int(data.duplicated().sum())

            st.metric("Duplicate rows beyond the first occurrence",
                      duplicate_count)
            total_findings += duplicate_count

            if duplicate_count:
                with st.expander("View exact duplicate records"):
                    st.dataframe(
                        data.loc[duplicate_mask],
                        use_container_width=True
                    )

            # Duplicate business IDs
            st.subheader("Duplicate business IDs")
            id_report = []

            for column in id_columns:
                values = data[column]
                non_missing = values.notna()
                repeated = non_missing & values.duplicated(keep=False)

                repeated_values = values[repeated].nunique()

                if repeated_values:
                    id_report.append({
                        "Column": str(column),
                        "Distinct repeated IDs": int(repeated_values),
                        "Rows with repeated IDs": int(repeated.sum())
                    })

            display_report(id_report)

            if id_report:
                total_findings += sum(
                    row["Rows with repeated IDs"] for row in id_report
                )

            # Whitespace
            st.subheader("Leading or trailing spaces")
            whitespace_report = []

            for column in data.select_dtypes(
                include=["object", "string"]
            ).columns:
                mask = data[column].apply(
                    lambda value: isinstance(value, str)
                    and value != value.strip()
                )

                if mask.any():
                    whitespace_report.append({
                        "Column": str(column),
                        "Affected values": int(mask.sum())
                    })

            display_report(whitespace_report)

            if whitespace_report:
                total_findings += sum(
                    row["Affected values"] for row in whitespace_report
                )

            # Emails
            st.subheader("Invalid-looking email addresses")
            email_columns = [
                col for col in columns if "email" in str(col).lower()
            ]
            email_report = []

            for column in email_columns:
                values = data[column].dropna().astype(str).str.strip()
                invalid = ~values.str.match(
                    r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
                    na=False
                )

                if invalid.any():
                    email_report.append({
                        "Column": str(column),
                        "Invalid-looking values": int(invalid.sum())
                    })

            display_report(email_report)

            if email_report:
                total_findings += sum(
                    row["Invalid-looking values"] for row in email_report
                )

            # Negative numbers
            st.subheader("Suspicious negative numeric values")
            negative_report = []

            for column in data.select_dtypes(include="number").columns:
                if any(
                    word in str(column).lower()
                    for word in [
                        "amount", "price", "quantity", "sales",
                        "revenue", "cost", "profit", "age", "total"
                    ]
                ):
                    negative_count = int((data[column] < 0).sum())

                    if negative_count:
                        negative_report.append({
                            "Column": str(column),
                            "Negative values": negative_count
                        })

            display_report(negative_report)

            if negative_report:
                total_findings += sum(
                    row["Negative values"] for row in negative_report
                )

            st.caption(
                "Negative values may be legitimate, such as refunds "
                "or returned goods. Review before changing them."
            )

            # Invalid dates
            st.subheader("Invalid dates")
            date_report = []

            for column in date_columns:
                values = data[column].dropna()

                parsed = pd.to_datetime(
                    values,
                    errors="coerce"
                )

                invalid_count = int(parsed.isna().sum())

                if invalid_count:
                    date_report.append({
                        "Column": str(column),
                        "Unparseable non-missing values": invalid_count
                    })

            display_report(date_report)

            if date_report:
                total_findings += sum(
                    row["Unparseable non-missing values"]
                    for row in date_report
                )

            # Numeric values stored as text
            st.subheader("Numeric values stored as text")
            type_report = []

            for column in numeric_text_columns:
                values = data[column].dropna()

                if values.empty:
                    continue

                normalized = (
                    values.astype(str)
                    .str.strip()
                    .str.replace(",", "", regex=False)
                    .str.replace("$", "", regex=False)
                    .str.replace("€", "", regex=False)
                    .str.replace("£", "", regex=False)
                    .str.replace("%", "", regex=False)
                )

                converted = pd.to_numeric(
                    normalized,
                    errors="coerce"
                )

                invalid_count = int(converted.isna().sum())

                if invalid_count:
                    type_report.append({
                        "Column": str(column),
                        "Values not interpretable as numbers": invalid_count
                    })

            display_report(type_report)

            if type_report:
                total_findings += sum(
                    row["Values not interpretable as numbers"]
                    for row in type_report
                )

            # Inconsistent category formatting
            st.subheader("Inconsistent category formatting")
            category_report = []

            for column in category_columns:
                values = data[column].dropna().astype(str)

                if values.empty:
                    continue

                grouped = pd.DataFrame({
                    "Original value": values,
                    "Normalized value": values.str.strip().str.casefold()
                })

                variations = grouped.groupby(
                    "Normalized value"
                )["Original value"].nunique()

                variations = variations[variations > 1]

                if not variations.empty:
                    category_report.append({
                        "Column": str(column),
                        "Groups with spacing/case variations": int(
                            len(variations)
                        )
                    })

            display_report(category_report)

            if category_report:
                total_findings += sum(
                    row["Groups with spacing/case variations"]
                    for row in category_report
                )

            st.caption(
                "This check identifies spacing and capitalization "
                "differences. It does not assume that different labels "
                "such as 'US' and 'United States' mean the same thing."
            )

            # Outliers (IQR)
            st.subheader("Potential numeric outliers")
            outlier_report = []

            for column in outlier_columns:
                values = data[column].dropna()

                if len(values) < 4 or values.nunique() < 2:
                    continue

                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1

                if iqr == 0:
                    continue

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                mask = (
                    data[column].notna()
                    & (
                        (data[column] < lower)
                        | (data[column] > upper)
                    )
                )

                count = int(mask.sum())

                if count:
                    outlier_report.append({
                        "Column": str(column),
                        "Potential outliers": count,
                        "Lower threshold": round(float(lower), 3),
                        "Upper threshold": round(float(upper), 3)
                    })

            display_report(outlier_report)

            if outlier_report:
                total_findings += sum(
                    row["Potential outliers"] for row in outlier_report
                )

            st.caption(
                "Outliers are unusual statistical values, not confirmed "
                "errors. Business context matters."
            )

            # Summary
            st.header("4. Investigation summary")
            st.metric("Total flagged findings", total_findings)

            st.info(
                "This is a total across checks, not a count of unique "
                "affected rows. One record may trigger multiple checks."
            )

            # Conservative cleaning preview
            st.header("5. Cleaning preview")
            st.write(
                "The preview trims leading/trailing spaces and removes "
                "exact duplicate rows only. It does not overwrite the "
                "original uploaded data."
            )

            cleaned_data = data.copy()

            for column in cleaned_data.select_dtypes(
                include=["object", "string"]
            ).columns:
                cleaned_data[column] = cleaned_data[column].apply(
                    lambda value: value.strip()
                    if isinstance(value, str)
                    else value
                )

            rows_before = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            duplicates_removed = rows_before - len(cleaned_data)

            col1, col2 = st.columns(2)
            col1.metric("Original rows", len(data))
            col2.metric("Rows in cleaning preview", len(cleaned_data))

            st.write(
                f"Exact duplicate rows removed in preview: "
                f"{duplicates_removed}"
            )

            st.dataframe(
                cleaned_data.head(10),
                use_container_width=True
            )

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
