import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(
    page_title="AI Data Quality Investigator",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 AI Data Quality Investigator")
st.write(
    "Investigate missing values, duplicates, formatting problems, "
    "invalid dates, inconsistent categories, and unusual numeric values."
)

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"]
)


def find_date_columns(df):
    """Suggest columns whose names indicate dates."""
    keywords = ["date", "time", "timestamp", "dob", "birth"]
    return [
        col for col in df.columns
        if any(word in str(col).lower() for word in keywords)
    ]


def find_id_columns(df):
    """Suggest likely business identifier columns."""
    keywords = ["id", "transaction", "customer", "order", "invoice"]
    return [
        col for col in df.columns
        if any(word in str(col).lower() for word in keywords)
    ]


def find_numeric_columns(df):
    """Return numeric columns."""
    return list(df.select_dtypes(include="number").columns)


if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.success(f"Loaded: {uploaded_file.name}")

        st.subheader("Dataset preview")
        st.dataframe(data.head(10), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", data.shape[0])
        col2.metric("Columns", data.shape[1])
        col3.metric("Exact duplicate rows", int(data.duplicated().sum()))

        st.subheader("Choose columns for investigation")

        id_candidates = find_id_columns(data)
        date_candidates = find_date_columns(data)
        numeric_candidates = find_numeric_columns(data)

        id_columns = st.multiselect(
            "Business ID columns",
            options=list(data.columns),
            default=id_candidates,
            help="For example, Transaction_ID or Customer_ID. "
                 "Repeated IDs are flagged, not automatically deleted."
        )

        date_columns = st.multiselect(
            "Date columns",
            options=list(data.columns),
            default=date_candidates,
            help="Select columns that should contain dates."
        )

        numeric_text_candidates = [
            col for col in data.columns
            if any(
                word in str(col).lower()
                for word in [
                    "amount", "price", "sales", "revenue",
                    "quantity", "cost", "profit", "total"
                ]
            )
        ]

        numeric_text_columns = st.multiselect(
            "Columns that should be numeric",
            options=list(data.columns),
            default=numeric_text_candidates,
            help="Useful when numbers have accidentally been stored as text."
        )

        category_candidates = [
            col for col in data.columns
            if (
                data[col].dtype == "object"
                or isinstance(data[col].dtype, pd.StringDtype)
            )
            and not any(
                word in str(col).lower()
                for word in ["email", "description", "comment", "address"]
            )
        ]

        category_columns = st.multiselect(
            "Category columns",
            options=list(data.columns),
            default=category_candidates,
            help="Examples: country, status, product category, or region."
        )

        numeric_columns = st.multiselect(
            "Numeric columns for outlier checks",
            options=numeric_candidates,
            default=numeric_candidates,
            help="Outliers are unusual values, not necessarily errors."
        )

        if st.button("Investigate dataset", type="primary"):

            st.header("Investigation results")

            issue_count = 0

            # 1. Missing values
            st.subheader("1. Missing values")

            missing = data.isna().sum()
            missing = missing[missing > 0]

            if missing.empty:
                st.success("No missing values detected.")
            else:
                issue_count += int(missing.sum())

                missing_report = (
                    missing.rename("Missing count")
                    .rename_axis("Column")
                    .reset_index()
                )
                missing_report["Percentage"] = (
                    missing_report["Missing count"] / len(data) * 100
                ).round(2)

                st.dataframe(missing_report, use_container_width=True)

            # 2. Exact duplicate rows
            st.subheader("2. Exact duplicate rows")

            exact_duplicates = int(data.duplicated().sum())
            issue_count += exact_duplicates

            st.metric("Exact duplicate rows", exact_duplicates)

            if exact_duplicates:
                with st.expander("Preview duplicate rows"):
                    st.dataframe(
                        data[data.duplicated(keep=False)],
                        use_container_width=True
                    )

            # 3. Repeated business IDs
            st.subheader("3. Duplicate business IDs")

            id_findings = []

            for column in id_columns:
                non_missing = data[column].notna()
                repeated_mask = (
                    non_missing
                    & data[column].duplicated(keep=False)
                )

                repeated_count = int(
                    data.loc[repeated_mask, column].nunique()
                )

                if repeated_count:
                    id_findings.append({
                        "Column": column,
                        "Repeated ID values": repeated_count,
                        "Affected rows": int(repeated_mask.sum())
                    })

            if id_findings:
                issue_count += sum(
                    item["Affected rows"] for item in id_findings
                )
                st.warning(
                    "Repeated identifiers found. Some IDs can legitimately "
                    "repeat, so review them using the business rules."
                )
                st.dataframe(
                    pd.DataFrame(id_findings),
                    use_container_width=True
                )

                with st.expander("Preview rows with repeated IDs"):
                    for column in id_columns:
                        non_missing = data[column].notna()
                        repeated_mask = (
                            non_missing
                            & data[column].duplicated(keep=False)
                        )
                        if repeated_mask.any():
                            st.write(f"Repeated values in: {column}")
                            st.dataframe(
                                data.loc[repeated_mask],
                                use_container_width=True
                            )
            else:
                st.success("No repeated non-missing IDs found in selected columns.")

            # 4. Extra spaces
            st.subheader("4. Leading or trailing spaces")

            whitespace_findings = []

            for column in data.select_dtypes(
                include=["object", "string"]
            ).columns:
                mask = data[column].apply(
                    lambda value: isinstance(value, str)
                    and value != value.strip()
                )
                count = int(mask.sum())

                if count:
                    whitespace_findings.append({
                        "Column": column,
                        "Affected values": count
                    })

            if whitespace_findings:
                issue_count += sum(
                    item["Affected values"]
                    for item in whitespace_findings
                )
                st.dataframe(
                    pd.DataFrame(whitespace_findings),
                    use_container_width=True
                )
            else:
                st.success("No leading or trailing spaces detected.")

            # 5. Invalid-looking emails
            st.subheader("5. Invalid-looking email addresses")

            email_columns = [
                col for col in data.columns
                if "email" in str(col).lower()
            ]

            email_findings = []

            for column in email_columns:
                mask = data[column].notna()
                values = data.loc[mask, column].astype(str).str.strip()

                invalid = ~values.str.match(
                    r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
                    na=False
                )

                if invalid.any():
                    email_findings.append({
                        "Column": column,
                        "Invalid-looking emails": int(invalid.sum())
                    })

            if email_findings:
                issue_count += sum(
                    item["Invalid-looking emails"]
                    for item in email_findings
                )
                st.dataframe(
                    pd.DataFrame(email_findings),
                    use_container_width=True
                )
            else:
                st.success("No invalid-looking emails detected.")

            # 6. Suspicious negative numbers
            st.subheader("6. Suspicious negative values")

            negative_findings = []

            for column in data.select_dtypes(include="number").columns:
                if any(
                    word in str(column).lower()
                    for word in [
                        "amount", "price", "quantity", "sales",
                        "revenue", "cost", "profit", "age", "total"
                    ]
                ):
                    count = int((data[column] < 0).sum())

                    if count:
                        negative_findings.append({
                            "Column": column,
                            "Negative values": count
                        })

            if negative_findings:
                issue_count += sum(
                    item["Negative values"]
                    for item in negative_findings
                )
                st.dataframe(
                    pd.DataFrame(negative_findings),
                    use_container_width=True
                )
                st.caption(
                    "Negative values may be legitimate, such as refunds "
                    "or returns. Review before changing them."
                )
            else:
                st.success("No suspicious negative values detected.")

            # 7. Invalid dates
            st.subheader("7. Invalid dates")

            date_findings = []

            for column in date_columns:
                non_missing = data[column].notna()
                parsed = pd.to_datetime(
                    data.loc[non_missing, column],
                    errors="coerce"
                )
                invalid_count = int(parsed.isna().sum())

                if invalid_count:
                    date_findings.append({
                        "Column": column,
                        "Unparseable dates": invalid_count
                    })

            if date_findings:
                issue_count += sum(
                    item["Unparseable dates"] for item in date_findings
                )
                st.dataframe(
                    pd.DataFrame(date_findings),
                    use_container_width=True
                )
            else:
                st.success(
                    "No unparseable dates found in selected date columns."
                )

            # 8. Numeric values stored as text
            st.subheader("8. Numeric values stored as text")

            type_findings = []

            for column in numeric_text_columns:
                if column in numeric_candidates:
                    continue

                non_missing = data[column].notna()

                if not non_missing.any():
                    continue

                text_values = data.loc[non_missing, column].astype(str).str.strip()
                normalized = (
                    text_values
                    .str.replace(",", "", regex=False)
                    .str.replace("$", "", regex=False)
                    .str.replace("€", "", regex=False)
                    .str.replace("£", "", regex=False)
                    .str.replace("%", "", regex=False)
                )

                converted = pd.to_numeric(normalized, errors="coerce")
                invalid_count = int(converted.isna().sum())

                if invalid_count:
                    type_findings.append({
                        "Column": column,
                        "Values not interpretable as numbers": invalid_count
                    })

            if type_findings:
                issue_count += sum(
                    item["Values not interpretable as numbers"]
                    for item in type_findings
                )
                st.dataframe(
                    pd.DataFrame(type_findings),
                    use_container_width=True
                )
            else:
                st.success(
                    "No non-numeric values detected in the selected columns."
                )

            # 9. Inconsistent categories
            st.subheader("9. Inconsistent category formatting")

            category_findings = []

            for column in category_columns:
                values = data[column].dropna().astype(str)
                normalized = values.str.strip().str.casefold()

                # Find normalized groups that have multiple original forms.
                groups = pd.DataFrame({
                    "Original": values,
                    "Normalized": normalized
                })

                inconsistent_groups = groups.groupby(
                    "Normalized"
                )["Original"].nunique()

                inconsistent_groups = inconsistent_groups[
                    inconsistent_groups > 1
                ]

                if not inconsistent_groups.empty:
                    category_findings.append({
                        "Column": column,
                        "Groups with formatting variations": int(
                            len(inconsistent_groups)
                        )
                    })

            if category_findings:
                issue_count += sum(
                    item["Groups with formatting variations"]
                    for item in category_findings
                )
                st.dataframe(
                    pd.DataFrame(category_findings),
                    use_container_width=True
                )
                st.caption(
                    "This check flags capitalization or spacing differences "
                    "after trimming and case normalization. Similar-looking "
                    "categories are not automatically merged."
                )
            else:
                st.success(
                    "No capitalization or whitespace category variations found."
                )

            # 10. Outliers using the IQR method
            st.subheader("10. Potential numeric outliers")

            outlier_findings = []

            for column in numeric_columns:
                series = data[column].dropna()

                if len(series) < 4 or series.nunique() < 2:
                    continue

                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1

                if iqr == 0:
                    continue

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                mask = (
                    data[column].notna()
                    & (
                        (data[column] < lower_bound)
                        | (data[column] > upper_bound)
                    )
                )

                count = int(mask.sum())

                if count:
                    outlier_findings.append({
                        "Column": column,
                        "Potential outliers": count,
                        "Lower threshold": round(float(lower_bound), 3),
                        "Upper threshold": round(float(upper_bound), 3)
                    })

            if outlier_findings:
                issue_count += sum(
                    item["Potential outliers"]
                    for item in outlier_findings
                )
                st.dataframe(
                    pd.DataFrame(outlier_findings),
                    use_container_width=True
                )
                st.caption(
                    "Outliers are statistical flags, not confirmed errors. "
                    "Review them using business context."
                )
            else:
                st.success(
                    "No potential outliers detected in the selected columns."
                )

            # Summary
            st.header("Investigation summary")
            st.metric("Total flagged findings", issue_count)
            st.caption(
                "This is a count of findings across checks, not a count "
                "of unique affected records. A record can be flagged by "
                "more than one check."
            )

            # Conservative cleaning preview
            st.header("Cleaning preview")

            st.write(
                "This preview trims leading and trailing spaces and removes "
                "exact duplicate rows. It does not alter the uploaded data, "
                "fill missing values, correct dates, or change IDs."
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

            before_duplicates = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            removed_duplicates = before_duplicates - len(cleaned_data)

            col1, col2 = st.columns(2)
            col1.metric("Original rows", len(data))
            col2.metric("Rows in preview", len(cleaned_data))

            st.write(
                f"Exact duplicate rows that would be removed: "
                f"{removed_duplicates}"
            )

            st.dataframe(
                cleaned_data.head(10),
                use_container_width=True
            )

            csv_data = cleaned_data.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download cleaned CSV preview",
                data=csv_data,
                file_name="cleaned_data_preview.csv",
                mime="text/csv"
            )

    except Exception as error:
        st.error(
            "The dataset could not be processed. "
            "Check the file and selected columns, then try again."
        )
        st.exception(error)
