import pandas as pd


def load_dataset(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    elif file_name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    else:
        raise ValueError(
            "Unsupported file type. Please upload CSV or XLSX."
        )