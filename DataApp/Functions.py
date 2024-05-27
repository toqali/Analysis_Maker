import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_file(uploaded_file):
    """
    Reads the uploaded file and returns a DataFrame.

    Parameters:
    uploaded_file (UploadedFile): The file uploaded by the user.

    Returns:
    DataFrame: The contents of the uploaded file as a DataFrame.
    """
    # Check the file type
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        st.write("CSV file uploaded successfully.")
    elif uploaded_file.name.endswith('.json'):
        df = pd.read_json(uploaded_file)
        st.write("JSON file uploaded successfully.")
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.write("Excel file uploaded successfully.")
    else:
        st.write("Unsupported file format.")
    return df      

def create_card(title, value, index, cols):
    """
    Creates a card in the Streamlit UI with a title and value.

    Parameters:
    title (str): The title of the card.
    value (str): The value to display on the card.
    index (int): The index of the column in the Streamlit layout.
    cols (list): List of columns in the Streamlit layout.
    """
    with cols[index]:
        st.info(title)
        st.info(value)

def imputation_null(missing_cols, df):
    """
    Imputes missing values in the DataFrame.

    Parameters:
    missing_cols (list): List of columns with missing values.
    df (DataFrame): The DataFrame with missing values.

    Returns:
    None
    """
    for col in missing_cols:
        if df[col].dtype in ["int64", "float64"]:
            # For numerical columns, fill with median
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype in ["object", "bool"]:
            # For categorical columns, fill with mode
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)

def drop_null(missing_cols, df):
    """
    Drops columns or rows with null values.

    Parameters:
    missing_cols (list): List of columns with missing values.
    df (DataFrame): The DataFrame with missing values.

    Returns:
    None
    """
    for col in missing_cols:
        if df[col].isna().mean() >= 0.5:
            # Drop columns with more than 50% missing values
            df.drop(columns=col, inplace=True)
        else:
            # Drop rows with missing values in these columns
            df.dropna(subset=[col], inplace=True)

def show_null(subheader, df):
    """
    Displays the number of missing values in each column.

    Parameters:
    subheader (str): The subheader to display in the Streamlit UI.
    df (DataFrame): The DataFrame with missing values.

    Returns:
    None
    """
    st.subheader(subheader)
    missing_values_after = df.isna().sum()
    st.write(missing_values_after)

def add_spaces(pixels):
    """
    Adds vertical spaces in the Streamlit UI.

    Parameters:
    pixels (int): The number of pixels to add as space.

    Returns:
    None
    """
    st.write(f"<div style='margin-bottom: {pixels}px;'></div>", unsafe_allow_html=True)

def tab_header(header):
    """
    Adds a header with vertical spaces in the Streamlit UI.

    Parameters:
    header (str): The header text to display.

    Returns:
    None
    """
    add_spaces(60)
    st.header(header)
    add_spaces(60)

def percentMissing(missing_cols, df):
    """
    Calculates the percentage of missing values in each column.

    Parameters:
    missing_cols (list): List of columns with missing values.
    df (DataFrame): The DataFrame with missing values.

    Returns:
    list: A list of strings indicating the percentage of missing values for each column.
    """
    cols_percent = [f"{col} : {round(df[col].isna().mean(), 4)}" for col in missing_cols]
    return cols_percent

def duplicated_rows_detect(df):
    """
    Detects and displays the number of duplicated rows in the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to check for duplicated rows.

    Returns:
    None
    """
    duplicated_rows = df.duplicated(keep="first").sum()
    st.info((duplicated_rows))

def statisticalEDA(df, categorical_cols, numeric_cols):
    """
    Performs and displays basic statistical analysis of the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to analyze.
    categorical_cols (list): List of categorical columns.
    numeric_cols (list): List of numerical columns.

    Returns:
    None
    """
    # Some statistics of numerical cols
    st.subheader("Some Statistics 📈📉")
    statistics = df.describe()
    st.write(statistics)

    # Some Categorical Info
    if len(categorical_cols):
        st.subheader("Some Categorical Info 📦📝")
        cat_info = df.describe(include="object")
        st.write(cat_info)

    # Calculate the correlation matrix
    corr = df[numeric_cols].corr()
    # Display the correlation matrix
    st.subheader("Correlation Matrix: 👇🤗")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, ax=ax, cmap="coolwarm")
    st.pyplot(fig)
