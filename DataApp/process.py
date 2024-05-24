import streamlit as st
from Functions import *
from ML_models import *
from visualization import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler

def processSection(selected_tab, uploaded_file, df, categorical_cols, numeric_cols):
    if selected_tab == "🧹 Process":
        
        tab_header("🧩☕ Process")
        if uploaded_file is not None:
            
            # Initialize session state variables to preserve the modifications
            if 'df' not in st.session_state:
                st.session_state.df = df.copy()
                st.session_state.missing_values_info = False
                st.session_state.duplicated_rows_info = False
                st.session_state.encoding_applied = False
            # To preserve the last value of modified df even the page reloaded    
            df = st.session_state.df

            # Create two columns for 4 cards
            cols = st.columns(4)
            create_card("No. Rows", df.shape[0], 0, cols)
            create_card("No. of Cols", df.shape[1], 1, cols)
            create_card("Numerical Cols", len(numeric_cols), 2, cols)
            create_card("Categorical Cols", len(categorical_cols), 3, cols)
            
            # sample of data
            st.subheader("Show sample of df")
            st.write(df.head())
            #  No. of Duplicated rows before processing
            st.subheader("Number of Duplicated Rows Before 🔢")
            duplicated_rows_detect(df)
            # No. of missing values before processing
            missing_cols = df.columns[df.isnull().any()].tolist()
            show_null("Show missing values before 🔍", df)
            
            
            # Sidebar, 1- Handling Misiing value
            st.sidebar.markdown('<h2 style="color: #4682b4;">Missing Values Handler 🔎👀🤗</h2>', unsafe_allow_html=True)  # Use a suitable color for the header
            missing_cols_copy = list(missing_cols)
            # 1st option : imputation by mean or mode
            impute_cols = st.sidebar.multiselect('Choose cols for Imputation', percentMissing(missing_cols_copy, df))
            impute_cols = [col.split(':')[0].strip() for col in impute_cols]
            missing_cols_copy = [col for col in missing_cols_copy if col not in impute_cols]
            # 2nd option : dropping rows with null values
            drop_cols = st.sidebar.multiselect('Choose cols for Dropping', percentMissing(missing_cols_copy, df))
            drop_cols = [col.split(':')[0].strip() for col in drop_cols]
            
            # drop duplicated rows
            drop_dupRows = st.sidebar.checkbox("Delete Duplicated_Rows 🤝😊")
            
            has_missing = df.isna().sum().any()
            has_dup_rows = df.duplicated(keep="first").sum() != 0 
            
            # apply handling methods on data
            if st.sidebar.button("Apply 🤗🔎"):
                if impute_cols:
                    imputation_null(impute_cols, df)
                if drop_cols:
                    drop_null(drop_cols, df)
                if drop_dupRows:
                    df.drop_duplicates(inplace=True)
                
                st.session_state.df = df  # save modified df

                st.session_state.missing_values_info = has_missing
                st.session_state.duplicated_rows_info = has_dup_rows
            # still df has missing or duplicated and you handle them >> the modified will stay displaying
            if st.session_state.missing_values_info:
                show_null("Show missing values after 🕵️‍♂️", df)
            if st.session_state.duplicated_rows_info:
                st.subheader("Number of Duplicated Rows After 🔢")
                duplicated_rows_detect(df)
            # some statistical measures & heatmap
            statisticalEDA(df, categorical_cols, numeric_cols)
            
            # Sidebar, 2- Scaling
            st.sidebar.markdown('<h2 style="color: #4682b4;">Scaling ✨📐⚖️</h2>', unsafe_allow_html=True)  # Use steel blue color for the header
            options_scaling = [("Standard Scaler", StandardScaler()), ("Min-Max Scaler", MinMaxScaler())]

            for option_name, scaler in options_scaling:
                selected_cols = st.sidebar.multiselect(f'Choose columns for {option_name}', numeric_cols, format_func=lambda x: x, key=f"select_{option_name}")  # Use orange color for the subheader
                if st.sidebar.button(f"Apply {option_name}", key=f"apply_{option_name}"):
                    for col in selected_cols:
                        df[col] = scaler.fit_transform(df[[col]])
                    st.session_state.df = df 
                    st.session_state.scaling_applied = True

            
            
            # Sidebar, 3- Encoding
            st.sidebar.markdown('<h2 style="color: #4682b4;">Encoding 🔤🔡✨</h2>', unsafe_allow_html=True)  # Use steel blue color for the header
            options = [("One hot encoding", OneHotEncoder()), ("Label encoding", LabelEncoder()), ("Ordinal encoding", OrdinalEncoder())]

            categorical_cols_copy = list(categorical_cols)

            for option, encoder in options:
                selected_cols = st.sidebar.multiselect(f'Choose columns for {option}', categorical_cols_copy, format_func=lambda x: x, key=f"select_{option}")  # Use orange color for the subheader
                if st.sidebar.button(f"Apply {option}", key=f"apply_{option}"):
                    for col in selected_cols:
                        column_2d = df[col].values.reshape(-1, 1)
                        encoded_column = encoder.fit_transform(column_2d)
                        df[col] = encoded_column.toarray() if isinstance(encoder, OneHotEncoder) else encoded_column
                        categorical_cols_copy.remove(col)
                    st.session_state.df = df  # preserve modified df
                    st.session_state.encoding_applied = True

            if st.session_state.encoding_applied:
                st.subheader('Show modified dataframe')
                st.write(df.head())

    return st.session_state.df # to be used with other section : visuals & ML
