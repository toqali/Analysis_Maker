import streamlit as st
import numpy as np
from Functions import *
from process import *
from visualization import *
from ML_models import modelsSection
from streamlit_option_menu import option_menu


# Welcome statement
st.header('🎉 Welcome to an Analysis Maker! 🚀')

# Add space
st.write("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)

# upload file
uploaded_file = st.file_uploader("", type=["csv", "json", "Excel File"])

# Add space
st.write("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

# Read the df
if uploaded_file is not None:  
        df = read_file(uploaded_file)
        categorical_cols = df.select_dtypes(include='object').columns
        numeric_cols = df.select_dtypes(include=np.number).columns    

# Tabs
selected_tab = option_menu(
    menu_title=None,
    options= ["🧹 Process", " 🎨 Visualization", " 💡 ML Model"],
   )

# Create Tabs
if uploaded_file is not None:
        modified_df = processSection(selected_tab, uploaded_file, df, categorical_cols, numeric_cols)
        visSection(selected_tab, uploaded_file, modified_df, categorical_cols, numeric_cols)
        modelsSection(selected_tab, uploaded_file, modified_df)                      


                

