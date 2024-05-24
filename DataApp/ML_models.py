import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from Functions import *
from pycaret.classification import setup as classification_setup, compare_models as compare_classification_models, plot_model as plot_classification_model, pull as pull_classification
from pycaret.regression import setup as regression_setup, compare_models as compare_regression_models, plot_model as plot_regression_model, pull as pull_regression
import joblib
import base64

def modelsSection(selected_tab, uploaded_file, df):
    if selected_tab == " 💡 ML Model":
        tab_header("✨💡 Modeling ")

        if uploaded_file is not None:
            # Initialize session state
            if 'df' not in st.session_state:
                st.session_state.df = None
            
            target = st.selectbox("### **Choose the Target**", df.columns)

            if st.button("Run Modeling"):
                summary_info(target, df)

            # Display the transformed dataset if it exists in session state
            if "setup_df" in st.session_state:
                st.subheader("Transformed Dataset after Setup:")
                st.dataframe(st.session_state.setup_df)

            trainModels()

def summary_info(target, df):
    '''
    Parametrs :
      - target : label 
      - df : data you want to train algorithm on
    Return :
     Some General info as type task, process method, ...  
    '''
    # this is trivial or general way you can use other ways to differentiate b/w continous & dicreate values
    if df[target].nunique() > 10:
        # Setup PyCaret for regression problem
        des = regression_setup(data=df, target=target, verbose=True)
        st.session_state.task = 'regression'
        setup_df = pull_regression()
        
    else:
        # Initialize LabelEncoder for classification problem
        label_encoder = LabelEncoder()
        df[target] = label_encoder.fit_transform(df[target])

        # Setup PyCaret for classification problem
        des = classification_setup(data=df, target=target, verbose=True)
        st.session_state.task = 'classification'
        setup_df = pull_classification()

    # Pull the transformed dataset after setup
    st.session_state.setup_df = setup_df

def trainModels():
    '''
    The aim od it is to train choosen algorithms
    parameters : None
    return :
     - comparison table of models
    '''
    if 'task' in st.session_state:
        if st.session_state.task == 'classification':
            models_list = ["gbc", "lr", "xgboost", "lightgbm", "rf", "et", "lda", "ada", "dummy", "dt", "knn", "svm", "nb"]
            compare_models_func = compare_classification_models
            plot_model_func = plot_classification_model
            pull_func = pull_classification
        elif st.session_state.task == 'regression':
            models_list = ["lr", "ridge", "rf", "et", "lightgbm", "dummy", "dt", "knn", "svm"]
            compare_models_func = compare_regression_models
            plot_model_func = plot_regression_model
            pull_func = pull_regression
        else:
            st.error("Invalid task type")
            return

        models_to_compare = st.multiselect("### **Choose Training Models 🤖✨**", models_list)

        if st.button("Train the Model"):
            add_spaces(50)
            st.subheader("Different Types of Models with their performance 🔥😊")
            
            # it return table with selected models with their performance measures
            # Compare specific models
            best_model = compare_models_func(include=models_to_compare)
            # Pull the comparison table
            comparison_table = pull_func()
            # Convert the comparison table to a DataFrame
            comparison_df = pd.DataFrame(comparison_table)
            # Display the comparison table in Streamlit
            st.table(comparison_df)

            # Generate the feature importance plot and save it
            plot_model_func(best_model, plot="feature", save=True)
            st.image("Feature Importance.png")

            if st.session_state.task == 'classification':
                confusion_matrix(plot_model_func, best_model)

            # Save the best model to session state for predictions
            st.session_state.best_model = best_model

            # Save the trained model
            save_model(best_model)

def save_model(model):
    """
    Save the trained model to a file.
    
    Parameters:
        model: The trained model object.
    """
    model_file_path = "trained_model.joblib"
    joblib.dump(model, model_file_path)
    st.info("Save the model : 👇✨🔥")
    # Provide download link for the saved model
    st.markdown(get_model_download_link(model_file_path), unsafe_allow_html=True)

def get_model_download_link(model_file_path):
    """
    Generates a download link for the saved model file.
    """
    with open(model_file_path, 'rb') as f:
        model_bytes = f.read()
    b64 = base64.b64encode(model_bytes).decode()
    href = f'<a href="data:file/model.joblib;base64,{b64}" download="trained_model.joblib">Download Trained Model</a>'
    return href

def confusion_matrix(plot_model_func, best_model):
    '''
    parametrs:
     - plot_model_fun : function used to plot
     - best_model : choose best model (highest accuracy as default)
    return :
     - confusion matrix 
    '''
    plot_model_func(best_model, plot="confusion_matrix", save=True)
    st.image("Confusion Matrix.png")

