from Functions import *
from process import *
# Visualization Tab        
def visSection(selected_tab, uploaded_file, df, categorical_cols, numeric_cols):
  if  selected_tab == " 🎨 Visualization":
    
    tab_header("✨👀🔎 EDA ")
    
    if uploaded_file is not None: 
    # Slider, 3- Visualization options
        st.sidebar.header("Visualizations 🎨")
        plot_options = ["Bar plot", "Scatter plot", "Histogram", "Box plot", "Pairplot"]
        selected_plot = st.sidebar.selectbox("Choose a plot type", plot_options)

        # 1st plot : Barplot
        if selected_plot == "Bar plot" and len(categorical_cols) !=0:
            x_axis = st.sidebar.selectbox("Select x-axis", categorical_cols)
            indices = df[x_axis].value_counts().sort_values(ascending=False).index
            values = df[x_axis].value_counts().sort_values(ascending=False).values
            bars = st.sidebar.slider("Number of Categories : ", 1, len(indices) )
            top_categories = indices[:bars]
            filtered_data = df[df[x_axis].isin(top_categories)]
            # Check if the button is clicked
            if st.sidebar.button("Generate Plot 📊"):
                st.write("Bar plot:")
                fig, ax = plt.subplots()
                sns.countplot(data=filtered_data, x=x_axis, order=top_categories, ax=ax)
                st.pyplot(fig)

            

        # 2nd plot : Scatter plot
        elif selected_plot == "Scatter plot" and len(numeric_cols) >1:
            x_axis = st.sidebar.selectbox("Select x-axis", numeric_cols)
            y_axis = st.sidebar.selectbox("Select y-axis", numeric_cols)
            hue = st.sidebar.selectbox("Select Hue", categorical_cols)
            if st.sidebar.button("Generate Plot 📊"):
                st.write("Scatter plot:")
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax, hue = df[hue])
                st.pyplot(fig)
        # 3rd plot : Histogram
        elif selected_plot == "Histogram" and len(numeric_cols) !=0:
            column = st.sidebar.selectbox("Select a column", numeric_cols)
            bins = st.sidebar.slider("Number of bins", 5, 100, 20)
            if st.sidebar.button("Generate Plot 📊"):
                st.write("Histogram:")
                fig, ax = plt.subplots()
                sns.histplot(df[column], bins=bins, ax=ax)
                st.pyplot(fig)

        # 4th plot: Box plot
        elif selected_plot == "Box plot" and len(numeric_cols) !=0:
            column = st.sidebar.selectbox("Select a column", numeric_cols)
            if st.sidebar.button("Generate Plot 📊"):
                st.write("Box plot:")
                fig, ax = plt.subplots()
                sns.boxplot(df[column], ax=ax)
                st.pyplot(fig) 
                
        # 5th plot : Pairplot          
        elif selected_plot == "Pairplot" and len(numeric_cols) >1:
            x_axis = st.sidebar.multiselect("Select x-axis columns", numeric_cols)
            hue = st.sidebar.selectbox("Choose hue : ", list(categorical_cols))
            if st.sidebar.button("Generate Plot 📊"):
                st.write("Pairplot:")
                fig = sns.pairplot(df,  vars=x_axis, hue= hue)
                st.pyplot(fig)  