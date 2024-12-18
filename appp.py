import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

st.set_page_config(page_title="Enhanced Air Quality Analysis", page_icon=":bar_chart:")

# Load the dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Allenstrange/CMP-Assessment/refs/heads/main/Air_Quality_Beijing.csv"
    return pd.read_csv(url)

# Load data into session state
if 'data' not in st.session_state:
    st.session_state['data'] = load_data()

# Page 1: Data Loading
def data_loading():
    st.title("Data Loading")
    st.write("""
    ### Overview
    This page allows you to load and preview the air quality dataset. You can view basic statistics 
    and inspect missing values in the dataset.
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # Data preview
        num_rows = st.slider("Select Number of Rows to Preview", 1, 100, 10)
        st.write("### Data Preview:")
        st.dataframe(data.head(num_rows))

        # Descriptive Statistics Options
        st.write("### Descriptive Statistics:")
        if st.checkbox("Show Descriptive Statistics Table"):
            st.write(data.describe().T)

        # Missing Value Statistics Options
        st.write("### Missing Values:")
        if st.checkbox("Show Missing Values Table"):
            missing_values = data.isnull().sum()
            missing_percentage = (data.isnull().sum() / len(data)) * 100
            missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
            st.table(missing_df)

        # Dataset Export
        st.write("### Export Dataset")
        if st.button("Download Dataset"):
            st.download_button("Download CSV", data.to_csv(index=False), "air_quality.csv", "text/csv")
    else:
        st.write("Data could not be loaded.")

# Page 2: Data Preprocessing
def data_preprocessing():
    st.title("Data Preprocessing")
    st.write("""
    ### Overview
    This page provides tools to clean and preprocess the dataset, including handling missing values, 
    dropping unnecessary columns, and feature engineering.
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # Section 1: Handling Missing Values
        st.header("Handling Missing Values")
        imputation_method = st.radio("Choose an imputation method:", ["Mean", "Median", "Mode"])
        columns_to_impute = st.multiselect("Select columns to impute:", data.columns)

        if st.button("Impute Missing Values"):
            for column in columns_to_impute:
                if imputation_method == "Mean":
                    data[column].fillna(data[column].mean(), inplace=True)
                elif imputation_method == "Median":
                    data[column].fillna(data[column].median(), inplace=True)
                elif imputation_method == "Mode":
                    data[column].fillna(data[column].mode()[0], inplace=True)
            st.success("Missing values imputed successfully.")

        # Section 2: Dropping Columns
        st.header("Dropping Columns")
        columns_to_drop = st.multiselect("Select columns to drop:", data.columns)

        if st.button("Drop Selected Columns"):
            data.drop(columns=columns_to_drop, inplace=True)
            st.success("Selected columns dropped successfully.")

        # Section 3: Feature Engineering
        st.header("Feature Engineering")
        if st.checkbox("Add Date Column"):
            data['Date'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
            st.success("Date column added successfully.")

        if st.checkbox("Add Season Column"):
            data['Season'] = data['month'].apply(lambda x: 'Spring' if 3 <= x <= 5 else 'Summer' if 6 <= x <= 8 else 'Autumn' if 9 <= x <= 11 else 'Winter')
            st.success("Season column added successfully.")

        # Processed Data Section
        st.header("Processed Data")
        if st.button("Show Processed Data"):
            st.dataframe(data)

    else:
        st.write("Data could not be loaded.")

# Page 3: Data Visualization
def data_visualization():
    st.title("Data Visualization")
    st.write("""
    ### Overview
    This page provides interactive visualizations to explore patterns and distributions in the dataset.
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # Visualization: Histogram of AQI Distribution
        if st.checkbox("Show AQI Distribution"):
            fig = px.histogram(data, x='AQI_Bucket', nbins=30, title='AQI Distribution', marginal='box')
            st.plotly_chart(fig)

        # Visualization: Heatmap of Correlation Matrix
        if st.checkbox("Show Correlation Matrix Heatmap"):
            corr_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'AQI']
            corr_matrix = data[corr_cols].corr()
            text_annotations = np.around(corr_matrix.values, decimals=2)
            fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_cols, y=corr_cols, colorscale='Viridis', text=text_annotations, texttemplate="%{text}"))
            fig.update_layout(title='Correlation Matrix of Weather Conditions and Pollutants/AQI')
            st.plotly_chart(fig)

        # Visualization: Parallel Coordinates Plot
        if st.checkbox("Show Parallel Coordinates Plot"):
            AQI_Bucket_mapping = {'Good': 1, 'Moderate': 2, 'Unhealthy for Sensitive Groups': 3, 'Unhealthy': 4, 'Very Unhealthy': 5, 'Hazardous': 6}
            data['AQI_Bucket_Num'] = data['AQI_Bucket'].map(AQI_Bucket_mapping)
            fig = px.parallel_coordinates(data, dimensions=['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'AQI'], color='AQI_Bucket_Num', color_continuous_scale=px.colors.diverging.Tealrose, title='Parallel Coordinates Plot of Weather and AQI')
            st.plotly_chart(fig)

    else:
        st.write("Data could not be loaded.")

# Page 4: Data Modeling and Evaluation
def data_modeling():
    st.title("Data Modeling and Evaluation")
    st.write("""
    ### Overview
    This page allows you to build and evaluate predictive models for air quality analysis.
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # Feature Selection Section
        st.header("Feature Selection")
        independent_vars = st.multiselect("Select Independent Variables:", options=data.columns)
        dependent_var = st.selectbox("Select Dependent Variable:", options=data.columns)

        if independent_vars and dependent_var:
            X = data[independent_vars]
            y = data[dependent_var]

            # Train-Test Split
            st.header("Train-Test Split")
            test_size = st.slider("Select Test Size (as a percentage):", min_value=10, max_value=50, value=20, step=5)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            st.write("Training Set Size:", len(X_train))
            st.write("Testing Set Size:", len(X_test))

            # Model Selection
            st.header("Model Selection")

            # Linear Regression
            if st.checkbox("Linear Regression"):
                model_lr = LinearRegression()
                model_lr.fit(X_train, y_train)
                y_pred_lr = model_lr.predict(X_test)

                mse_lr = mean_squared_error(y_test, y_pred_lr)
                rmse_lr = np.sqrt(mse_lr)
                r2_lr = r2_score(y_test, y_pred_lr)
                mae_lr = mean_absolute_error(y_test, y_pred_lr)

                st.write("### Linear Regression Results")
                st.write(f"Mean Squared Error: {mse_lr}")
                st.write(f"Root Mean Squared Error: {rmse_lr}")
                st.write(f"R-squared: {r2_lr}")
                st.write(f"Mean Absolute Error: {mae_lr}")

            # K-Nearest Neighbors
            if st.checkbox("K-Nearest Neighbors"):
                k_value = st.slider("Select K Value:", min_value=1, max_value=20, value=5, step=1)
                weights_option = st.selectbox("Select Weight Option:", options=["uniform", "distance"])
                model_knn = KNeighborsRegressor(n_neighbors=k_value, weights=weights_option)
                model_knn.fit(X_train, y_train)
                y_pred_knn = model_knn.predict(X_test)

                mse_knn = mean_squared_error(y_test, y_pred_knn)
                rmse_knn = np.sqrt(mse_knn)
                r2_knn = r2_score(y_test, y_pred_knn)
                mae_knn = mean_absolute_error(y_test, y_pred_knn)

                st.write("### K-Nearest Neighbors Results")
                st.write(f"Mean Squared Error: {mse_knn}")
                st.write(f"Root Mean Squared Error: {rmse_knn}")
                st.write(f"R-squared: {r2_knn}")
                st.write(f"Mean Absolute Error: {mae_knn}")
    else:
        st.write("Data could not be loaded.")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Loading", "Data Preprocessing", "Data Visualization", "Data Modeling and Evaluation"])

if page == "Data Loading":
    data_loading()
elif page == "Data Preprocessing":
    data_preprocessing()
elif page == "Data Visualization":
    data_visualization()
elif page == "Data Modeling and Evaluation":
    data_modeling()
