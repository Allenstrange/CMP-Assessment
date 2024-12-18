import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

st.set_page_config(page_title="Air Quality Analysis App", page_icon=":bar_chart:")

# Load the dataset locally
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
    
    # Add page description
    st.markdown("""
    This page allows you to explore the Beijing Air Quality dataset. The data contains hourly measurements of various air pollutants 
    and weather conditions across different monitoring stations in Beijing. Here you can:
    
    - Preview the raw data
    - View basic descriptive statistics
    - Check for missing values in the dataset
    
    This initial exploration helps understand the structure and quality of our data before proceeding with analysis.
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data']
        
        # Data preview
        st.header("Data Preview")
        st.write("""
        Below is a preview of our dataset. Use the slider to adjust how many rows you'd like to see. 
        The dataset includes measurements of pollutants like PM2.5, PM10, SO2, NO2, CO, and O3, 
        along with weather conditions like temperature, pressure, and wind speed.
        """)
        num_rows = st.slider("Select Number of Rows to Preview", 1, 100, 10)
        st.dataframe(data.head(num_rows))

        # Descriptive Statistics
        st.header("Descriptive Statistics")
        st.write("""
        These statistics provide a summary of the numerical features in our dataset, including:
        - Count of non-null values
        - Mean, standard deviation
        - Minimum and maximum values
        - Quartile distributions
        """)
        show_desc_table = st.checkbox("Show Descriptive Statistics Table")
        if show_desc_table:
            st.write(data.describe().T)

        # Missing Values
        st.header("Missing Values Analysis")
        st.write("""
        Understanding missing values is crucial for data quality assessment. This section shows:
        - Number of missing values per column
        - Percentage of missing data
        This information helps determine appropriate strategies for data cleaning.
        """)
        show_missing_table = st.checkbox("Show Missing Values Table")
        if show_missing_table:
            missing_values = data.isnull().sum()
            missing_percentage = (data.isnull().sum() / len(data)) * 100
            missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
            st.table(missing_df)

    else:
        st.error("Error: Data could not be loaded. Please check the data source.")

# Page 2: Data Preprocessing
def data_preprocessing():
    st.title("Data Preprocessing")
    
    # Add page description
    st.markdown("""
    This page focuses on cleaning and preparing the data for analysis. Here you can:
    
    - Handle missing values using different imputation methods
    - Remove unnecessary columns
    - Create new features to enrich the analysis
    - Explore data distributions to inform preprocessing decisions
    
    Proper preprocessing is crucial for generating reliable insights and building effective models.
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data'].copy()

        # Section 1: Handling Missing Values
        st.header("Handling Missing Values")
        st.write("""
        Choose an appropriate method to fill in missing values:
        - Mean: Best for normally distributed data
        - Median: Recommended for skewed distributions
        - Mode: Useful for categorical data
        """)
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

        # Rest of the preprocessing code remains the same...

# Page 3: Data Visualization
def data_visualization():
    st.title("Data Visualization")
    
    # Add page description
    st.markdown("""
    This page provides interactive visualizations to help understand patterns and relationships in the air quality data. You can explore:
    
    - Pollution levels across different monitoring stations
    - Distribution of Air Quality Index (AQI)
    - Correlations between weather conditions and pollutants
    - Temporal patterns in air quality
    
    Use the checkboxes below to toggle different visualizations and gain insights into Beijing's air quality patterns.
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # Rest of the visualization code remains the same...

# Page 4: Data Modeling and Evaluation
def data_modeling():
    st.title("Data Modeling and Evaluation")
    
    # Add page description
    st.markdown("""
    This page allows you to build and evaluate predictive models for air quality analysis. You can:
    
    - Select features for modeling
    - Choose between different machine learning algorithms
    - Tune model parameters
    - Evaluate model performance using various metrics
    
    The models can help understand relationships between variables and predict air quality indicators.
    
    Key Features:
    - Linear Regression for understanding linear relationships
    - K-Nearest Neighbors for non-linear patterns
    - Grid Search CV for parameter optimization
    - Various performance metrics (MSE, RMSE, R², MAE)
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # Rest of the modeling code remains the same...

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Loading", "Data Preprocessing", "Data Visualization", "Data Modeling and Evaluation"])

# Add sidebar description
st.sidebar.markdown("""
---
### About This App
This application analyzes air quality data from Beijing, helping users understand pollution patterns
and their relationships with weather conditions. Use the navigation above to move through different
stages of the analysis pipeline.
""")

if page == "Data Loading":
    data_loading()
elif page == "Data Preprocessing":
    data_preprocessing()
elif page == "Data Visualization":
    data_visualization()
elif page == "Data Modeling and Evaluation":
    data_modeling()
