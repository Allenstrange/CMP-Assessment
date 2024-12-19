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

# Configure the page
st.set_page_config(page_title="Enhanced Streamlit App", page_icon=":bar_chart:", layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Allenstrange/CMP-Assessment/refs/heads/main/Air_Quality_Beijing.csv"
    return pd.read_csv(url)

# Load data into session state
if 'data' not in st.session_state:
    st.session_state['data'] = load_data()

# Function to reset data to original
if 'original_data' not in st.session_state:
    st.session_state['original_data'] = st.session_state['data'].copy()

def reset_data():
    st.session_state['data'] = st.session_state['original_data'].copy()
    st.success("Data reset to original state.")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Loading", "Data Preprocessing", "Data Visualization", "Data Modeling and Evaluation"])

if st.sidebar.button("Reset Data"):
    reset_data()

# Data Loading Page
def data_loading():
    st.title("Data Loading 🛠️")

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # Dataset Metadata
        st.write("### Dataset Metadata")
        st.write(f"**Number of Rows:** {data.shape[0]}")
        st.write(f"**Number of Columns:** {data.shape[1]}")
        st.write("**Column Data Types:**")
        st.dataframe(data.dtypes, use_container_width=True)

        # Data Preview
        num_rows = st.slider("Select Number of Rows to Preview", 1, 100, 10)
        st.write("### Data Preview 🔍")
        st.dataframe(data.head(num_rows))

        # Descriptive Statistics Options
        st.write("### Descriptive Statistics 📊")
        if st.checkbox("Show Descriptive Statistics Table"):
            st.write(data.describe().T)

        # Missing Value Visualization
        st.write("### Missing Values ❓")
        if st.checkbox("Visualize Missing Values"):
            missing_df = data.isnull().sum().reset_index()
            missing_df.columns = ['Column', 'Missing Values']
            missing_df['Percentage'] = (missing_df['Missing Values'] / data.shape[0]) * 100
            fig = px.bar(missing_df, x='Column', y='Percentage', title='Percentage of Missing Values')
            st.plotly_chart(fig)

# Data Preprocessing Page
def data_preprocessing():
    st.title("Data Preprocessing 🧹")

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # Handle Missing Values
        st.header("Handling Missing Values ❓")
        imputation_method = st.radio("Choose an imputation method:", ["Mean", "Median", "Mode"], index=0)
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

        # Drop Columns
        st.header("Dropping Columns 🗑️")
        columns_to_drop = st.multiselect("Select columns to drop:", data.columns)

        if st.button("Drop Selected Columns"):
            data.drop(columns=columns_to_drop, inplace=True)
            st.success("Selected columns dropped successfully.")

        # Feature Engineering
        st.header("Feature Engineering 🛠️")
        if st.checkbox("Add Date Column"):
            data['Date'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
            st.success("Date column added successfully.")

        if st.checkbox("Add Season Column"):
            data['Season'] = data['month'].apply(lambda x: 'Spring' if 3 <= x <= 5 else 'Summer' if 6 <= x <= 8 else 'Autumn' if 9 <= x <= 11 else 'Winter')
            st.success("Season column added successfully.")

        if st.checkbox("Add AQI Column"):
            def calculate_aqi(row):
                breakpoints = {
                    'PM2.5': [(0, 35, 0, 50), (36, 75, 51, 100), (76, 115, 101, 150), (116, 150, 151, 200), (151, 250, 201, 300), (251, 500, 301, 500)],
                    'PM10': [(0, 50, 0, 50), (51, 150, 51, 100), (151, 250, 101, 150), (251, 350, 151, 200), (351, 420, 201, 300), (421, 600, 301, 500)],
                    'SO2': [(0, 150, 0, 50), (151, 500, 51, 100), (501, 650, 101, 150), (651, 800, 151, 200), (801, 1600, 201, 300), (1601, 2100, 301, 500)],
                    'NO2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 150), (181, 280, 151, 200), (281, 560, 201, 300), (561, 940, 301, 500)],
                    'CO': [(0, 2, 0, 50), (2.1, 4, 51, 100), (4.1, 14, 101, 150), (14.1, 24, 151, 200), (24.1, 36, 201, 300), (36.1, 60, 301, 500)],
                    'O3': [(0, 180, 0, 50), (181, 240, 51, 100), (241, 340, 101, 150), (341, 420, 151, 200), (421, 500, 201, 300), (501, 800, 301, 500)]
                }

                def get_aqi(concentration, breakpoints):
                    for bp in breakpoints:
                        if bp[0] <= concentration <= bp[1]:
                            return int((bp[2] + bp[3]) / 2)
                    return 0

                aqi_values = {
                    'PM2.5': get_aqi(row['PM2.5'], breakpoints['PM2.5']),
                    'PM10': get_aqi(row['PM10'], breakpoints['PM10']),
                    'SO2': get_aqi(row['SO2'], breakpoints['SO2']),
                    'NO2': get_aqi(row['NO2'], breakpoints['NO2']),
                    'CO': get_aqi(row['CO'], breakpoints['CO']),
                    'O3': get_aqi(row['O3'], breakpoints['O3']),
                }

                return max(aqi_values.values())

            data['AQI'] = data.apply(calculate_aqi, axis=1)
            st.success("AQI column added successfully.")

        if st.checkbox("Add AQI_Bucket Column"):
            def get_aqi_bucket(aqi):
                if aqi <= 50:
                    return 'Good'
                elif aqi <= 100:
                    return 'Moderate'
                elif aqi <= 150:
                    return 'Unhealthy for Sensitive Groups'
                elif aqi <= 200:
                    return 'Unhealthy'
                elif aqi <= 300:
                    return 'Very Unhealthy'
                else:
                    return 'Hazardous'

            data['AQI_Bucket'] = data['AQI'].apply(get_aqi_bucket)
            st.success("AQI_Bucket column added successfully.")

        st.header("Processed Data 📑")
        if st.button("Show Processed Data"):
            st.dataframe(data)

# Data Visualization Page
def data_visualization():
    st.title("Data Visualization 📊")

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        st.write("### Visualizations 🖼️")

        # Average Pollution Levels by Station
        if st.checkbox("Show Average Pollution Levels by Station"):
            station_stats = data.groupby('station')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean().reset_index()
            station_stats_melted = pd.melt(station_stats, id_vars=['station'], value_vars=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'], var_name='Pollutant', value_name='Average Concentration')
            fig = px.bar(station_stats_melted, x='station', y='Average Concentration', color='Pollutant', barmode='stack', title='Average Pollution Levels by Station')
            st.plotly_chart(fig)

        # Correlation Matrix Heatmap
        if st.checkbox("Show Correlation Matrix Heatmap"):
            corr_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
            corr_matrix = data[corr_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)

# Data Modeling and Evaluation Page
def data_modeling():
    st.title("Data Modeling and Evaluation 🤖")

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        st.header("Feature Selection 🔬")
        independent_vars = st.multiselect("Select Independent Variables:", options=data.columns)
        dependent_var = st.selectbox("Select Dependent Variable:", options=data.columns)

        if independent_vars and dependent_var:
            X = data[independent_vars]
            y = data[dependent_var]

            # Standardization
            st.header("Data Standardization ⚖️")
            standardize = st.checkbox("Standardize Data using StandardScaler")

            if standardize:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            # Train-Test Split
            st.header("Train-Test Split ✂️")
            test_size = st.slider("Select Test Size (as a percentage):", 10, 50, 20, 5)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            st.write("Training Set Size:", len(X_train))
            st.write("Testing Set Size:", len(X_test))

            # Model Selection
            st.header("Model Selection 🛠️")
            if st.checkbox("Linear Regression"):
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.write("### Linear Regression Results")
                st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
                st.write(f"R-squared: {r2_score(y_test, y_pred):.2f}")

            if st.checkbox("K-Nearest Neighbors"):
                k_value = st.slider("Select K Value:", min_value=1, max_value=20, value=5, step=1)
                weights_option = st.selectbox("Select Weight Option:", options=["uniform", "distance"])
                model_knn = KNeighborsRegressor(n_neighbors=k_value, weights=weights_option)
                model_knn.fit(X_train, y_train)
                y_pred_knn = model_knn.predict(X_test)

                st.write("### K-Nearest Neighbors Results")
                st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_knn):.2f}")
                st.write(f"R-squared: {r2_score(y_test, y_pred_knn):.2f}")

# Page Navigation
if page == "Data Loading":
    data_loading()
elif page == "Data Preprocessing":
    data_preprocessing()
elif page == "Data Visualization":
    data_visualization()
elif page == "Data Modeling and Evaluation":
    data_modeling()
