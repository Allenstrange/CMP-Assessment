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

        # Section 2: Dropping Columns
        st.header("Dropping Columns")
        st.write("Select columns that you want to remove from the dataset. This can help simplify the analysis by removing irrelevant features.")
        columns_to_drop = st.multiselect("Select columns to drop:", data.columns)

        if st.button("Drop Selected Columns"):
            data.drop(columns=columns_to_drop, inplace=True)
            st.success("Selected columns dropped successfully.")

        # Data Exploration: Pollutant and Weather Distributions
        st.write("### Data Exploration")
        st.write("""This data exploration aims to analyze feature distributions and identify skewness to inform the imputation of missing values.
The mean will be used for normally distributed features, while the median will be applied to skewed features.
This approach ensures the most accurate imputation for each feature, preserving data integrity for subsequent analysis.""")
        # Select columns for histograms
        selected_columns = st.multiselect("Select columns to display histograms",
                                        data.columns,
                                        default=['PM2.5', 'PM10', 'TEMP', 'PRES'])  # Default to some columns

        # Create histograms for selected columns
        if selected_columns:
            num_cols = 2  # Number of columns in the subplot grid
            num_rows = int(np.ceil(len(selected_columns) / num_cols))  # Number of rows

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6 * num_rows))  # Adjust figsize as needed
            axes = axes.flatten()  # Flatten the axes array for easier iteration

            for i, col in enumerate(selected_columns):
                ax = axes[i]
                data[col].hist(ax=ax, bins=30)  # Adjust bins as needed
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')

            plt.tight_layout()
            st.pyplot(fig)  # Display the figure in Streamlit

        # Feature Engineering Section
        st.header("Feature Engineering")
        st.write("Create new features to enhance the analysis. These derived features can provide additional insights into the data.")
        
        if st.checkbox("Add Date Column"):
            data['Date'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
            st.success("Date column added successfully.")

        if st.checkbox("Add Season Column"):
            data['Season'] = data['month'].apply(lambda x: 'Spring' if 3 <= x <= 5 else 'Summer' if 6 <= x <= 8 else 'Autumn' if 9 <= x <= 11 else 'Winter')
            st.success("Season column added successfully.")

        if st.checkbox("Add AQI Column"):
            breakpoints = {
                'PM2.5': [(0, 35, 0, 50), (36, 75, 51, 100), (76, 115, 101, 150), (116, 150, 151, 200), (151, 250, 201, 300), (251, 500, 301, 500)],
                'PM10': [(0, 50, 0, 50), (51, 150, 51, 100), (151, 250, 101, 150), (251, 350, 151, 200), (351, 420, 201, 300), (421, 600, 301, 500)],
                'SO2': [(0, 150, 0, 50), (151, 500, 51, 100), (501, 650, 101, 150), (651, 800, 151, 200), (801, 1600, 201, 300), (1601, 2100, 301, 500)],
                'NO2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 150), (181, 280, 151, 200), (281, 560, 201, 300), (561, 940, 301, 500)],
                'CO': [(0, 2, 0, 50), (2.1, 4, 51, 100), (4.1, 14, 101, 150), (14.1, 24, 151, 200), (24.1, 36, 201, 300), (36.1, 60, 301, 500)],
                'O3': [(0, 180, 0, 50), (181, 240, 51, 100), (241, 340, 101, 150), (341, 420, 151, 200), (421, 500, 201, 300), (501, 800, 301, 500)]
            }

            def calculate_aqi(row):
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

        # Processed Data Section
        st.header("Processed Data")
        st.write("View the data after all preprocessing steps have been applied.")
        if st.button("Show Processed Data"):
            st.dataframe(data)

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

        st.write("### Visualizations")

        # Visualization 3: Stacked Bar Chart of Average Pollution Levels by Station
        if st.checkbox("Show Average Pollution Levels by Station"):
            st.write("""
            This visualization shows the average concentration of different pollutants at each monitoring station.
            The stacked bars help compare the total pollution load across stations while showing the contribution of each pollutant.
            """)
            station_stats = data.groupby('station')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean().reset_index()
            station_stats_melted = pd.melt(station_stats, id_vars=['station'], value_vars=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'], var_name='Pollutant', value_name='Average Concentration')
            fig = px.bar(station_stats_melted, x='station', y='Average Concentration', color='Pollutant', barmode='stack', title='Average Pollution Levels by Station')
            st.plotly_chart(fig)

        # Visualization 4: Bar Chart of Average Concentration of Each Pollutant
        if st.checkbox("Show Average Concentration of Each Pollutant"):
            st.write("""
            This chart displays the average concentration of each pollutant across all stations.
            It helps identify which pollutants are present in higher concentrations overall.
            """)
            pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
            mean_pollutants = data[pollutants].mean()
            fig = px.bar(x=pollutants, y=mean_pollutants, title='Average Concentration of Each Pollutant', labels={'x': 'Pollutant', 'y': 'Average Concentration'})
            st.plotly_chart(fig)

        # Visualization 5: Histogram of AQI Distribution
        if st.checkbox("Show AQI Distribution"):
            st.write("""
            This histogram shows the distribution of Air Quality Index (AQI) values.
            The distribution helps understand the frequency of different air quality levels.
            """)
            fig = px.histogram(data, x='AQI_Bucket', nbins=30, title='AQI Distribution', marginal='box')
            st.plotly_chart(fig)

        # Visualization 6: Heatmap of Correlation Matrix
        if st.checkbox("Show Correlation Matrix Heatmap"):
            st.write("""
            This heatmap shows correlations between different variables.
            Stronger colors indicate stronger correlations (positive or negative).
            This helps identify relationships between weather conditions and pollution levels.
            """)
            corr_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'AQI']
            corr_matrix = data[corr_cols].corr()
            text_annotations = np.around(corr_matrix.values, decimals=2)
            fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_cols, y=corr_cols, colorscale='Viridis', text=text_annotations, texttemplate="%{text}"))
            fig.update_layout(title='Correlation Matrix of Weather Conditions and Pollutants/AQI')
            st.plotly_chart(fig)

        # Visualization 7: Stacked Bar Chart of AQI Distribution by Station
        if st.checkbox("Show AQI Distribution by Station"):
            st.write("""
            This chart shows how air quality levels are distributed across different stations.
            The stacked bars represent different AQI categories, helping compare air quality patterns between stations.
            """)
            fig = px.histogram(data, x='station', color='AQI_Bucket', title='AQI Distribution by Station', barmode='stack', 
                             category_orders={'AQI_Bucket': ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']})
            st.plotly_chart(fig)

        # Visualization 8: Parallel Coordinates Plot of Weather and AQI
        if st.checkbox("Show Parallel Coordinates Plot of Weather and AQI"):
            st.write("""
            This parallel coordinates plot shows relationships between weather conditions and AQI.
            Each line represents a data point, and the path it takes shows the relationships between variables.
            This helps identify patterns between weather conditions and air quality.
            """)
            AQI_Bucket_mapping = {'Good': 1, 'Moderate': 2, 'Unhealthy for Sensitive Groups': 3, 'Unhealthy': 4, 'Very Unhealthy': 5, 'Hazardous': 6}
            data['AQI_Bucket_Num'] = data['AQI_Bucket'].map(AQI_Bucket_mapping)
            fig = px.parallel_coordinates(data, dimensions=['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'AQI'], 
                                       color='AQI_Bucket_Num', color_continuous_scale=px.colors.diverging.Tealrose,
                                       title='Parallel Coordinates Plot of Weather and AQI')
            fig.update_layout(coloraxis_colorbar=dict(tickvals=list(AQI_Bucket_mapping.values()), 
                                                    ticktext=list(AQI_Bucket_mapping.keys())))
            st.plotly_chart(fig)

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

        # Feature Selection Section
        st.header("Feature Selection")
        st.write("""
        Select the variables for your model:
        - Independent Variables: The features you want to use for prediction
        - Dependent Variable: The target variable you want to predict
        """)

        # Correlation Heatmap
        if st.checkbox("Show Correlation Heatmap"):
            st.write("""
            This heatmap helps identify correlations between variables to inform feature selection.
            Stronger correlations (closer to 1 or -1) indicate stronger relationships.
            """)
            corr_cols = data.select_dtypes(include=[np.number]).columns
            corr_matrix = data[corr_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)

        # Select Dependent and Independent Variables
        independent_vars = st.multiselect("Select Independent Variables:", options=data.columns)
        dependent_var = st.selectbox("Select Dependent Variable:", options=data.columns)

        if independent_vars and dependent_var:
            X = data[independent_vars]
            y = data[dependent_var]

            # Standardization Option
            st.header("Data Standardization")
            st.write("""
            Standardization scales the features to have zero mean and unit variance.
            This is important when features are on different scales or when using distance-based algorithms like KNN.
            """)
            standardize = st.checkbox("Standardize Data using StandardScaler")

            if standardize:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            # Train-Test Split
            st.header("Train-Test Split")
            st.write("""
            Split the data into training and testing sets:
            - Training set: Used to train the model
            - Testing set: Used to evaluate model performance
            Adjust the test size using the slider below.
            """)
            test_size = st.slider("Select Test Size (as a percentage):", min_value=10, max_value=50, value=20, step=5)
            X_train, X_test, y_train, y_test

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
