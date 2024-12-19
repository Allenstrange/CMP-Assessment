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

# Set up the main configuration for the Streamlit app
st.set_page_config(page_title="Air Quality Analysis App", page_icon=":bar_chart:")

# Load the dataset once and cache it to optimize performance
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Allenstrange/CMP-Assessment/refs/heads/main/Air_Quality_Beijing.csv"
    return pd.read_csv(url)

# Load data into session state to persist between pages
if 'data' not in st.session_state:
    st.session_state['data'] = load_data()

# -------------------------------------
# Page 1: Data Loading
# -------------------------------------
def data_loading():
    st.title("Data Loading")
    
    st.markdown("""
    **Goal:** Before we dive into analysis, let’s get acquainted with the dataset.
    The Beijing Air Quality dataset contains hourly readings of pollutants (like PM2.5, PM10, SO2, NO2, CO, and O3) 
    and weather-related conditions (like temperature, pressure, wind speed, etc.) across multiple monitoring stations.
    
    On this page, you can:
    - **Preview raw data:** Understand how the dataset is structured.
    - **View descriptive statistics:** Get a quick summary of the data distributions and ranges.
    - **Check for missing values:** Identify data quality issues early on.
    
    This initial look at the data helps us recognize potential issues, such as missing values or outliers, 
    and sets a foundation for further preprocessing steps.
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data']
        
        # Data preview section
        st.header("Data Preview")
        st.write("""
        Preview the dataset to see how it's organized. Adjust the slider to view 
        different numbers of rows. Observing the raw dataset helps in understanding 
        the type of data (categorical, numerical, date/time) and potential data quality issues.
        """)
        num_rows = st.slider("Select Number of Rows to Preview", 1, 100, 10)
        st.dataframe(data.head(num_rows))

        # Descriptive statistics section
        st.header("Descriptive Statistics")
        st.write("""
        These statistics provide a quick snapshot of the dataset, including mean, median, min/max, 
        and quartiles. Descriptive statistics help identify the scale of variables, detect anomalies, 
        and guide decisions on data transformations.
        """)
        show_desc_table = st.checkbox("Show Descriptive Statistics Table")
        if show_desc_table:
            st.write(data.describe().T)

        # Missing values section
        st.header("Missing Values Analysis")
        st.write("""
        Missing values can bias the results or weaken the model’s predictive power. Understanding 
        which columns have missing data and how much of it is missing enables us to plan appropriate 
        strategies (imputation, dropping columns, etc.) in subsequent steps.
        """)
        show_missing_table = st.checkbox("Show Missing Values Table")
        if show_missing_table:
            missing_values = data.isnull().sum()
            missing_percentage = (data.isnull().sum() / len(data)) * 100
            missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
            st.table(missing_df)

    else:
        st.error("Error: Data could not be loaded. Please check the data source.")

# -------------------------------------
# Page 2: Data Preprocessing
# -------------------------------------
def data_preprocessing():
    st.title("Data Preprocessing")
    
    st.markdown("""
    **Goal:** Transform the raw dataset into a cleaner, more useful form.  
    By preprocessing the data, we improve its quality and relevance. This step ensures that 
    subsequent analyses and models are built on a solid foundation.
    
    On this page, you can:
    - **Handle missing data:** Impute missing values using mean, median, or mode.
    - **Drop irrelevant columns:** Remove columns that may not contribute to analysis.
    - **Explore data distributions:** Identify skewness or outliers that might influence models.
    - **Feature engineering:** Add new columns (e.g., date/time features, seasonal indicators, 
      or an AQI column) to enhance the dataset’s value.
    
    Proper preprocessing ensures that the final dataset is well-prepared for visualization, 
    correlation analysis, and modeling.
    """)

    if st.session_state['data'] is not None:
        # Work on a copy so changes persist across the session
        data = st.session_state['data'].copy()

        # Handling Missing Values
        st.header("Handling Missing Values")
        st.write("""
        Choose a method to fill in missing values. The choice depends on the type and distribution of data:
        - **Mean:** Useful for approximately symmetric distributions.
        - **Median:** More robust to outliers and skewed distributions.
        - **Mode:** Suitable for categorical data or data with repeating values.
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
            st.session_state['data'] = data  # Update session state data after imputation

        # Dropping Columns
        st.header("Dropping Columns")
        st.write("""
        Removing irrelevant or redundant columns simplifies the dataset and can improve model 
        performance by reducing noise. Select columns that do not add value to the analysis.
        """)
        columns_to_drop = st.multiselect("Select columns to drop:", data.columns)
        if st.button("Drop Selected Columns"):
            data.drop(columns=columns_to_drop, inplace=True)
            st.success("Selected columns dropped successfully.")
            st.session_state['data'] = data  # Update session state after dropping columns

        # Data Distribution Exploration
        st.header("Distribution Exploration")
        st.write("""
        Visualizing distributions can reveal insights about the data's shape, spread, and 
        presence of outliers. This informs the choice of imputation methods, transformations, 
        and modeling approaches.
        """)
        selected_columns = st.multiselect(
            "Select columns to display histograms",
            data.columns,
            default=['PM2.5', 'PM10', 'TEMP', 'PRES']
        )

        if selected_columns:
            fig, axes = plt.subplots(len(selected_columns), 1, figsize=(10, 5 * len(selected_columns)))
            if len(selected_columns) == 1:
                axes = [axes]
            for ax, col in zip(axes, selected_columns):
                data[col].hist(ax=ax, bins=30)
                ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

        # Feature Engineering
        st.header("Feature Engineering")
        st.write("""
        By deriving new features from existing data, we can highlight important patterns and 
        improve model performance. Consider adding:
        - A **Date** column to analyze temporal trends.
        - A **Season** column to capture seasonal variations.
        - An **AQI** (Air Quality Index) column to simplify pollutant measurements into a single index.
        - An **AQI_Bucket** column to classify AQI levels into categories like 'Good' or 'Unhealthy'.
        """)

        # Add Date Column
        if st.checkbox("Add Date Column"):
            data['Date'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
            st.success("Date column added successfully.")
            st.session_state['data'] = data

        # Add Season Column
        if st.checkbox("Add Season Column"):
            data['Season'] = data['month'].apply(
                lambda x: 'Spring' if 3 <= x <= 5 else 
                          'Summer' if 6 <= x <= 8 else 
                          'Autumn' if 9 <= x <= 11 else 
                          'Winter'
            )
            st.success("Season column added successfully.")
            st.session_state['data'] = data

        # Add AQI Column
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

                def get_aqi(concentration, bps):
                    for bp in bps:
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
            st.session_state['data'] = data

        # Add AQI_Bucket Column
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
            st.session_state['data'] = data

        # Show processed data
        st.header("View Processed Data")
        st.write("""
        After applying all preprocessing steps, you can take a final look at the dataset. 
        Confirm that the imputation, column drops, and feature engineering steps have produced 
        the desired transformations.
        """)
        if st.button("Show Processed Data"):
            st.dataframe(data)
    else:
        st.error("Data could not be loaded.")

# -------------------------------------
# Page 3: Data Visualization
# -------------------------------------
def data_visualization():
    st.title("Data Visualization")
    
    st.markdown("""
    **Goal:** Gain insights into data patterns and relationships through visual exploration.
    
    On this page, you can:
    - Examine pollution levels across various stations to identify hotspots.
    - Compare different pollutant averages to understand their relative significance.
    - Investigate the distribution of AQI categories.
    - Explore correlations between pollutants and weather parameters.
    - Visualize multivariate relationships using parallel coordinates.
    
    Interactive and intuitive charts help highlight trends, seasonal effects, 
    and correlations that might be less obvious in raw data tables.
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # Average Pollution Levels by Station
        if st.checkbox("Show Average Pollution Levels by Station"):
            st.write("""
            View how different stations vary in terms of average pollution levels. 
            Identifying stations with consistently higher pollutant concentrations 
            can guide targeted policy interventions.
            """)
            station_stats = data.groupby('station')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean().reset_index()
            station_stats_melted = pd.melt(
                station_stats,
                id_vars=['station'],
                value_vars=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'],
                var_name='Pollutant',
                value_name='Average Concentration'
            )
            fig = px.bar(
                station_stats_melted,
                x='station',
                y='Average Concentration',
                color='Pollutant',
                barmode='stack',
                title='Average Pollution Levels by Station'
            )
            st.plotly_chart(fig)

        # Average Concentration of Each Pollutant
        if st.checkbox("Show Average Concentration of Each Pollutant"):
            st.write("""
            Compare the average levels of each pollutant to identify which contaminants 
            are most prevalent. This helps prioritize which pollutants to focus on 
            for targeted improvements.
            """)
            pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
            mean_pollutants = data[pollutants].mean()
            fig = px.bar(
                x=pollutants,
                y=mean_pollutants,
                title='Average Concentration of Each Pollutant',
                labels={'x': 'Pollutant', 'y': 'Average Concentration'}
            )
            st.plotly_chart(fig)

        # AQI Distribution
        if st.checkbox("Show AQI Distribution"):
            st.write("""
            Examine how frequently the air quality falls into different AQI categories. 
            Understanding the distribution of AQI levels helps gauge overall air quality 
            conditions and frequency of poor air quality events.
            """)
            fig = px.histogram(
                data,
                x='AQI_Bucket',
                nbins=30,
                title='AQI Distribution',
                marginal='box'
            )
            st.plotly_chart(fig)

        # Correlation Matrix Heatmap
        if st.checkbox("Show Correlation Matrix Heatmap"):
            st.write("""
            A correlation matrix reveals how closely different variables move together. 
            Strong correlations may indicate redundant variables or potential predictors 
            for modeling. Weather variables correlated with pollutants can help 
            understand environmental factors influencing air quality.
            """)
            corr_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'AQI']
            corr_matrix = data[corr_cols].corr()
            text_annotations = np.around(corr_matrix.values, decimals=2)
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_cols,
                y=corr_cols,
                colorscale='Viridis',
                text=text_annotations,
                texttemplate="%{text}"
            ))
            fig.update_layout(title='Correlation Matrix of Weather Conditions and Pollutants/AQI')
            st.plotly_chart(fig)

        # Parallel Coordinates Plot
        if st.checkbox("Show Parallel Coordinates Plot of Weather and AQI"):
            st.write("""
            Parallel coordinates allow you to visualize multi-dimensional relationships 
            in a single chart. Here, you can see how various weather factors and AQI 
            levels interact. Color-coding by AQI category makes it easier to compare 
            patterns between different air quality levels.
            """)
            AQI_Bucket_mapping = {
                'Good': 1,
                'Moderate': 2,
                'Unhealthy for Sensitive Groups': 3,
                'Unhealthy': 4,
                'Very Unhealthy': 5,
                'Hazardous': 6
            }
            data['AQI_Bucket_Num'] = data['AQI_Bucket'].map(AQI_Bucket_mapping)
            fig = px.parallel_coordinates(
                data,
                dimensions=['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'AQI'],
                color='AQI_Bucket_Num',
                color_continuous_scale=px.colors.diverging.Tealrose,
                title='Parallel Coordinates Plot of Weather and AQI'
            )
            fig.update_layout(
                coloraxis_colorbar=dict(
                    tickvals=list(AQI_Bucket_mapping.values()),
                    ticktext=list(AQI_Bucket_mapping.keys())
                )
            )
            st.plotly_chart(fig)
    else:
        st.error("Data could not be loaded.")

# -------------------------------------
# Page 4: Data Modeling and Evaluation
# -------------------------------------
def data_modeling():
    st.title("Data Modeling and Evaluation")
    
    st.markdown("""
    **Goal:** Leverage the cleaned and explored dataset to build predictive models.
    
    On this page, you can:
    - **Select Features:** Choose independent variables (X) and a target variable (y).
    - **Apply Standardization:** Ensure all features contribute equally to model training.
    - **Split the Data:** Create training and testing subsets to evaluate model generalization.
    - **Train and Compare Models:** Try different algorithms like Linear Regression for 
      baseline understanding and K-Nearest Neighbors for non-linear patterns.
    - **Evaluate Performance:** Use metrics such as MSE, RMSE, R², MAE to judge model accuracy.
    - **Hyperparameter Tuning:** Use Grid Search CV to optimize model parameters for better performance.
    
    By understanding model performance and tuning parameters, we can improve predictions 
    and gain deeper insights into the factors driving air quality.
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # Feature Selection
        st.header("Feature Selection")
        st.write("""
        Choose which variables to use as features (X) and which variable to predict (y). 
        Consider using the visualization insights and correlation findings to pick 
        informative variables.
        """)
        independent_vars = st.multiselect("Select Independent Variables:", options=data.columns)
        dependent_var = st.selectbox("Select Dependent Variable:", options=data.columns)

        if independent_vars and dependent_var:
            X = data[independent_vars]
            y = data[dependent_var]

            # Data Standardization
            st.header("Data Standardization")
            st.write("""
            Standardization transforms features to have zero mean and unit variance. 
            This can help models that are sensitive to feature scales (like KNN) 
            perform more consistently.
            """)
            standardize = st.checkbox("Standardize Data using StandardScaler")

            if standardize:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            # Train-Test Split
            st.header("Train-Test Split")
            st.write("""
            Splitting the dataset into training and testing sets ensures we can evaluate 
            the model’s ability to generalize to unseen data. Choose the test size below.
            """)
            test_size = st.slider("Select Test Size (as a percentage):", min_value=10, max_value=50, value=20, step=5)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            st.write(f"Training Set Size: {len(X_train)} rows")
            st.write(f"Testing Set Size: {len(X_test)} rows")

            # Model Selection and Evaluation
            st.header("Model Selection and Evaluation")
            st.write("""
            Choose from different models and evaluate their performance:
            - **Linear Regression:** Provides a simple baseline and interpretable coefficients.
            - **K-Nearest Neighbors:** Captures non-linear patterns. Adjust k and weight parameters.
            """)

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
                st.write(f"**Mean Squared Error (MSE):** {mse_lr}")
                st.write(f"**Root Mean Squared Error (RMSE):** {rmse_lr}")
                st.write(f"**R-squared (R²):** {r2_lr}")
                st.write(f"**Mean Absolute Error (MAE):** {mae_lr}")

            # K-Nearest Neighbors
            if st.checkbox("K-Nearest Neighbors"):
                st.write("""
                Adjust the number of neighbors (k) and select how distances affect predictions 
                (uniform or distance-weighted) to find the best setup.
                """)
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
                st.write(f"**Mean Squared Error (MSE):** {mse_knn}")
                st.write(f"**Root Mean Squared Error (RMSE):** {rmse_knn}")
                st.write(f"**R-squared (R²):** {r2_knn}")
                st.write(f"**Mean Absolute Error (MAE):** {mae_knn}")

                # Grid Search for KNN
                if st.checkbox("Perform Grid Search for KNN"):
                    st.write("""
                    Grid Search tests various parameter combinations automatically, 
                    helping find the best values for k and weights. This can improve 
                    model performance significantly.
                    """)
                    param_grid = {
                        'n_neighbors': range(1, 21),
                        'weights': ['uniform', 'distance']
                    }
                    grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
                    grid_search.fit(X_train, y_train)

                    st.write("### Best Parameters from Grid Search")
                    st.write(grid_search.best_params_)
                    st.write(f"**Best Cross-Validated Score (MSE):** {-grid_search.best_score_}")

    else:
        st.error("Data could not be loaded.")

# -------------------------------------
# Sidebar Navigation
# -------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Loading", "Data Preprocessing", "Data Visualization", "Data Modeling and Evaluation"])

# Additional sidebar info
st.sidebar.markdown("""
---
### About This App

This application provides an end-to-end analysis pipeline for Beijing's air quality data:
1. **Data Loading:** Get an overview of the raw dataset.
2. **Data Preprocessing:** Clean and enhance data to improve analysis quality.
3. **Data Visualization:** Uncover patterns and insights through interactive charts.
4. **Data Modeling and Evaluation:** Build and assess predictive models to understand what influences air quality.

Use the navigation above to explore each stage of the analysis.
""")

# Page navigation logic
if page == "Data Loading":
    data_loading()
elif page == "Data Preprocessing":
    data_preprocessing()
elif page == "Data Visualization":
    data_visualization()
elif page == "Data Modeling and Evaluation":
    data_modeling()
