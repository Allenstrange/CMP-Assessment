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
    st.title("Data Loading :open_file_folder:")

    st.markdown("""
    **Goal:** Before we dive into analysis, let’s get acquainted with the dataset.
    
    The Beijing Air Quality dataset contains hourly readings of pollutants (like PM2.5, PM10, SO2, NO2, CO, and O3) 
    and weather-related conditions (like temperature, pressure, wind speed, etc.) across multiple monitoring stations.
    
    On this page, you can:
    - **Preview raw data** :mag: 
    - **View descriptive statistics** :bar_chart:
    - **Check for missing values** :grey_question:
    
    This initial overview helps ensure we understand the data structure, identify potential quality issues, 
    and set the stage for further preprocessing. :seedling:
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data']
        
        # Data preview section
        st.header("Data Preview :eyes:")
        st.write("""
        Get a quick look at the dataset structure. Adjust the slider to view more or fewer rows. 
        Understanding the raw data layout is the first step toward making informed decisions in the analysis process.
        """)
        num_rows = st.slider("Select Number of Rows to Preview", 1, 100, 10)
        st.dataframe(data.head(num_rows))

        # Descriptive statistics section
        st.header("Descriptive Statistics :clipboard:")
        st.write("""
        These statistics provide insights into the central tendencies (mean, median) and spread (standard deviation) 
        of numerical features. This helps identify any anomalies or unexpected ranges in the data.
        """)
        show_desc_table = st.checkbox("Show Descriptive Statistics Table")
        if show_desc_table:
            st.write(data.describe().T)

        # Missing values section
        st.header("Missing Values Analysis :question:")
        st.write("""
        Missing values can affect data quality and model performance. Understanding which columns have missing values 
        and how many is critical for data cleaning and imputation strategies.
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
    st.title("Data Preprocessing :broom:")

    st.markdown("""
    **Goal:** Transform the raw dataset into a cleaner, more useful form. :sparkles:
    
    On this page, you can:
    - **Handle missing values** :umbrella:
    - **Remove irrelevant columns** :wastebasket:
    - **Explore data distributions** :bar_chart:
    - **Create new features** :hammer_and_wrench:
    
    Proper preprocessing ensures that the final dataset is well-prepared for visualization, 
    correlation analysis, and modeling. Let's set the stage for robust analysis and reliable insights! :rocket:
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data'].copy()

        # Handling Missing Values
        st.header("Handling Missing Values :umbrella:")
        st.write("""
        Choose a method to address missing values:
        - **Mean:** :balance_scale: Ideal for approximately normal distributions.
        - **Median:** :scales: Useful for skewed distributions.
        - **Mode:** :repeat: Good for categorical or frequently repeating values.
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
            st.session_state['data'] = data

        # Dropping Columns
        st.header("Dropping Columns :wastebasket:")
        st.write("""
        Remove columns that are irrelevant or redundant. Streamlining the dataset can improve 
        performance and clarity, making subsequent analysis more focused.
        """)
        columns_to_drop = st.multiselect("Select columns to drop:", data.columns)
        if st.button("Drop Selected Columns"):
            data.drop(columns=columns_to_drop, inplace=True)
            st.success("Selected columns dropped successfully.")
            st.session_state['data'] = data

        # Data Distribution Exploration
        st.header("Distribution Exploration :bar_chart:")
        st.write("""
        Visualizing the distribution of features can reveal skewness, outliers, or unexpected patterns. 
        This helps guide decisions on transformations and better informs imputation methods.
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
        st.header("Feature Engineering :hammer_and_wrench:")
        st.write("""
        Derive new features to enrich the dataset:
        - **Date column** :calendar: For temporal analysis.
        - **Season column** :four_leaf_clover: To capture seasonal patterns.
        - **AQI column** :cloud: To aggregate pollutant levels into a single air quality index.
        - **AQI_Bucket** :green_heart: To categorize AQI into qualitative groups (Good, Moderate, etc.).
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
        st.header("View Processed Data :eyes:")
        st.write("""
        After applying all preprocessing steps, review your dataset to ensure changes were applied as intended. 
        Confirm that imputation, column removal, and new feature creation have enhanced the dataset's quality and usability.
        """)
        if st.button("Show Processed Data"):
            st.dataframe(data)
    else:
        st.error("Data could not be loaded.")


# -------------------------------------
# Page 3: Data Visualization
# -------------------------------------
def data_visualization():
    st.title("Data Visualization :art:")

    st.markdown("""
    **Goal:** Gain insights into data patterns and relationships through visual exploration. :crystal_ball:
    
    On this page, you can:
    - **Examine pollution levels across stations** :cityscape:
    - **Compare pollutant averages** :bar_chart:
    - **Explore AQI distribution** :rainbow:
    - **Analyze correlations between variables** :link:
    - **Visualize multi-dimensional relationships** :dna:
    
    Interactive charts make it easier to spot trends, seasonal effects, and correlations 
    that might not be clear from raw data tables. :sparkles:
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # Average Pollution Levels by Station
        if st.checkbox("Show Average Pollution Levels by Station :cityscape:"):
            st.write("""
            Discover which stations report higher pollutant concentrations on average. 
            Identifying pollution hotspots can inform targeted policy interventions and resource allocation.
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
        if st.checkbox("Show Average Concentration of Each Pollutant :thermometer:"):
            st.write("""
            Compare the average levels of each pollutant across the entire dataset. 
            Identifying the dominant pollutants can guide which factors to address first.
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
        if st.checkbox("Show AQI Distribution :cloud:"):
            st.write("""
            Examine how often the air quality falls into each AQI category. 
            Understanding the prevalence of 'Unhealthy' or 'Hazardous' conditions 
            can motivate policy changes or health advisories.
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
        if st.checkbox("Show Correlation Matrix Heatmap :link:"):
            st.write("""
            A correlation matrix reveals relationships between variables. 
            Strong correlations can indicate predictive power or redundancy, 
            guiding feature selection and dimensionality reduction.
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
        if st.checkbox("Show Parallel Coordinates Plot of Weather and AQI :dna:"):
            st.write("""
            Parallel coordinates let you visualize high-dimensional relationships on a single chart. 
            Observe how weather conditions align with AQI levels, providing insights into 
            the environmental drivers of poor air quality.
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
    st.title("Data Modeling and Evaluation :robot_face:")

    st.markdown("""
    **Goal:** Use the cleaned and explored dataset to build predictive models. :crystal_ball:
    
    On this page, you can:
    - **Select Features and Target** :dart:
    - **Apply Standardization** :balance_scale:
    - **Train-Test Split** :scissors:
    - **Train Models (Linear Regression, KNN)** :chart_with_upwards_trend:
    - **Evaluate Performance** :checkered_flag:
    - **Hyperparameter Tuning** :game_die:
    
    By building and comparing models, we learn how well various factors predict air quality. 
    Tuning parameters helps maximize model performance, giving more accurate insights 
    into the underlying dynamics. :rocket:
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # Feature Selection
        st.header("Feature Selection :dart:")
        st.write("""
        Choose which variables to include as features (X) and which variable to predict (y).
        Drawing on previous insights, select the most relevant, non-redundant features 
        to improve model accuracy and interpretability.
        """)
        independent_vars = st.multiselect("Select Independent Variables:", options=data.columns)
        dependent_var = st.selectbox("Select Dependent Variable:", options=data.columns)

        if independent_vars and dependent_var:
            X = data[independent_vars]
            y = data[dependent_var]

            # Data Standardization
            st.header("Data Standardization :balance_scale:")
            st.write("""
            Standardizing features can improve model performance, especially for algorithms 
            sensitive to feature scales. Apply standardization to ensure equal emphasis on all features.
            """)
            standardize = st.checkbox("Standardize Data using StandardScaler")

            if standardize:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            # Train-Test Split
            st.header("Train-Test Split :scissors:")
            st.write("""
            Splitting data into training and test sets allows unbiased evaluation of model performance 
            on unseen data, ensuring better generalization and reliability in predictions.
            """)
            test_size = st.slider("Select Test Size (as a percentage):", min_value=10, max_value=50, value=20, step=5)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            st.write(f"Training Set Size: {len(X_train)} rows")
            st.write(f"Testing Set Size: {len(X_test)} rows")

            # Model Selection and Evaluation
            st.header("Model Selection and Evaluation :chart_with_upwards_trend:")
            st.write("""
            Evaluate different models using metrics like MSE, RMSE, R², and MAE. 
            Compare performance to select the best approach or understand how factors drive predictions.
            """)

            # Linear Regression
            if st.checkbox("Linear Regression :straight_ruler:"):
                model_lr = LinearRegression()
                model_lr.fit(X_train, y_train)
                y_pred_lr = model_lr.predict(X_test)

                mse_lr = mean_squared_error(y_test, y_pred_lr)
                rmse_lr = np.sqrt(mse_lr)
                r2_lr = r2_score(y_test, y_pred_lr)
                mae_lr = mean_absolute_error(y_test, y_pred_lr)

                st.write("### Linear Regression Results")
                st.write(f"**MSE:** {mse_lr}")
                st.write(f"**RMSE:** {rmse_lr}")
                st.write(f"**R²:** {r2_lr}")
                st.write(f"**MAE:** {mae_lr}")

            # K-Nearest Neighbors
            if st.checkbox("K-Nearest Neighbors :woman_running:"):
                st.write("""
                Tune 'k' and choose how to weight neighbors. 
                KNN can capture complex patterns if the right parameters are found.
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
                st.write(f"**MSE:** {mse_knn}")
                st.write(f"**RMSE:** {rmse_knn}")
                st.write(f"**R²:** {r2_knn}")
                st.write(f"**MAE:** {mae_knn}")

                # Grid Search for Hyperparameter Tuning with user-defined K range
                if st.checkbox("Perform Grid Search for KNN :game_die:"):
                    st.write("""
                    Adjust the range of `n_neighbors` to explore different K values during hyperparameter tuning. 
                    Use the sliders below to select the start and end of the K range.
                    """)

                    start_k = st.slider("Start of K range", 1, 10, 1)
                    end_k = st.slider("End of K range", start_k + 1, 50, 20, help="Select an upper bound larger than the start.")

                    param_grid = {
                        'n_neighbors': range(start_k, end_k + 1),
                        'weights': ['uniform', 'distance']
                    }

                    st.write("Selected K range:", list(param_grid['n_neighbors']))

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
st.sidebar.title("Navigation :compass:")
page = st.sidebar.radio("Go to", ["Data Loading", "Data Preprocessing", "Data Visualization", "Data Modeling and Evaluation"])

st.sidebar.markdown("""
---
### About This App :information_source:

This application provides an end-to-end analysis pipeline for Beijing's air quality data:
1. **Data Loading:** :open_file_folder: Get an overview of the raw dataset.
2. **Data Preprocessing:** :broom: Clean and enhance the data for better quality.
3. **Data Visualization:** :art: Uncover patterns and insights through interactive charts.
4. **Data Modeling and Evaluation:** :robot_face: Build and assess predictive models.

Use the navigation above to explore each stage of the analysis. Enjoy discovering insights! :sparkles:
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
