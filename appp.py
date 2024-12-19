import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Set the Streamlit page configuration
st.set_page_config(page_title="Air Quality Analysis App", page_icon=":bar_chart:")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Allenstrange/CMP-Assessment/refs/heads/main/Air_Quality_Beijing.csv"
    return pd.read_csv(url)

# Load data into session state
if 'data' not in st.session_state:
    st.session_state['data'] = load_data()

# -------------------------------------
# Page 1: Data Loading
# -------------------------------------
def data_loading():
    st.title("Data Loading :open_file_folder:")

    st.markdown("""
    **Goal:** Before we dive into analysis, let’s get acquainted with the dataset.
    
    The Beijing Air Quality dataset contains hourly readings of pollutants (PM2.5, PM10, SO2, NO2, CO, O3) 
    and weather conditions (TEMP, PRES, DEWP, RAIN, WSPM) across multiple stations.
    
    On this page, you can:
    - **Preview raw data** :mag: 
    - **View descriptive statistics** :bar_chart:
    - **Check for missing values** :grey_question:
    
    This initial overview helps identify data structure and quality issues, setting a foundation for further steps. :seedling:
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data']
        
        # Data preview
        st.header("Data Preview :eyes:")
        num_rows = st.slider("Select Number of Rows to Preview", 1, 100, 10)
        st.dataframe(data.head(num_rows))

        # Descriptive Statistics
        st.header("Descriptive Statistics :clipboard:")
        show_desc_table = st.checkbox("Show Descriptive Statistics Table")
        if show_desc_table:
            st.write(data.describe().T)

        # Missing Values
        st.header("Missing Values Analysis :question:")
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
    
    Proper preprocessing ensures the dataset is ready for deeper analysis and modeling. :rocket:
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data'].copy()

        # Handling Missing Values
        st.header("Handling Missing Values :umbrella:")
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
        columns_to_drop = st.multiselect("Select columns to drop:", data.columns)
        if st.button("Drop Selected Columns"):
            data.drop(columns=columns_to_drop, inplace=True)
            st.success("Selected columns dropped successfully.")
            st.session_state['data'] = data

        # Data Distribution Exploration
        st.header("Distribution Exploration :bar_chart:")
        selected_columns = st.multiselect("Select columns to display histograms",
                                          data.columns,
                                          default=['PM2.5', 'PM10', 'TEMP', 'PRES'])
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
        if st.checkbox("Add Date Column"):
            data['Date'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
            st.success("Date column added successfully.")
            st.session_state['data'] = data

        if st.checkbox("Add Season Column"):
            data['Season'] = data['month'].apply(
                lambda x: 'Spring' if 3 <= x <= 5 else 
                          'Summer' if 6 <= x <= 8 else 
                          'Autumn' if 9 <= x <= 11 else 
                          'Winter'
            )
            st.success("Season column added successfully.")
            st.session_state['data'] = data

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

        st.header("View Processed Data :eyes:")
        if st.button("Show Processed Data"):
            st.dataframe(data)
    else:
        st.error("Data could not be loaded.")

# -------------------------------------
# Page 3: Data Visualization (Updated with Key Findings)
# -------------------------------------
def data_visualization():
    st.title("Data Visualization :art:")

    st.markdown("""
    **Goal:** Gain insights into data patterns and relationships through visual exploration. :crystal_ball:
    
    On this page, you can:
    - **Examine pollution levels across stations** :cityscape:
    - **Compare pollutant averages** :bar_chart:
    - **Explore AQI distribution** :rainbow:
    - **Analyze correlations** :link:
    - **Visualize multi-dimensional relationships** :dna:
    
    Interactive charts help reveal trends, seasonal effects, and correlations. :sparkles:
    """)

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # Average Pollution Levels by Station
        if st.checkbox("Show Average Pollution Levels by Station :cityscape:"):
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

            st.markdown("""
            **Key Finding:** Some stations consistently show higher pollutant levels, indicating localized pollution sources 
            or environmental conditions. This suggests that targeted interventions at specific locations could be more effective.
            """)

        # Average Concentration of Each Pollutant
        if st.checkbox("Show Average Concentration of Each Pollutant :thermometer:"):
            pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
            mean_pollutants = data[pollutants].mean()
            fig = px.bar(
                x=pollutants,
                y=mean_pollutants,
                title='Average Concentration of Each Pollutant',
                labels={'x': 'Pollutant', 'y': 'Average Concentration'}
            )
            st.plotly_chart(fig)

            st.markdown("""
            **Key Finding:** Certain pollutants stand out with higher average concentrations. Identifying these primary pollutants 
            can help focus mitigation efforts where they will have the greatest impact on overall air quality.
            """)

        # AQI Distribution
        if st.checkbox("Show AQI Distribution :cloud:"):
            fig = px.histogram(
                data,
                x='AQI_Bucket',
                nbins=30,
                title='AQI Distribution',
                marginal='box'
            )
            st.plotly_chart(fig)

            st.markdown("""
            **Key Finding:** Most readings fall into 'Good' or 'Moderate' categories, indicating that severe pollution events 
            are relatively infrequent. This suggests the region typically experiences mild to moderate pollution rather than persistent extremes.
            """)

        # Correlation Matrix Heatmap
        if st.checkbox("Show Correlation Matrix Heatmap :link:"):
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

            st.markdown("""
            **Key Finding:** Certain pollutants are closely correlated with each other and with weather conditions. 
            For instance, PM2.5 and PM10 often rise together, and weather factors like temperature or pressure may 
            influence pollutant levels. Understanding these relationships helps target the root causes of pollution.
            """)

        # Parallel Coordinates Plot
        if st.checkbox("Show Parallel Coordinates Plot of Weather and AQI :dna:"):
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

            st.markdown("""
            **Key Finding:** Weather conditions and AQI levels are interconnected. Certain weather patterns are associated 
            with better air quality, while others coincide with poorer AQI. This suggests a strategic focus on managing emissions 
            under specific weather conditions could yield better results.
            """)

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
    - **Choose from Multiple Models (LR, KNN, Decision Tree, Random Forest)** :chart_with_upwards_trend:
    - **Use Cross-Validation** :scissors: for more reliable performance estimates.
    - **Perform Hyperparameter Tuning** :game_die: with flexible scoring options.
    - **View Metrics and Plots** :bar_chart: for deeper model evaluation.
    """)

    if st.session_state['data'] is None:
        st.error("Data could not be loaded. Please check the Data Loading page.")
        return

    data = st.session_state['data']

    # Feature Selection
    st.header("Feature Selection :dart:")
    independent_vars = st.multiselect("Select Independent Variables:", options=data.columns)
    dependent_var = st.selectbox("Select Dependent Variable:", options=data.columns)
    if not independent_vars or not dependent_var:
        st.warning("Please select both independent and dependent variables.")
        return

    X = data[independent_vars]
    y = data[dependent_var]

    # Data Standardization
    st.header("Data Standardization :balance_scale:")
    standardize = st.checkbox("Standardize Data using StandardScaler", help="Standardization may improve model performance.")
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = X.values

    # Train-Test Split or Cross-Validation
    st.header("Train-Test or Cross-Validation Setup :scissors:")
    use_cv = st.checkbox("Use k-fold Cross-Validation", help="Provides more stable performance estimates.")
    if use_cv:
        k_folds = st.slider("Number of folds (k)", 2, 10, 5)
    else:
        test_size = st.slider("Test Size (as %):", 10, 50, 20, step=5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    # Model Selection
    st.header("Model Selection :gear:")
    model_choice = st.radio("Select a Model:", ["Linear Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest"], 
                            help="Choose one model to train and evaluate.")

    # Hyperparameter Tuning Controls
    st.header("Hyperparameter Tuning :game_die:")
    scoring_metric = st.selectbox("Scoring Metric for Grid Search:", ["neg_mean_squared_error", "r2", "neg_mean_absolute_error"], 
                                  help="The metric that GridSearchCV will optimize.")
    
    param_grid = {}
    model = None

    if model_choice == "Linear Regression":
        model = LinearRegression()
        st.write("No hyperparameters available for Linear Regression.")
    elif model_choice == "K-Nearest Neighbors":
        max_k = st.slider("Max K for Grid Search:", 2, 50, 20, help="Tests all k from 1 to max_k")
        param_grid = {'n_neighbors': range(1, max_k + 1), 'weights': ['uniform', 'distance']}
        model = KNeighborsRegressor()
    elif model_choice == "Decision Tree":
        max_depth = st.number_input("Max Depth (for Grid Search):", 1, 50, 10)
        param_grid = {'max_depth': range(1, max_depth + 1)}
        model = DecisionTreeRegressor(random_state=42)
    elif model_choice == "Random Forest":
        max_depth = st.number_input("Max Depth (for Grid Search):", 1, 50, 10)
        n_estimators = st.slider("Max n_estimators for Grid Search:", 50, 500, 100, step=50, 
                                 help="Number of trees to test in the forest.")
        param_grid = {
            'max_depth': range(1, max_depth + 1),
            'n_estimators': range(50, n_estimators + 1, 50)
        }
        model = RandomForestRegressor(random_state=42)

    perform_grid_search = st.checkbox("Perform Grid Search")
    if perform_grid_search and param_grid:
        gs = GridSearchCV(model, param_grid, cv=3, scoring=scoring_metric)
        if use_cv:
            gs.fit(X, y)
        else:
            gs.fit(X_train, y_train)

        best_model = gs.best_estimator_
        st.write("### Best Parameters from Grid Search")
        st.write(gs.best_params_)

        if use_cv:
            cv_scores_mse = cross_val_score(best_model, X, y, cv=k_folds, scoring='neg_mean_squared_error')
            cv_scores_mae = cross_val_score(best_model, X, y, cv=k_folds, scoring='neg_mean_absolute_error')
            cv_scores_r2 = cross_val_score(best_model, X, y, cv=k_folds, scoring='r2')

            results_df = pd.DataFrame({
                "Metric": ["MSE", "MAE", "R²"],
                "Mean Score": [-np.mean(cv_scores_mse), -np.mean(cv_scores_mae), np.mean(cv_scores_r2)],
                "Std Dev": [np.std(cv_scores_mse), np.std(cv_scores_mae), np.std(cv_scores_r2)]
            })
            st.write("### Cross-Validation Results for Best Model")
            st.table(results_df)
            best_model.fit(X, y)
            model = best_model
            y_pred = best_model.predict(X)
            y_true = y
        else:
            y_pred = best_model.predict(X_test)
            mse_val = mean_squared_error(y_test, y_pred)
            mae_val = mean_absolute_error(y_test, y_pred)
            r2_val = r2_score(y_test, y_pred)
            rmse_val = np.sqrt(mse_val)

            metrics_df = pd.DataFrame({
                "Metric": ["MSE", "RMSE", "MAE", "R²"],
                "Value": [mse_val, rmse_val, mae_val, r2_val]
            })
            st.write("### Test Set Results for Best Model")
            st.table(metrics_df)
            model = best_model
            y_true = y_test
    else:
        # No Grid Search
        if use_cv:
            cv_scores_mse = cross_val_score(model, X, y, cv=k_folds, scoring='neg_mean_squared_error')
            cv_scores_mae = cross_val_score(model, X, y, cv=k_folds, scoring='neg_mean_absolute_error')
            cv_scores_r2 = cross_val_score(model, X, y, cv=k_folds, scoring='r2')

            results_df = pd.DataFrame({
                "Metric": ["MSE", "MAE", "R²"],
                "Mean Score": [-np.mean(cv_scores_mse), -np.mean(cv_scores_mae), np.mean(cv_scores_r2)],
                "Std Dev": [np.std(cv_scores_mse), np.std(cv_scores_mae), np.std(cv_scores_r2)]
            })
            st.write("### Cross-Validation Results")
            st.table(results_df)
            model.fit(X, y)
            y_pred = model.predict(X)
            y_true = y
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse_val = mean_squared_error(y_test, y_pred)
            mae_val = mean_absolute_error(y_test, y_pred)
            r2_val = r2_score(y_test, y_pred)
            rmse_val = np.sqrt(mse_val)

            metrics_df = pd.DataFrame({
                "Metric": ["MSE", "RMSE", "MAE", "R²"],
                "Value": [mse_val, rmse_val, mae_val, r2_val]
            })
            st.write("### Test Set Results")
            st.table(metrics_df)
            y_true = y_test

    # Visualization of Model Performance
    if model is not None:
        st.header("Visualization of Model Performance :bar_chart:")
        residuals = y_true - y_pred

        # Residual Distribution
        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_title("Residual Distribution")
        st.pyplot(fig)

        # Prediction vs Actual Plot
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Prediction vs. Actual")
        st.pyplot(fig)

    # Model Persistence and Comparison
    if model is not None and st.checkbox("Save Model for Comparison"):
        if 'saved_models' not in st.session_state:
            st.session_state['saved_models'] = []
        st.session_state['saved_models'].append((model_choice, model))
        st.success(f"{model_choice} model saved for comparison.")

    if 'saved_models' in st.session_state and st.session_state['saved_models']:
        st.write("### Saved Models for Comparison")
        for idx, (name, m) in enumerate(st.session_state['saved_models']):
            st.write(f"{idx+1}. {name}")
        # Additional comparison logic can be implemented here if desired.

# -------------------------------------
# Sidebar Navigation
# -------------------------------------
st.sidebar.title("Navigation :compass:")
page = st.sidebar.radio("Go to", ["Data Loading", "Data Preprocessing", "Data Visualization", "Data Modeling and Evaluation"])

st.sidebar.markdown("""
---
### About This App :information_source:

This application provides an end-to-end analysis pipeline for Beijing's air quality data:
1. **Data Loading:** :open_file_folder: Explore the raw dataset.
2. **Data Preprocessing:** :broom: Clean and enhance the dataset.
3. **Data Visualization:** :art: Gain insights through interactive charts.
4. **Data Modeling and Evaluation:** :robot_face: Build, tune, and evaluate predictive models.

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
