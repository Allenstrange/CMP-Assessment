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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

# Configure the Streamlit page and set an icon
st.set_page_config(page_title="Multi-Page Streamlit App", page_icon=":bar_chart:")

@st.cache_data
def load_data():
    # Load the dataset from a GitHub repository and return it as a DataFrame.
    # Caching ensures that data loading is done only once to improve performance.
    url = "https://raw.githubusercontent.com/Allenstrange/CMP-Assessment/refs/heads/main/Air_Quality_Beijing.csv"
    return pd.read_csv(url)

# Store the loaded data in session state to persist across pages
if 'data' not in st.session_state:
    st.session_state['data'] = load_data()

# Sidebar: Basic dataset information
st.sidebar.title("Dataset Overview")
st.sidebar.write(
    "### Air Quality in Beijing 🌫️\n"
    "This dataset includes hourly readings of various pollutants (PM2.5, PM10, SO2, NO2, CO, O3) and weather conditions (TEMP, PRES, DEWP, RAIN, WSPM), along with location and time details. The goal is to understand pollution dynamics in relation to weather factors."
)


# -------------------------------------
# Page 1: Data Loading and Initial Inspection
# -------------------------------------
def data_loading():
    # This page focuses on the first stage of data exploration:
    # 1. Viewing raw samples of the dataset.
    # 2. Checking basic descriptive statistics.
    # 3. Identifying missing values to inform later cleaning steps.

    st.title("Data Loading 🛠️")
    st.write(
        "On this page, you can preview the dataset to understand its structure, "
        "view summary statistics for a quick quantitative overview, and examine missing values."
    )

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # Allow user to select how many rows of the data to preview
        num_rows = st.slider("Select Number of Rows to Preview", 1, 100, 10)
        st.write("### Data Preview 🔍:")
        st.write("Below are the first few rows of the dataset:")
        st.dataframe(data.head(num_rows))

        # Optionally display descriptive statistics
        st.write("### Descriptive Statistics 📊:")
        st.write("View summary statistics like mean, median, and standard deviation.")
        show_desc_table = st.checkbox("Show Descriptive Statistics Table")
        if show_desc_table:
            st.write(data.describe().T)

        # Optionally display missing value information
        st.write("### Missing Values ❓:")
        st.write("Examine how many values are missing to guide data cleaning decisions.")
        show_missing_table = st.checkbox("Show Missing Values Table")
        if show_missing_table:
            missing_values = data.isnull().sum()
            missing_percentage = (data.isnull().sum() / len(data)) * 100
            missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
            st.table(missing_df)
    else:
        st.write("Data could not be loaded.")


# -------------------------------------
# Page 2: Data Preprocessing
# -------------------------------------
def data_preprocessing():
    # This page deals with making the raw dataset more analyzable.
    # Steps include:
    # 1. Handling missing values via imputation.
    # 2. Dropping irrelevant or redundant columns.
    # 3. Preliminary data exploration (histograms) to understand distribution.
    # 4. Feature Engineering: Adding columns like Date, Season, AQI, and AQI_Bucket.

    st.title("Data Preprocessing 🧹")
    st.write(
        "Preprocess your data here by handling missing values, removing unnecessary columns, "
        "and creating new features. This step ensures your dataset is cleaner and richer "
        "for further analysis."
    )

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # 1. Handling Missing Values
        st.header("Handling Missing Values ❓")
        st.write(
            "Select a strategy for imputation (Mean, Median, Mode) and apply it to chosen columns. "
            "This helps ensure that missing data doesn't skew your analysis."
        )
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

        # 2. Dropping Columns
        st.header("Dropping Columns 🗑️")
        st.write(
            "You may have columns not needed for your analysis. "
            "Select columns to remove them from the dataset."
        )
        columns_to_drop = st.multiselect("Select columns to drop:", data.columns)

        if st.button("Drop Selected Columns"):
            data.drop(columns=columns_to_drop, inplace=True)
            st.success("Selected columns dropped successfully.")

        # Basic distribution exploration
        st.write("### Data Exploration 📈")
        st.write(
            "Visualize the distribution of selected features with histograms. "
            "This helps identify skewness, outliers, and data spread, guiding decisions on transformations."
        )
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
        st.header("Feature Engineering 🛠️")
        st.write(
            "Enhance your dataset by adding columns derived from existing data. "
            "This can create more meaningful variables for analysis and modeling."
        )

        # Add a Date column
        if st.checkbox("Add Date Column"):
            data['Date'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
            st.success("Date column added successfully.")

        # Add a Season column
        if st.checkbox("Add Season Column"):
            data['Season'] = data['month'].apply(
                lambda x: 'Spring' if 3 <= x <= 5 else 'Summer' if 6 <= x <= 8 else 'Autumn' if 9 <= x <= 11 else 'Winter'
            )
            st.success("Season column added successfully.")

        # Add an AQI column
        # This transforms multiple pollutant readings into a single Air Quality Index value.
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

        # Add an AQI_Bucket column
        # This assigns a qualitative category (Good, Moderate, Unhealthy, etc.) based on the AQI.
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

        # Show the processed data if requested
        st.header("Processed Data 📑")
        st.write("View the updated dataset with applied transformations and new features.")
        if st.button("Show Processed Data"):
            st.dataframe(data)
    else:
        st.write("Data could not be loaded.")


# -------------------------------------
# Page 3: Data Visualization
# -------------------------------------
def data_visualization():
    # This page focuses on visually exploring the relationships within the dataset.
    # Plots can reveal trends, outliers, and correlations that are not immediately visible
    # in raw numbers.

    st.title("Data Visualization 📊")
    st.write(
        "Use interactive charts to understand how pollutants correlate with each other and "
        "with weather conditions. Discover seasonal patterns, station-level differences, and "
        "the overall distribution of air quality."
    )

    if st.session_state['data'] is not None:
        data = st.session_state['data']
        st.write("### Visualizations 🖼️")

        # Show average pollution levels by station
        if st.checkbox("Show Average Pollution Levels by Station"):
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

        # Show average concentration of each pollutant globally
        if st.checkbox("Show Average Concentration of Each Pollutant"):
            pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
            mean_pollutants = data[pollutants].mean()
            fig = px.bar(
                x=pollutants,
                y=mean_pollutants,
                title='Average Concentration of Each Pollutant',
                labels={'x': 'Pollutant', 'y': 'Average Concentration'}
            )
            st.plotly_chart(fig)

        # Show distribution of AQI categories
        if st.checkbox("Show AQI Distribution"):
            fig = px.histogram(
                data,
                x='AQI_Bucket',
                nbins=30,
                title='AQI Distribution',
                marginal='box'
            )
            st.plotly_chart(fig)

        # Show correlation matrix heatmap
        if st.checkbox("Show Correlation Matrix Heatmap"):
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

        # Show parallel coordinates plot
        if st.checkbox("Show Parallel Coordinates Plot of Weather and AQI"):
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
        st.write("Data could not be loaded.")


# -------------------------------------
# Page 4: Data Modeling and Evaluation
# -------------------------------------
def data_modeling():
    # This page allows the user to build predictive models.
    # Steps include:
    # 1. Selecting target and feature variables.
    # 2. (Optionally) Standardizing data to ensure all features contribute equally.
    # 3. Splitting the dataset into training and testing sets for unbiased evaluation.
    # 4. Training models like Linear Regression and K-Nearest Neighbors.
    # 5. Evaluating model performance using metrics like MSE, RMSE, R², and MAE.
    # 6. Optionally running a Grid Search to optimize KNN hyperparameters.

    st.title("Data Modeling and Evaluation 🤖")
    st.write(
        "Build and evaluate predictive models using selected features. "
        "Assess model performance with various metrics and even tune parameters "
        "to achieve better results."
    )

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # Feature Selection
        st.header("Feature Selection 🔬")
        st.write("Choose which variables will be used as features (X) and which will be the target (y).")

        # Correlation heatmap to guide feature selection if desired
        if st.checkbox("Show Correlation Heatmap"):
            corr_cols = data.select_dtypes(include=[np.number]).columns
            corr_matrix = data[corr_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)

        independent_vars = st.multiselect("Select Independent Variables:", options=data.columns)
        dependent_var = st.selectbox("Select Dependent Variable:", options=data.columns)

        if independent_vars and dependent_var:
            X = data[independent_vars]
            y = data[dependent_var]

            # Data Standardization Option
            st.header("Data Standardization ⚖️")
            st.write(
                "Standardization transforms features to have a mean of 0 and a standard deviation of 1, "
                "which can improve model performance and stability."
            )
            standardize = st.checkbox("Standardize Data using StandardScaler")

            if standardize:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            # Train-Test Split
            st.header("Train-Test Split ✂️")
            st.write(
                "Splitting your data into training and testing sets allows you to evaluate "
                "model performance on unseen data, ensuring the model generalizes well."
            )
            test_size = st.slider("Select Test Size (as a percentage):", min_value=10, max_value=50, value=20, step=5)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            st.write(f"Training Set Size: {len(X_train)} rows")
            st.write(f"Testing Set Size: {len(X_test)} rows")

            # Model Selection
            st.header("Model Selection 🛠️")
            st.write("Train different models and evaluate their performance:")

            # Linear Regression Model
            if st.checkbox("Linear Regression"):
                model_lr = LinearRegression()
                model_lr.fit(X_train, y_train)
                y_pred_lr = model_lr.predict(X_test)

                mse_lr = mean_squared_error(y_test, y_pred_lr)
                rmse_lr = np.sqrt(mse_lr)
                r2_lr = r2_score(y_test, y_pred_lr)
                mae_lr = mean_absolute_error(y_test, y_pred_lr)

                st.write("### Linear Regression Results")
                st.write(f"Mean Squared Error (MSE): {mse_lr}")
                st.write(f"Root Mean Squared Error (RMSE): {rmse_lr}")
                st.write(f"R-squared (R²): {r2_lr}")
                st.write(f"Mean Absolute Error (MAE): {mae_lr}")

            # K-Nearest Neighbors Model
            if st.checkbox("K-Nearest Neighbors"):
                st.write(
                    "Adjust the number of neighbors (k) and weighting scheme to find "
                    "a good fit for your data."
                )
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
                st.write(f"Mean Squared Error (MSE): {mse_knn}")
                st.write(f"Root Mean Squared Error (RMSE): {rmse_knn}")
                st.write(f"R-squared (R²): {r2_knn}")
                st.write(f"Mean Absolute Error (MAE): {mae_knn}")

                # Optional Grid Search for KNN
                if st.checkbox("Perform Grid Search for KNN"):
                    st.write(
                        "Grid search automates hyperparameter tuning by testing multiple combinations "
                        "of k and weight settings. The best combination is displayed below."
                    )
                    param_grid = {
                        'n_neighbors': range(1, 21),
                        'weights': ['uniform', 'distance']
                    }
                    grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
                    grid_search.fit(X_train, y_train)

                    st.write("### Best Parameters from Grid Search:")
                    st.write(grid_search.best_params_)
                    st.write(f"Best Cross-Validated Score (MSE): {-grid_search.best_score_}")

    else:
        st.write("Data could not be loaded.")


# -------------------------------------
# Sidebar Navigation
# -------------------------------------
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
