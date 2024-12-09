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

st.set_page_config(page_title="Multi-Page Streamlit App", page_icon=":bar_chart:")

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

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # Data preview
        num_rows = st.slider("Select Number of Rows to Preview", 1, 100, 10)
        st.write("### Data Preview:")
        st.write("This is a dataset about pollutants")
        st.dataframe(data.head(num_rows))

        # Descriptive Statistics Options
        st.write("### Descriptive Statistics:")
        show_desc_table = st.checkbox("Show Descriptive Statistics Table")
        if show_desc_table:
            st.write(data.describe().T)

        # Missing Value Statistics Options
        st.write("### Missing Values:")
        show_missing_table = st.checkbox("Show Missing Values Table")
        if show_missing_table:
            # Calculate missing values and percentages
            missing_values = data.isnull().sum()
            missing_percentage = (data.isnull().sum() / len(data)) * 100
            # Create a DataFrame for display
            missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
            st.table(missing_df)  # Display as a table

    else:
        st.write("Data could not be loaded.")

# Page 2: Data Preprocessing
def data_preprocessing():
    st.title("Data Preprocessing")

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
        if st.button("Show Processed Data"):
            st.dataframe(data)

    else:
        st.write("Data could not be loaded.")

# Page 3: Data Visualization
def data_visualization():
    st.title("Data Visualization")

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        st.write("### Visualizations")

        # Visualization 3: Stacked Bar Chart of Average Pollution Levels by Station
        if st.checkbox("Show Average Pollution Levels by Station"):
            station_stats = data.groupby('station')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean().reset_index()
            station_stats_melted = pd.melt(station_stats, id_vars=['station'], value_vars=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'], var_name='Pollutant', value_name='Average Concentration')
            fig = px.bar(station_stats_melted, x='station', y='Average Concentration', color='Pollutant', barmode='stack', title='Average Pollution Levels by Station')
            st.plotly_chart(fig)

        # Visualization 4: Bar Chart of Average Concentration of Each Pollutant
        if st.checkbox("Show Average Concentration of Each Pollutant"):
            pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
            mean_pollutants = data[pollutants].mean()
            fig = px.bar(x=pollutants, y=mean_pollutants, title='Average Concentration of Each Pollutant', labels={'x': 'Pollutant', 'y': 'Average Concentration'})
            st.plotly_chart(fig)

        # Visualization 5: Histogram of AQI Distribution
        if st.checkbox("Show AQI Distribution"):
            fig = px.histogram(data, x='AQI_Bucket', nbins=30, title='AQI Distribution', marginal='box')
            st.plotly_chart(fig)

        # Visualization 6: Heatmap of Correlation Matrix
        if st.checkbox("Show Correlation Matrix Heatmap"):
            corr_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'AQI']
            corr_matrix = data[corr_cols].corr()
            text_annotations = np.around(corr_matrix.values, decimals=2)
            fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_cols, y=corr_cols, colorscale='Viridis', text=text_annotations, texttemplate="%{text}"))
            fig.update_layout(title='Correlation Matrix of Weather Conditions and Pollutants/AQI')
            st.plotly_chart(fig)

        # Visualization 7: Stacked Bar Chart of AQI Distribution by Station
        if st.checkbox("Show AQI Distribution by Station"):
            fig = px.histogram(data, x='station', color='AQI_Bucket', title='AQI Distribution by Station', barmode='stack', category_orders={'AQI_Bucket': ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']})
            st.plotly_chart(fig)

        # Visualization 8: Parallel Coordinates Plot of Weather and AQI
        if st.checkbox("Show Parallel Coordinates Plot of Weather and AQI"):
            AQI_Bucket_mapping = {'Good': 1, 'Moderate': 2, 'Unhealthy for Sensitive Groups': 3, 'Unhealthy': 4, 'Very Unhealthy': 5, 'Hazardous': 6}
            data['AQI_Bucket_Num'] = data['AQI_Bucket'].map(AQI_Bucket_mapping)
            fig = px.parallel_coordinates(data, dimensions=['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'AQI'], color='AQI_Bucket_Num', color_continuous_scale=px.colors.diverging.Tealrose, title='Parallel Coordinates Plot of Weather and AQI')
            fig.update_layout(coloraxis_colorbar=dict(tickvals=list(AQI_Bucket_mapping.values()), ticktext=list(AQI_Bucket_mapping.keys())))
            st.plotly_chart(fig)

    else:
        st.write("Data could not be loaded.")
# Page 4: Data Modeling and Evaluation
def data_modeling():
    st.title("Data Modeling and Evaluation")

    if st.session_state['data'] is not None:
        data = st.session_state['data']

        # Feature Selection Section
        st.header("Feature Selection")
        st.write("Select dependent and independent variables for modeling.")

        # Correlation Heatmap
        if st.checkbox("Show Correlation Heatmap"):
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
            standardize = st.checkbox("Standardize Data using StandardScaler")

            if standardize:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            # Train-Test Split
            st.header("Train-Test Split")
            test_size = st.slider("Select Test Size (as a percentage):", min_value=10, max_value=50, value=20, step=5)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            st.write("Training Set Size:", len(X_train))
            st.write("Testing Set Size:", len(X_test))

            # Model Selection Section
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

                # Grid Search CV for KNN
                if st.checkbox("Perform Grid Search CV for KNN"):
                    param_grid = {
                        'n_neighbors': range(1, st.slider("Select Max K Value for Grid Search:", min_value=5, max_value=50, value=20, step=5)),
                        'weights': ["uniform", "distance"]
                    }
                    grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5)
                    grid_search.fit(X_train, y_train)

                    st.write("Best Parameters:", grid_search.best_params_)
                    st.write("Best Score:", grid_search.best_score_)

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
