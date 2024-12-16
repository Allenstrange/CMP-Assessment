import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# Function to load data
def load_data():
    st.title("Assessment App")  # App title
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Ensure column types are clean for serialization
            for col in df.select_dtypes(include=['object']):
                df[col] = df[col].astype(str)
            st.session_state["data"] = df  # Save to session state
            st.success("File uploaded and loaded successfully!")
            return True
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return False
    return False

# Load data if not already in session state
if "data" not in st.session_state or st.session_state["data"] is None:
    file_loaded = load_data()
else:
    file_loaded = True

# Check if data is loaded
if file_loaded and "data" in st.session_state:
    data = st.session_state["data"]

    # Tabs for EDA, Visualization, and Modeling
    tab1, tab2, tab3 = st.tabs(["Basic EDA", "Visualization", "Modeling"])

    # Tab 1: Basic EDA
    with tab1:
        st.header("Basic EDA")
        selected_EDA = st.multiselect(
            "Select EDA operations",
            ["Basic Statistics", "Missing Values", "Data Description", "Missing Values Percentage"]
        )

        if "Basic Statistics" in selected_EDA:
            st.subheader("Basic Statistics")
            st.write(data.describe())

        if "Missing Values" in selected_EDA:
            st.subheader("Missing Values")
            st.write(data.isnull().sum())

        if "Data Description" in selected_EDA:
            st.subheader("Data Description")
            st.write(data.describe(include="all"))

        if "Missing Values Percentage" in selected_EDA:
            st.subheader("Missing Values Percentage")
            st.write(data.isnull().mean() * 100)

    # Tab 2: Visualization
    with tab2:
        st.header("Visualization")
        columns = data.columns.tolist()
        chart_type = st.selectbox("Select Chart Type", ["Scatter Plot", "Bar Chart", "Line Chart", "Word Cloud"])

        if chart_type in ["Scatter Plot", "Bar Chart", "Line Chart"]:
            x_axis = st.selectbox("Select X-axis", columns)
            y_axis = st.selectbox("Select Y-axis", columns)
            if x_axis and y_axis:
                if pd.api.types.is_numeric_dtype(data[x_axis]) and pd.api.types.is_numeric_dtype(data[y_axis]):
                    if chart_type == "Scatter Plot":
                        fig = px.scatter(data, x=x_axis, y=y_axis, title=f"{chart_type} of {x_axis} vs {y_axis}")
                    elif chart_type == "Bar Chart":
                        fig = px.bar(data, x=x_axis, y=y_axis, title=f"{chart_type} of {x_axis} vs {y_axis}")
                    elif chart_type == "Line Chart":
                        fig = px.line(data, x=x_axis, y=y_axis, title=f"{chart_type} of {x_axis} vs {y_axis}")
                    st.plotly_chart(fig)
                else:
                    st.error("Both X-axis and Y-axis columns must contain numeric data.")

        elif chart_type == "Word Cloud":
            text_column = st.selectbox("Select Text Column", columns)
            if text_column:
                try:
                    text_data = " ".join(data[text_column].astype(str).dropna())
                    wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white").generate(text_data)
                    plt.figure(figsize=(10, 6))
                    plt.imshow(wordcloud, interpolation="bilinear")
                    plt.axis("off")
                    st.pyplot(plt)
                except Exception as e:
                    st.error(f"Error generating Word Cloud: {e}")

    # Tab 3: Modeling
    with tab3:
        st.header("Modeling")
        target = st.selectbox("Select Target Variable", data.columns)
        features = st.multiselect("Select Feature Variables", [col for col in data.columns if col != target])

        if target and features:
            X = data[features]
            y = data[target]

            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())

            # Train-Test Split
            test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Model Selection
            model_type = st.selectbox(
                "Select Model",
                ["Linear Regression", "Decision Tree", "Random Forest", "SVR", "KNN"]
            )

            try:
                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Decision Tree":
                    model = DecisionTreeRegressor()
                elif model_type == "Random Forest":
                    model = RandomForestRegressor()
                elif model_type == "SVR":
                    model = SVR()
                elif model_type == "KNN":
                    model = KNeighborsRegressor()

                # Fit and Predict
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Metrics
                mse = mean_squared_error(y_test, predictions)
                st.subheader("Model Performance")
                st.write(f"Mean Squared Error: {mse:.2f}")
            except Exception as e:
                st.error(f"Error in modeling: {e}")
else:
    st.warning("Please upload a CSV file to proceed.")
