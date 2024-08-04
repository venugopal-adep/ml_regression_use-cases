import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Set page config
st.set_page_config(page_title="Boston Housing Price Predictor", layout="wide", page_icon="🏘️")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #4e8df5;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0c43a3;
    }
    .stMarkdown {
        text-align: left;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('boston.csv')
    return data

data = load_data()

# Column explanations
column_explanations = {
    "CRIM": "Per capita crime rate by town",
    "ZN": "Proportion of residential land zoned for lots over 25,000 sq.ft.",
    "INDUS": "Proportion of non-retail business acres per town",
    "CHAS": "Charles River dummy variable (1 if tract bounds river; 0 otherwise)",
    "NOX": "Nitric oxides concentration (parts per 10 million)",
    "RM": "Average number of rooms per dwelling",
    "AGE": "Proportion of owner-occupied units built prior to 1940",
    "DIS": "Weighted distances to five Boston employment centres",
    "RAD": "Index of accessibility to radial highways",
    "TAX": "Full-value property-tax rate per $10,000",
    "PTRATIO": "Pupil-teacher ratio by town",
    "B": "1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town",
    "LSTAT": "% lower status of the population",
    "MEDV": "Median value of owner-occupied homes in $1000's (Target Variable)"
}

# Sidebar
st.sidebar.title("🏘️ Boston Housing Price Predictor")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Select Regression Algorithm",
    ["Linear Regression", "Ridge Regression", "Lasso Regression", 
     "Decision Tree", "Random Forest", "Support Vector Regression"]
)

# Main content
st.title("Boston Housing Price Predictor")
st.write("Predict the median value of owner-occupied homes based on various features.")

# Prepare the data
@st.cache_data
def prepare_data(data):
    # Separate features and target
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return X, y, preprocessor, numeric_features, categorical_features

X, y, preprocessor, numeric_features, categorical_features = prepare_data(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and prediction
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Support Vector Regression": SVR()
}

# Create a pipeline with preprocessor and model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', models[algorithm])
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📘 Learn", "📊 Data Analysis", "🧮 Model Performance", "🔮 Prediction", "🧠 Quiz"])

with tab1:
    st.header("Understanding Boston Housing Price Prediction")
    
    st.subheader("What is Boston Housing Price Prediction?")
    st.write("""
    Boston Housing Price Prediction is a classic machine learning problem where we try to estimate the median value of owner-occupied homes in Boston based on various features of the neighborhood and the house itself.
    """)
    
    st.subheader("Why is it important?")
    st.write("""
    1. **Real Estate Market Analysis**: Helps in understanding factors affecting housing prices.
    2. **Urban Planning**: Provides insights into how different factors influence property values.
    3. **Investment Decisions**: Aids investors in making informed decisions about real estate investments.
    4. **Policy Making**: Helps policymakers understand the impact of various factors on housing affordability.
    """)
    
    st.subheader("Key Factors in Predicting Housing Prices")
    st.write("""
    - **CRIM**: Crime rate in the area
    - **RM**: Average number of rooms per dwelling
    - **LSTAT**: Percentage of lower status of the population
    - **PTRATIO**: Pupil-teacher ratio by town
    - **NOX**: Nitric oxides concentration
    """)
    
    st.subheader("How Machine Learning Helps")
    st.write("""
    Machine learning algorithms can:
    1. Identify complex relationships between various features and housing prices.
    2. Handle large datasets with multiple variables efficiently.
    3. Provide insights into which factors most strongly influence housing prices.
    4. Make predictions on new, unseen data based on learned patterns.
    """)

with tab2:
    st.header("Data Analysis")
    
    # Dataset Explorer
    st.subheader("Dataset Explorer")
    st.dataframe(data.head())
    
    # Column Explanations
    st.subheader("Column Explanations")
    for col, explanation in column_explanations.items():
        st.write(f"**{col}**: {explanation}")
    
    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(data.describe())
    
    # Univariate Analysis
    st.subheader("Univariate Analysis")
    feature_univariate = st.selectbox("Select a feature for univariate analysis", X.columns)
    fig_univariate = px.histogram(data, x=feature_univariate, marginal="box")
    st.plotly_chart(fig_univariate)
    
    # Bivariate Analysis
    st.subheader("Bivariate Analysis")
    feature_bivariate = st.selectbox("Select a feature for bivariate analysis with MEDV", 
                                     [col for col in X.columns if col != 'MEDV'])
    fig_bivariate = px.scatter(data, x=feature_bivariate, y='MEDV', color="RAD", hover_data=['CRIM'])
    st.plotly_chart(fig_bivariate)
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = data.corr()
    fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(fig_corr)

with tab3:
    st.header("Model Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted Housing Price")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                 mode='lines', name='Ideal'))
        fig.update_layout(xaxis_title="Actual MEDV", yaxis_title="Predicted MEDV")
        st.plotly_chart(fig)
        
    with col2:
        st.subheader("Model Performance Metrics")
        metrics = {
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MSE": mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2 Score": r2_score(y_test, y_pred)
        }
        for metric, value in metrics.items():
            st.metric(metric, f"{value:.4f}")
        
    if algorithm in ["Decision Tree", "Random Forest"]:
        st.subheader("Feature Importance")
        importances = model.named_steps['regressor'].feature_importances_
        feature_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
        feature_imp = feature_imp.sort_values('importance', ascending=False).reset_index(drop=True)
        fig = px.bar(feature_imp, x='importance', y='feature', orientation='h')
        st.plotly_chart(fig)

with tab4:
    st.header("Make a Prediction")
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    with col1:
        for feature in X.columns[:len(X.columns)//2]:
            input_data[feature] = st.slider(f"{feature}", float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))
    
    with col2:
        for feature in X.columns[len(X.columns)//2:]:
            input_data[feature] = st.slider(f"{feature}", float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))

    if st.button("🏘️ Predict Housing Price", key="predict_button"):
        # Create a DataFrame with all features
        input_df = pd.DataFrame([input_data])
        
        prediction = model.predict(input_df)
        st.success(f"Predicted Median Value: ${prediction[0]:.2f}k")

with tab5:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What does MEDV represent in the Boston Housing dataset?",
            "options": ["Median value of owner-occupied homes", "Mean value of all homes", "Maximum value of rented homes"],
            "correct": 0
        },
        {
            "question": "Which of the following is NOT a feature in the Boston Housing dataset?",
            "options": ["Crime rate", "Number of rooms", "House age", "Pupil-teacher ratio"],
            "correct": 2
        },
        {
            "question": "Why is the Boston Housing dataset important in machine learning?",
            "options": ["It's a classic dataset for regression problems", "It only contains data about Boston", "It's the largest housing dataset available"],
            "correct": 0
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}")
        user_answer = st.radio(q["question"], q["options"], key=f"q{i}")
        if st.button(f"Check Answer {i+1}", key=f"check{i}"):
            if q["options"].index(user_answer) == q["correct"]:
                st.success("Correct!")
            else:
                st.error(f"Incorrect. The correct answer is: {q['options'][q['correct']]}")

# Run the Streamlit app
if __name__ == "__main__":
    st.sidebar.info("This app predicts Boston housing prices based on various features.")

    # Add some additional information or insights
    st.sidebar.subheader("Did you know?")
    st.sidebar.write("The Boston Housing dataset is a famous dataset in machine learning.")
    st.sidebar.write("It contains information collected by the U.S Census Service concerning housing in the area of Boston, MA.")

    # Add a footer
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: #FAFAFA;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    </style>
    <div class="footer">
        Developed by Venugopal Adep | Data source: https://www.kaggle.com/datasets/schirmerchad/bostonhoustingmlnd
    </div>
    """, unsafe_allow_html=True)
