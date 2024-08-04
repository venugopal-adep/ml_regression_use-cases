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

# Set page config
st.set_page_config(page_title="E-commerce Customer Spending Predictor", layout="wide", page_icon="🛒")

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
    data = pd.read_csv('EcommerceCustomers.csv')
    return data

data = load_data()

# Column explanations
column_explanations = {
    "Email": "Customer's email address",
    "Address": "Customer's physical address",
    "Avatar": "Color of the customer's avatar",
    "Avg. Session Length": "Average length of in-store style advice sessions",
    "Time on App": "Average time spent on App in minutes",
    "Time on Website": "Average time spent on Website in minutes",
    "Length of Membership": "Number of years the customer has been a member",
    "Yearly Amount Spent": "Amount spent by the customer per year (target variable)"
}

# Sidebar
st.sidebar.title("🛒 E-commerce Predictor")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Select Regression Algorithm",
    ["Linear Regression", "Ridge Regression", "Lasso Regression", 
     "Decision Tree", "Random Forest", "Support Vector Regression"]
)

# Main content
st.title("E-commerce Customer Spending Predictor")
st.write("Predict yearly spending of e-commerce customers based on their behavior.")

# Prepare the data
X = data.drop(['Yearly Amount Spent', 'Email', 'Address', 'Avatar'], axis=1)
y = data['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training and prediction
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Support Vector Regression": SVR()
}

model = models[algorithm]
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "📊 Data Analysis", "🧮 Model Performance", "🔮 Prediction", "🧠 Quiz"])

with tab1:
    st.header("Understanding E-commerce Customer Spending Prediction")
    
    st.subheader("What is E-commerce Customer Spending Prediction?")
    st.write("""
    E-commerce customer spending prediction is a technique used to estimate how much a customer is likely to spend on an online platform based on various factors such as their behavior, engagement, and historical data.
    """)
    
    st.subheader("Why is it important?")
    st.write("""
    1. **Personalized Marketing**: Tailor marketing strategies to different customer segments.
    2. **Inventory Management**: Anticipate demand and manage stock levels effectively.
    3. **Customer Retention**: Identify high-value customers and implement retention strategies.
    4. **Revenue Forecasting**: Make more accurate business projections and financial plans.
    """)
    
    st.subheader("Key Factors in Predicting Customer Spending")
    st.write("""
    - **Avg. Session Length**: Longer sessions may indicate higher engagement and potential spending.
    - **Time on App**: More time spent on the app could lead to more purchases.
    - **Time on Website**: Similar to app usage, website engagement can influence spending.
    - **Length of Membership**: Long-term customers might have different spending patterns.
    """)
    
    st.subheader("How Machine Learning Helps")
    st.write("""
    Machine learning algorithms can:
    1. Identify complex patterns in customer behavior.
    2. Handle large amounts of data efficiently.
    3. Continuously improve predictions as new data becomes available.
    4. Provide insights into which factors most influence customer spending.
    """)

with tab2:
    st.header("Data Analysis")
    
    # Dataset Explorer
    st.subheader("Dataset Explorer")
    columns_to_show = st.multiselect("Select columns to display", list(data.columns), default=list(data.columns[:5]))
    st.dataframe(data[columns_to_show])
    
    # Column Explanations
    st.subheader("Column Explanations")
    for col, explanation in column_explanations.items():
        st.write(f"**{col}**: {explanation}")
    
    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(data.describe())
    
    # Univariate Analysis
    st.subheader("Univariate Analysis")
    feature_univariate = st.selectbox("Select a feature for univariate analysis", list(X.columns) + ['Yearly Amount Spent'])
    fig_univariate = px.histogram(data, x=feature_univariate, marginal="box")
    st.plotly_chart(fig_univariate)
    
    # Bivariate Analysis
    st.subheader("Bivariate Analysis")
    feature_bivariate = st.selectbox("Select a feature for bivariate analysis with Yearly Amount Spent", list(X.columns))
    fig_bivariate = px.scatter(data, x=feature_bivariate, y='Yearly Amount Spent', color="Avatar", hover_data=['Email'])
    st.plotly_chart(fig_bivariate)
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = data[list(X.columns) + ['Yearly Amount Spent']].corr()
    fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(fig_corr)

with tab3:
    st.header("Model Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted Spending")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                 mode='lines', name='Ideal'))
        fig.update_layout(xaxis_title="Actual Yearly Spending", yaxis_title="Predicted Yearly Spending")
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
            st.metric(metric, f"{value:.2f}")
        
    if algorithm in ["Decision Tree", "Random Forest"]:
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        feature_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
        feature_imp = feature_imp.sort_values('importance', ascending=False).reset_index(drop=True)
        fig = px.bar(feature_imp, x='importance', y='feature', orientation='h')
        st.plotly_chart(fig)

with tab4:
    st.header("Make a Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        avg_session_length = st.slider("Average Session Length", min_value=0.0, max_value=40.0, value=30.0)
        time_on_app = st.slider("Time on App (minutes)", min_value=0.0, max_value=15.0, value=10.0)
    
    with col2:
        time_on_website = st.slider("Time on Website (minutes)", min_value=0.0, max_value=45.0, value=35.0)
        length_of_membership = st.slider("Length of Membership (years)", min_value=0.0, max_value=6.0, value=3.0)

    if st.button("💰 Predict Yearly Spending"):
        input_data = pd.DataFrame({
            'Avg. Session Length': [avg_session_length],
            'Time on App': [time_on_app],
            'Time on Website': [time_on_website],
            'Length of Membership': [length_of_membership]
        })
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Yearly Spending: ${prediction[0]:,.2f}")

with tab5:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main goal of e-commerce customer spending prediction?",
            "options": ["To increase website traffic", "To estimate how much a customer is likely to spend", "To design new products"],
            "correct": 1
        },
        {
            "question": "Which of the following is NOT a key factor in predicting customer spending in this model?",
            "options": ["Average Session Length", "Time on App", "Customer's Age", "Length of Membership"],
            "correct": 2
        },
        {
            "question": "Why is customer spending prediction important for e-commerce businesses?",
            "options": ["It helps in personalizing marketing strategies", "It's not important for e-commerce", "It directly increases sales"],
            "correct": 0
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}")
        user_answer = st.radio(q["question"], q["options"])
        if st.button(f"Check Answer {i+1}"):
            if q["options"].index(user_answer) == q["correct"]:
                st.success("Correct!")
            else:
                st.error(f"Incorrect. The correct answer is: {q['options'][q['correct']]}")

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
    Developed by Your Name | Data source: Your Data Source
</div>
""", unsafe_allow_html=True)
