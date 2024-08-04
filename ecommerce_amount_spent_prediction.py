import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set page config
st.set_page_config(page_title="E-commerce Spending Predictor", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better appearance
st.markdown("""
<style>
.stApp {
    background-color: #f0f8ff;
}
.stButton>button {
    background-color: #4b0082;
    color: white;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #e6e6fa;
    border-radius: 4px 4px 0 0;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #8a2be2;
    color: white;
}
.highlight {
    background-color: #ffd700;
    padding: 5px;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🛒 E-commerce Customer Spending Predictor")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Predict customer spending based on their behavior and engagement!")

# Helper functions
@st.cache_data
def load_data():
    data = pd.read_csv('EcommerceCustomers.csv')
    return data

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, y_pred, mse, mae, r2

def plot_feature_importance(feature_names, feature_importance):
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    fig = px.bar(feature_importance_df, x='feature', y='importance', title='Feature Importance')
    return fig

def plot_actual_vs_predicted(y_test, y_pred):
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Spending', 'y': 'Predicted Spending'})
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                             mode='lines', name='Ideal'))
    fig.update_layout(title="Actual vs Predicted Spending")
    return fig

# Load data
data = load_data()

# Sidebar
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox('Select Model', ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
                                                   'Decision Tree', 'Random Forest', 'SVR'])
test_size = st.sidebar.slider('Test Set Size (%)', min_value=10, max_value=50, value=20, step=5) / 100.0
random_state = st.sidebar.number_input('Random State', min_value=0, max_value=100, value=42)

# Prepare data
X = data.drop(['Yearly Amount Spent', 'Email', 'Address', 'Avatar'], axis=1)
y = data['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate model
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'SVR': SVR()
}

model = models[model_name]
model, y_pred, mse, mae, r2 = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, model)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "📊 Data Visualization", "📈 Model Performance", "🔮 Prediction", "🧠 Quiz"])

with tab1:
    st.header("Understanding E-commerce Customer Spending Prediction")
    
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px;">
    <h3>What is E-commerce Customer Spending Prediction?</h3>
    <p>E-commerce customer spending prediction is a technique used to estimate how much a customer is likely to spend on an online platform based on various factors such as their behavior, engagement, and historical data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Key Factors in Predicting Customer Spending</h3>
    <ul>
        <li><strong>Avg. Session Length:</strong> Duration of customer's sessions on the platform.</li>
        <li><strong>Time on App:</strong> Time spent using the mobile application.</li>
        <li><strong>Time on Website:</strong> Time spent browsing the website.</li>
        <li><strong>Length of Membership:</strong> Duration of customer's association with the platform.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0fff0; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Why is it important?</h3>
    <ul>
        <li><span class="highlight">Personalized Marketing:</span> Tailor marketing strategies to different customer segments.</li>
        <li><span class="highlight">Inventory Management:</span> Anticipate demand and manage stock levels effectively.</li>
        <li><span class="highlight">Customer Retention:</span> Identify high-value customers and implement retention strategies.</li>
        <li><span class="highlight">Revenue Forecasting:</span> Make more accurate business projections and financial plans.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("Data Visualization")
    
    # Pairplot
    st.subheader("Feature Relationships")
    fig_pairplot = px.scatter_matrix(data, dimensions=['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent'])
    st.plotly_chart(fig_pairplot)
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = data[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent']].corr()
    fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(fig_corr)

with tab3:
    st.header("Model Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Squared Error", f"{mse:.2f}")
    with col2:
        st.metric("Mean Absolute Error", f"{mae:.2f}")
    with col3:
        st.metric("R-squared Score", f"{r2:.2f}")
    
    st.subheader("Actual vs Predicted Spending")
    fig_actual_vs_pred = plot_actual_vs_predicted(y_test, y_pred)
    st.plotly_chart(fig_actual_vs_pred)
    
    if model_name in ['Decision Tree', 'Random Forest']:
        st.subheader("Feature Importance")
        fig_importance = plot_feature_importance(X.columns, model.feature_importances_)
        st.plotly_chart(fig_importance)

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
            "correct": 1,
            "explanation": "The main goal is to estimate future customer spending based on their behavior and engagement data."
        },
        {
            "question": "Which of the following is NOT a key factor in predicting customer spending in this model?",
            "options": ["Average Session Length", "Time on App", "Customer's Age", "Length of Membership"],
            "correct": 2,
            "explanation": "Customer's Age is not one of the features used in this model. The key factors are Average Session Length, Time on App, Time on Website, and Length of Membership."
        },
        {
            "question": "Why is customer spending prediction important for e-commerce businesses?",
            "options": ["It helps in personalizing marketing strategies", "It's not important for e-commerce", "It directly increases sales"],
            "correct": 0,
            "explanation": "Customer spending prediction is important as it helps in personalizing marketing strategies, among other benefits like inventory management and customer retention."
        }
    ]

    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}: {q['question']}")
        user_answer = st.radio(f"Select your answer for Question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! Great job!")
            else:
                st.error("Not quite. Let's learn from this!")
            st.info(f"Explanation: {q['explanation']}")
        st.write("---")

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates e-commerce customer spending prediction. Adjust the model and data split, then explore the different tabs to learn more!")
