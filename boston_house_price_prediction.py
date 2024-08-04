import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set page config
st.set_page_config(page_title="Boston Housing Price Predictor", layout="wide", initial_sidebar_state="expanded")

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
st.title("🏘️ Boston Housing Price Predictor")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Discover the factors influencing Boston housing prices and predict home values!")

# Helper functions
def load_dataset():
    boston = load_boston()
    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    data['PRICE'] = boston.target
    return data, boston.feature_names

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, y_pred, mse, r2

def plot_feature_importance(feature_names, feature_importance):
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    fig = px.bar(feature_importance_df, x='importance', y='feature', title='Feature Importance')
    return fig

def plot_actual_vs_predicted(y_test, y_pred):
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Price', 'y': 'Predicted Price'})
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                             mode='lines', name='Ideal'))
    fig.update_layout(title="Actual vs Predicted Housing Prices")
    return fig

# Load data
data, feature_names = load_dataset()

# Sidebar
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox('Select Model', ['Linear Regression', 'Decision Tree', 'Random Forest'])
test_size = st.sidebar.slider('Test Set Size (%)', min_value=10, max_value=50, value=20, step=5) / 100.0
random_state = st.sidebar.number_input('Random State', min_value=0, max_value=100, value=42)

# Prepare data
X = data.drop('PRICE', axis=1)
y = data['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate model
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor()
}

model = models[model_name]
model, y_pred, mse, r2 = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, model)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "📊 Data Visualization", "📈 Model Performance", "🔮 Prediction", "🧠 Quiz"])

with tab1:
    st.header("Understanding Boston Housing Price Prediction")
    
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px;">
    <h3>What is the Boston Housing Dataset?</h3>
    <p>The Boston Housing Dataset is a famous dataset in machine learning that contains information about various features of houses in Boston and their median values.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Key Features in the Dataset</h3>
    <ul>
        <li><strong>CRIM:</strong> Per capita crime rate by town</li>
        <li><strong>RM:</strong> Average number of rooms per dwelling</li>
        <li><strong>AGE:</strong> Proportion of owner-occupied units built prior to 1940</li>
        <li><strong>LSTAT:</strong> Percentage of lower status of the population</li>
        <li><strong>PRICE:</strong> Median value of owner-occupied homes in $1000s (Target variable)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0fff0; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Why is this Dataset Important?</h3>
    <ul>
        <li><span class="highlight">Benchmark Dataset:</span> Widely used for testing machine learning algorithms</li>
        <li><span class="highlight">Real-world Application:</span> Provides insights into factors affecting housing prices</li>
        <li><span class="highlight">Feature Relationships:</span> Demonstrates complex interactions between various features</li>
        <li><span class="highlight">Regression Problem:</span> Excellent for learning and practicing regression techniques</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("Data Visualization")
    
    st.subheader("Feature Distributions")
    feature_to_plot = st.selectbox("Select a feature to visualize", feature_names)
    fig = px.histogram(data, x=feature_to_plot, nbins=30, marginal="box")
    st.plotly_chart(fig)
    
    st.subheader("Feature Correlations with Price")
    fig = px.scatter_matrix(data, dimensions=feature_names + ['PRICE'], color='PRICE')
    st.plotly_chart(fig)

    st.subheader("Correlation Heatmap")
    fig = px.imshow(data.corr(), color_continuous_scale='viridis')
    st.plotly_chart(fig)

with tab3:
    st.header("Model Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Squared Error", f"{mse:.2f}")
    with col2:
        st.metric("R-squared Score", f"{r2:.2f}")
    
    st.subheader("Actual vs Predicted Prices")
    fig = plot_actual_vs_predicted(y_test, y_pred)
    st.plotly_chart(fig)
    
    if model_name in ['Decision Tree', 'Random Forest']:
        st.subheader("Feature Importance")
        fig = plot_feature_importance(feature_names, model.feature_importances_)
        st.plotly_chart(fig)

with tab4:
    st.header("Make a Prediction")
    
    input_data = {}
    col1, col2 = st.columns(2)
    
    with col1:
        for feature in feature_names[:len(feature_names)//2]:
            input_data[feature] = st.slider(f"{feature}", float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))
    
    with col2:
        for feature in feature_names[len(feature_names)//2:]:
            input_data[feature] = st.slider(f"{feature}", float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))

    if st.button("🏘️ Predict Housing Price"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Housing Price: ${prediction[0]*1000:.2f}")

with tab5:
    st.header("Test Your Knowledge")

    questions = [
        {
            "question": "What does the target variable 'PRICE' represent in the Boston Housing dataset?",
            "options": ["Actual price of the house", "Median value of owner-occupied homes in $1000s", "Price per square foot"],
            "correct": 1,
            "explanation": "In the Boston Housing dataset, 'PRICE' represents the median value of owner-occupied homes in $1000s."
        },
        {
            "question": "Which of the following is NOT a feature in the Boston Housing dataset?",
            "options": ["CRIM (Per capita crime rate)", "RM (Average number of rooms)", "POPULATION (Town population)"],
            "correct": 2,
            "explanation": "POPULATION is not a feature in the Boston Housing dataset. The dataset includes features like CRIM and RM, but not town population directly."
        },
        {
            "question": "Why is the Boston Housing dataset popular in machine learning?",
            "options": ["It's the largest housing dataset available", "It's a good benchmark dataset for regression problems", "It only contains data about luxury homes"],
            "correct": 1,
            "explanation": "The Boston Housing dataset is popular because it's a good benchmark dataset for regression problems, containing various features that can influence housing prices."
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
st.sidebar.info("This app demonstrates the impact of various features on Boston housing prices. Adjust the model and data split, then explore the different tabs to learn more!")
