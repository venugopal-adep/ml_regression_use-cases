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

# Set page config
st.set_page_config(page_title="Avocado Price Predictor", layout="wide", page_icon="🥑")

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f0f8ff;
}
.stButton>button {
    background-color: #4CAF50;
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
    background-color: #4CAF50;
    color: white;
}
.highlight {
    background-color: #e6e6fa;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🥑 Avocado Price Predictor")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Explore factors influencing avocado prices and predict future prices!")

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('avocado_data.csv')
    return data

data = load_data()

# Prepare the data
@st.cache_data
def prepare_data(data):
    # Convert Date to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Extract features from Date
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Month'] = data['Date'].dt.month
    
    # Create a copy of the original data
    original_data = data.copy()
    
    # Convert categorical variables to numeric
    data['type'] = pd.Categorical(data['type']).codes
    data['region'] = pd.Categorical(data['region']).codes
    
    X = data.drop(['AveragePrice', 'Date'], axis=1)
    y = data['AveragePrice']
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return X, y, imputer, original_data

X, y, imputer, original_data = prepare_data(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sidebar
st.sidebar.header("Parameters")
algorithm = st.sidebar.selectbox(
    "Select Regression Algorithm",
    ["Linear Regression", "Ridge Regression", "Lasso Regression", 
     "Decision Tree", "Random Forest", "Support Vector Regression"]
)

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

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "📊 Data Analysis", "🧮 Model Performance", "🔮 Predictions", "🧠 Quiz"])

with tab1:
    st.header("📚 Learn About Avocado Prices")
    
    st.markdown("""
    <div class="highlight">
    <h3>Key Factors Influencing Avocado Prices</h3>
    <ul>
        <li>Type: Conventional or organic</li>
        <li>Region: Location where avocados are sold</li>
        <li>Total Volume: Total number of avocados sold</li>
        <li>PLU4046, PLU4225, PLU4770: Sales of different avocado sizes</li>
        <li>Total Bags, Small Bags, Large Bags, XLarge Bags: Sales by packaging</li>
        <li>Season: Time of year can affect prices</li>
        <li>Year: Prices can trend over time</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>Why These Factors Matter</h3>
    <ul>
        <li>Supply and Demand: Total volume and regional differences affect pricing</li>
        <li>Consumer Preferences: Different sizes and types have varying demand</li>
        <li>Seasonal Effects: Avocado production and consumption vary throughout the year</li>
        <li>Long-term Trends: Changes in dietary habits and production methods influence prices over years</li>
        <li>Packaging: Different packaging types can affect pricing strategies</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>Tips for Understanding Avocado Prices</h3>
    <ul>
        <li>Monitor seasonal trends to anticipate price fluctuations</li>
        <li>Consider regional differences in supply and demand</li>
        <li>Pay attention to the difference between organic and conventional avocados</li>
        <li>Understand how different sizes and packaging options affect pricing</li>
        <li>Keep an eye on long-term trends in avocado consumption and production</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("📊 Data Analysis")
    
    # Dataset Explorer
    st.subheader("Dataset Explorer")
    st.dataframe(data.head())
    
    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(data.describe())
    
    # Feature Distribution
    st.subheader("Feature Distribution")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    feature = st.selectbox("Select a feature to visualize:", numeric_columns)
    fig_dist = px.histogram(data, x=feature, marginal="box", hover_data=data.columns)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Scatter Plot
    st.subheader("Scatter Plot")
    x_axis = st.selectbox("Select X-axis:", numeric_columns, index=0)
    y_axis = st.selectbox("Select Y-axis:", numeric_columns, index=min(1, len(numeric_columns) - 1))
    color_by = st.selectbox("Color by:", data.columns, index=min(2, len(data.columns) - 1))
    fig_scatter = px.scatter(data, x=x_axis, y=y_axis, color=color_by, hover_data=data.columns)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = data[numeric_columns].corr()
    fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Time Series Plot
    st.subheader("Time Series Plot")
    fig_time = px.line(data, x='Date', y='AveragePrice', color='type')
    st.plotly_chart(fig_time, use_container_width=True)

with tab3:
    st.header("🧮 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted Average Price")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                 mode='lines', name='Ideal'))
        fig.update_layout(xaxis_title="Actual Average Price", yaxis_title="Predicted Average Price")
        st.plotly_chart(fig, use_container_width=True)
        
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
    
    if algorithm in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
        st.subheader("Feature Coefficients")
        coeffs = pd.DataFrame({'feature': X.columns, 'coefficient': model.coef_})
        coeffs = coeffs.sort_values('coefficient', key=abs, ascending=False).reset_index(drop=True)
        fig_coeffs = px.bar(coeffs, x='coefficient', y='feature', orientation='h')
        st.plotly_chart(fig_coeffs, use_container_width=True)
    
    elif algorithm in ["Decision Tree", "Random Forest"]:
        st.subheader("Feature Importance")
        importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
        importances = importances.sort_values('importance', ascending=False).reset_index(drop=True)
        fig_importance = px.bar(importances, x='importance', y='feature', orientation='h')
        st.plotly_chart(fig_importance, use_container_width=True)

with tab4:
    st.header("🔮 Make a Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        avocado_type = st.selectbox("Type", original_data['type'].unique())
        region = st.selectbox("Region", original_data['region'].unique())
        year = st.number_input("Year", min_value=data['year'].min(), max_value=data['year'].max(), value=2018)
        total_volume = st.number_input("Total Volume", min_value=0, value=100000)
        plu4046 = st.number_input("PLU4046 (Small/Medium Hass)", min_value=0, value=1000)
    
    with col2:
        plu4225 = st.number_input("PLU4225 (Large Hass)", min_value=0, value=1000)
        plu4770 = st.number_input("PLU4770 (Extra Large Hass)", min_value=0, value=1000)
        total_bags = st.number_input("Total Bags", min_value=0, value=1000)
        small_bags = st.number_input("Small Bags", min_value=0, value=500)
        large_bags = st.number_input("Large Bags", min_value=0, value=400)
        xlarge_bags = st.number_input("XLarge Bags", min_value=0, value=100)
        
    # Add day of week and month (assuming current date)
    current_date = pd.Timestamp.now()
    day_of_week = current_date.dayofweek
    month = current_date.month

    if st.button("🥑 Predict Price"):
        input_data = pd.DataFrame(0, index=[0], columns=X.columns)
        
        input_data['type'] = pd.Categorical(original_data['type']).codes[list(original_data['type'].unique()).index(avocado_type)]
        input_data['region'] = pd.Categorical(original_data['region']).codes[list(original_data['region'].unique()).index(region)]
        input_data['year'] = year
        input_data['Total Volume'] = total_volume
        input_data['PLU4046'] = plu4046
        input_data['PLU4225'] = plu4225
        input_data['PLU4770'] = plu4770
        input_data['Total Bags'] = total_bags
        input_data['Small Bags'] = small_bags
        input_data['Large Bags'] = large_bags
        input_data['XLarge Bags'] = xlarge_bags
        input_data['DayOfWeek'] = day_of_week
        input_data['Month'] = month
        
        input_data = pd.DataFrame(imputer.transform(input_data), columns=input_data.columns)
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Average Price: ${prediction[0]:.2f}")
        
        # Radar chart for input features
        features = ['Total Volume', 'PLU4046', 'PLU4225', 'PLU4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']
        values = input_data[features].values[0].tolist()
        
        fig = go.Figure(data=go.Scatterpolar(
          r=values,
          theta=features,
          fill='toself'
        ))

        fig.update_layout(
          polar=dict(
            radialaxis=dict(
              visible=True,
              range=[0, max(values)]
            )),
          showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "Which of these factors is NOT typically considered in avocado price prediction?",
            "options": [
                "Total Volume",
                "Region",
                "Customer Age",
                "Avocado Type (Conventional/Organic)"
            ],
            "correct": 2,
            "explanation": "Customer Age is not typically considered in avocado price prediction. The focus is more on supply, demand, and product characteristics."
        },
        {
            "question": "What does PLU stand for in the context of avocado sales?",
            "options": [
                "Price Look-Up",
                "Produce Listing Unit",
                "Package Label Unit",
                "Product Line Usage"
            ],
            "correct": 0,
            "explanation": "PLU stands for Price Look-Up. It's a code used to identify and price produce items."
        },
        {
            "question": "Which of these is likely to have a strong influence on avocado prices?",
            "options": [
                "The color of the store's walls",
                "The day of the week",
                "Total Volume of avocados sold",
                "The store manager's favorite fruit"
            ],
            "correct": 2,
            "explanation": "Total Volume of avocados sold typically has a strong influence on prices. It's a direct measure of supply and demand."
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}: {q['question']}")
        user_answer = st.radio(f"Select your answer for Question {i+1}:", q['options'], key=f"q{i}")
        
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! Well done!")
            else:
                st.error("Not quite right. Let's learn from this!")
            st.info(f"Explanation: {q['explanation']}")
        st.write("---")

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates the factors influencing avocado prices. Adjust the parameters and explore the different tabs to learn more!")

# Footer
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
    Developed by Venugopal Adep | Data source: Avocado Prices Dataset
</div>
""", unsafe_allow_html=True)

# Add some spacing at the bottom to prevent content from being hidden by the footer
st.write("<br><br><br>", unsafe_allow_html=True)
