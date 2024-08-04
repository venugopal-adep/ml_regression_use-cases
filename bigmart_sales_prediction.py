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
st.set_page_config(page_title="BigMart Sales Predictor", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better appearance
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
st.title("🛒 BigMart Sales Predictor")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Explore factors influencing product sales in BigMart stores!")

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('bigmart_sales.csv')
    return data

data = load_data()

# Prepare the data
@st.cache_data
def prepare_data(data):
    # Convert categorical variables to numeric
    data['Item_Fat_Content'] = data['Item_Fat_Content'].map({'Low Fat': 0, 'Regular': 1, 'LF': 0, 'reg': 1, 'low fat': 0})
    data = pd.get_dummies(data, columns=['Item_Type', 'Outlet_Type', 'Outlet_Size', 'Outlet_Location_Type'])
    
    # Drop unnecessary columns
    X = data.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'], axis=1)
    y = data['Item_Outlet_Sales']
    
    return X, y

X, y = prepare_data(data)

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
    st.header("📚 Learn About BigMart Sales")
    
    st.markdown("""
    <div class="highlight">
    <h3>Key Factors in BigMart Sales</h3>
    <ul>
        <li>Item Weight: Weight of the product</li>
        <li>Item Fat Content: Whether the product is low fat or regular</li>
        <li>Item Visibility: The percentage of total display area of all products in a store allocated to the particular product</li>
        <li>Item Type: The category to which the product belongs</li>
        <li>Item MRP: Maximum Retail Price (list price) of the product</li>
        <li>Outlet Establishment Year: The year in which store was established</li>
        <li>Outlet Size: The size of the store in terms of ground area covered</li>
        <li>Outlet Location Type: The type of city in which the store is located</li>
        <li>Outlet Type: Whether the outlet is just a grocery store or some sort of supermarket</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>Why These Factors Matter</h3>
    <ul>
        <li>Product Characteristics: Weight, fat content, and type can influence consumer preferences</li>
        <li>Pricing: MRP is a crucial factor in determining sales</li>
        <li>Store Features: Size, location, and type of outlet can affect customer footfall and sales</li>
        <li>Visibility: Higher visibility can lead to more sales</li>
        <li>Establishment Year: Older stores might have a loyal customer base</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>Tips for Improving Sales</h3>
    <ul>
        <li>Optimize product placement to increase visibility</li>
        <li>Tailor product offerings based on outlet location and type</li>
        <li>Consider promotional strategies for items with higher MRP</li>
        <li>Analyze the performance of different product types and adjust inventory accordingly</li>
        <li>Leverage the strengths of each outlet type (e.g., convenience of grocery stores vs. variety in supermarkets)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("📊 Data Analysis")
    
    # Dataset Explorer
    st.subheader("Dataset Explorer")
    st.dataframe(data)
    
    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(data.describe())
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    corr = data[numeric_columns].corr()
    fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Feature Distribution
    st.subheader("Feature Distribution")
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

with tab3:
    st.header("🧮 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted Sales")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                 mode='lines', name='Ideal'))
        fig.update_layout(xaxis_title="Actual Sales", yaxis_title="Predicted Sales")
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
        item_weight = st.slider("Item Weight", 0.0, 50.0, 10.0, 0.1)
        item_visibility = st.slider("Item Visibility", 0.0, 0.3, 0.1, 0.01)
        item_mrp = st.slider("Item MRP", 0.0, 300.0, 100.0, 1.0)
        outlet_establishment_year = st.slider("Outlet Establishment Year", 1950, 2022, 1990)
    
    with col2:
        item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
        outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
        outlet_location_type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
        outlet_type = st.selectbox("Outlet Type", data['Outlet_Type'].unique())
        item_type = st.selectbox("Item Type", data['Item_Type'].unique())

    if st.button("🛒 Predict Sales"):
        input_data = pd.DataFrame(0, index=[0], columns=X.columns)
        
        input_data['Item_Weight'] = item_weight
        input_data['Item_Visibility'] = item_visibility
        input_data['Item_MRP'] = item_mrp
        input_data['Outlet_Establishment_Year'] = outlet_establishment_year
        input_data['Item_Fat_Content'] = 0 if item_fat_content == "Low Fat" else 1
        input_data[f'Outlet_Size_{outlet_size}'] = 1
        input_data[f'Outlet_Location_Type_{outlet_location_type}'] = 1
        input_data[f'Outlet_Type_{outlet_type}'] = 1
        input_data[f'Item_Type_{item_type}'] = 1
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Sales: ${prediction[0]:.2f}")
        
        # Radar chart for input features
        features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']
        values = [item_weight, item_visibility, item_mrp, outlet_establishment_year]
        
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
            "question": "Which of these factors is NOT typically considered in BigMart sales prediction?",
            "options": [
                "Item Weight",
                "Item Visibility",
                "Customer Age",
                "Outlet Type"
            ],
            "correct": 2,
            "explanation": "Customer Age is not typically considered in BigMart sales prediction. The focus is more on product characteristics and store features."
        },
        {
            "question": "What does MRP stand for in the context of BigMart sales?",
            "options": [
                "Most Recent Price",
                "Maximum Retail Price",
                "Minimum Required Profit",
                "Market Rate Price"
            ],
            "correct": 1,
            "explanation": "MRP stands for Maximum Retail Price. It's the list price of the product and a crucial factor in determining sales."
        },
        {
            "question": "Which of these is likely to have a strong influence on a product's sales?",
            "options": [
                "The color of the product packaging",
                "The day of the week",
                "Item Visibility",
                "The store manager's name"
            ],
            "correct": 2,
            "explanation": "Item Visibility typically has a strong influence on sales. Products that are more visible in the store are more likely to be purchased."
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
st.sidebar.info("This app demonstrates the factors influencing BigMart sales. Adjust the parameters and explore the different tabs to learn more!")

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
    Developed by Venugopal Adep | Data source: BigMart Sales Dataset
</div>
""", unsafe_allow_html=True)

# Add some spacing at the bottom to prevent content from being hidden by the footer
st.write("<br><br><br>", unsafe_allow_html=True)
