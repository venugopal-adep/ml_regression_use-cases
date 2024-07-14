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
import io

# Set page config
st.set_page_config(page_title="BigMart Sales Predictor", layout="wide", page_icon="🛒")

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
    data = pd.read_csv('bigmart_sales.csv')
    return data

data = load_data()

# Column explanations
column_explanations = {
    "Item_Identifier": "Unique product ID",
    "Item_Fat_Content": "Whether the product is low fat or regular",
    "Item_Type": "The category to which the product belongs",
    "Outlet_Identifier": "Unique store ID",
    "Outlet_Type": "The size of the store in terms of ground area covered",
    "Outlet_Size": "The type of city in which the store is located",
    "Outlet_Location_Type": "The type of outlet (grocery store or supermarket)",
    "Item_Weight": "Weight of product",
    "Item_Visibility": "The percentage of total display area of all products in a store allocated to the particular product",
    "Item_MRP": "Maximum Retail Price (list price) of the product",
    "Outlet_Establishment_Year": "The year in which store was established",
    "Item_Outlet_Sales": "Sales of the product in the particular store (Target Variable)"
}

# Sidebar
st.sidebar.title("🛒 BigMart Sales Predictor")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Select Regression Algorithm",
    ["Linear Regression", "Ridge Regression", "Lasso Regression", 
     "Decision Tree", "Random Forest", "Support Vector Regression"]
)

# Main content
st.title("BigMart Sales Predictor")
st.write('**Developed by : Venugopal Adep**')
st.write("Predict sales of products in different outlets based on various features.")

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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Analysis", "🧮 Model Performance", "📘 Model Explanation", "🔮 Prediction"])

with tab1:
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
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    feature_univariate = st.selectbox("Select a feature for univariate analysis", numeric_columns)
    fig_univariate = px.histogram(data, x=feature_univariate, marginal="box")
    st.plotly_chart(fig_univariate)
    
    # Bivariate Analysis
    st.subheader("Bivariate Analysis")
    feature_bivariate = st.selectbox("Select a feature for bivariate analysis with Item_Outlet_Sales", 
                                     [col for col in numeric_columns if col != 'Item_Outlet_Sales'])
    fig_bivariate = px.scatter(data, x=feature_bivariate, y='Item_Outlet_Sales', color="Item_Type", hover_data=['Item_Identifier'])
    st.plotly_chart(fig_bivariate)
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = data[numeric_columns].corr()
    fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(fig_corr)

with tab2:
    st.header("Model Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted Sales")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                 mode='lines', name='Ideal'))
        fig.update_layout(xaxis_title="Actual Sales", yaxis_title="Predicted Sales")
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

with tab3:
    st.header("Model Explanation")
    
    if algorithm == "Linear Regression":
        st.write("""
        Linear Regression finds the best linear relationship between the input features and the target variable (Item_Outlet_Sales).
        It assumes that the sales can be predicted as a weighted sum of the input features.

        The equation is: Item_Outlet_Sales = w1*x1 + w2*x2 + ... + wn*xn + b

        Where w1, w2, ..., wn are the weights for each feature, x1, x2, ..., xn are the feature values, and b is the bias term.
        """)

    elif algorithm == "Ridge Regression":
        st.write("""
        Ridge Regression is similar to Linear Regression but adds a penalty term to prevent overfitting.
        It's useful when there might be high correlations between input features.

        The objective is to minimize: ||y - Xw||² + α||w||²

        Where α is the regularization strength, controlling the impact of the penalty term.
        """)

    elif algorithm == "Lasso Regression":
        st.write("""
        Lasso Regression also adds a penalty term, but it can completely eliminate the impact of less important features.
        This makes it useful for feature selection.

        The objective is to minimize: ||y - Xw||² + α||w||₁

        Where α is the regularization strength, controlling the impact of the penalty term.
        """)

    elif algorithm == "Decision Tree":
        st.write("""
        Decision Tree creates a tree-like model of decisions based on the input features.
        It splits the data based on different conditions to make predictions.

        For example: If (Item_MRP > 100) and (Outlet_Type = Supermarket Type1), then predict high sales.

        The tree is created by minimizing impurity (often measured by mean squared error for regression) at each split.
        """)

    elif algorithm == "Random Forest":
        st.write("""
        Random Forest is an ensemble of Decision Trees. It creates multiple trees and aggregates their predictions.
        This helps to reduce overfitting and improve generalization.

        The final prediction is typically the average of all individual tree predictions:
        Prediction = (Tree1 + Tree2 + ... + TreeN) / N

        Where N is the number of trees in the forest.
        """)

    elif algorithm == "Support Vector Regression":
        st.write("""
        Support Vector Regression tries to find a function that deviates from y by a value no greater than ε for each training point x.

        It aims to solve:
        minimize 1/2 ||w||² subject to |y - f(x)| ≤ ε

        Where f(x) is the prediction function and ε is the maximum allowed deviation.
        """)

with tab4:
    st.header("Make a Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        item_weight = st.number_input("Item Weight", min_value=0.0, max_value=50.0, value=10.0)
        item_visibility = st.number_input("Item Visibility", min_value=0.0, max_value=1.0, value=0.1)
        item_mrp = st.number_input("Item MRP", min_value=0.0, max_value=300.0, value=100.0)
    
    with col2:
        item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
        outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
        outlet_location_type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
        outlet_type = st.selectbox("Outlet Type", data['Outlet_Type'].unique())
        item_type = st.selectbox("Item Type", data['Item_Type'].unique())

    if st.button("🛒 Predict Sales", key="predict_button"):
        # Create a DataFrame with all features, initialized with zeros
        input_data = pd.DataFrame(0, index=[0], columns=X.columns)
        
        # Fill in the values for the features we have
        input_data['Item_Weight'] = item_weight
        input_data['Item_Visibility'] = item_visibility
        input_data['Item_MRP'] = item_mrp
        input_data['Item_Fat_Content'] = 0 if item_fat_content == "Low Fat" else 1
        input_data['Outlet_Establishment_Year'] = 2022  # Assuming current year
        
        # One-hot encode categorical variables
        input_data[f'Outlet_Size_{outlet_size}'] = 1
        input_data[f'Outlet_Location_Type_{outlet_location_type}'] = 1
        input_data[f'Outlet_Type_{outlet_type}'] = 1
        input_data[f'Item_Type_{item_type}'] = 1
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Item Outlet Sales: ${prediction[0]:,.2f}")

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
        Developed by Venugopal Adep | Data source: https://www.kaggle.com/datasets/uniabhi/bigmart-sales-data
    </div>
    """, unsafe_allow_html=True)