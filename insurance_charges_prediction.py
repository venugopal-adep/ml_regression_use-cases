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
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Insure Charges Predictor", layout="wide", page_icon="🏥")

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
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stMarkdown {
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('insurance.csv')
    return data

data = load_data()

# Preprocess data
def preprocess_data(data):
    categorical_columns = ['sex', 'smoker', 'region']
    numerical_columns = ['age', 'bmi', 'children']
    
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    return data_encoded[numerical_columns + list(set(data_encoded.columns) - set(numerical_columns))]

data_processed = preprocess_data(data)

# Sidebar
st.sidebar.title("🎛️ Control Panel")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Select Regression Algorithm",
    ["Linear Regression", "Ridge Regression", "Lasso Regression", 
     "Decision Tree", "Random Forest", "Support Vector Regression"]
)

# Main content
st.title("🏥 Insurance Charges Prediction")
st.write("**Developed by: Venugopal Adep**")
st.write("Predict insurance charges with cutting-edge machine learning algorithms.")

# Prepare the data
X = data_processed.drop('charges', axis=1)
y = data['charges']
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
    
    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(data.describe())
    
    # Univariate Analysis
    st.subheader("Univariate Analysis")
    feature_univariate = st.selectbox("Select a feature for univariate analysis", list(data.columns))
    fig_univariate = px.histogram(data, x=feature_univariate, marginal="box")
    st.plotly_chart(fig_univariate)
    
    # Bivariate Analysis
    st.subheader("Bivariate Analysis")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    feature_bivariate = st.selectbox("Select a feature for bivariate analysis with charges", 
                                     [col for col in numeric_columns if col != 'charges'])
    fig_bivariate = px.scatter(data, x=feature_bivariate, y='charges', color="sex", hover_data=['age'])
    st.plotly_chart(fig_bivariate)
    
    # Boxplot
    st.subheader("Boxplot")
    feature_box = st.selectbox("Select a feature for boxplot", list(data.columns))
    fig_box = px.box(data, y=feature_box, x="sex", color="smoker")
    st.plotly_chart(fig_box)
    
    # Wordcloud
    st.subheader("Wordcloud of Regions")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data['region']))
    fig_wordcloud, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig_wordcloud)
    
    # Crosstab Analysis
    st.subheader("Crosstab Analysis")
    discrete_columns = ['sex', 'smoker', 'region', 'children']
    col1, col2 = st.columns(2)
    with col1:
        feature1 = st.selectbox("Select first feature for crosstab", discrete_columns)
    with col2:
        feature2 = st.selectbox("Select second feature for crosstab", 
                                [col for col in discrete_columns if col != feature1])
    crosstab = pd.crosstab(data[feature1], data[feature2])
    st.dataframe(crosstab)
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(fig_corr)

with tab2:
    st.header("Model Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted Charges")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                 mode='lines', name='Ideal'))
        fig.update_layout(xaxis_title="Actual Charges", yaxis_title="Predicted Charges")
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
        Linear Regression is like drawing a straight line through a scatter plot of data points. It tries to find the best line that minimizes the overall distance between the line and all the points.

        Example: Imagine you're trying to predict a house's price based on its size. Linear regression would try to find the best straight line relationship between size and price.

        The mathematical equation is: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

        Where:
        - y is the predicted value (e.g., insurance charges)
        - β₀ is the y-intercept
        - β₁, β₂, ..., βₙ are the coefficients for each feature
        - x₁, x₂, ..., xₙ are the feature values
        - ε is the error term
        """)

    elif algorithm == "Ridge Regression":
        st.write("""
        Ridge Regression is similar to Linear Regression, but it adds a penalty term to prevent overfitting. It's like telling the model to keep the coefficients small unless they're really important.

        Example: In predicting insurance charges, Ridge Regression might reduce the impact of less important factors, focusing more on key predictors like age or smoking status.

        The mathematical equation is: min(||y - Xβ||² + α||β||²)

        Where:
        - ||y - Xβ||² is the ordinary least squares error
        - α||β||² is the ridge penalty term
        - α is the regularization strength
        """)

    elif algorithm == "Lasso Regression":
        st.write("""
        Lasso Regression is another variation of Linear Regression that can completely eliminate the impact of less important features. It's like a feature selection tool built into the regression.

        Example: When predicting insurance charges, Lasso might completely ignore factors that don't significantly impact the charges, simplifying the model.

        The mathematical equation is: min(||y - Xβ||² + α||β||₁)

        Where:
        - ||y - Xβ||² is the ordinary least squares error
        - α||β||₁ is the lasso penalty term
        - α is the regularization strength
        """)

    elif algorithm == "Decision Tree":
        st.write("""
        A Decision Tree is like a flowchart of yes/no questions. It splits the data based on features, trying to group similar outcomes together.

        Example: For insurance charges, it might first ask "Is the person a smoker?", then "Are they over 50?", and so on, creating a tree-like structure of decisions.

        The mathematical concept involves minimizing the impurity at each split, often using metrics like Gini impurity or entropy.

        Gini impurity: 1 - Σ(pᵢ²)
        Entropy: -Σ(pᵢ * log₂(pᵢ))

        Where pᵢ is the probability of an item being classified for a particular class.
        """)

    elif algorithm == "Random Forest":
        st.write("""
        Random Forest is like having a committee of decision trees. Each tree is slightly different, and they vote on the final prediction.

        Example: Multiple trees might predict insurance charges differently based on slightly different rules or data samples, and the final prediction is an average of all these predictions.

        Mathematically, it's an ensemble method that combines multiple decision trees:

        final_prediction = 1/n * Σ(tree_predictionᵢ)

        Where n is the number of trees in the forest.
        """)

    elif algorithm == "Support Vector Regression":
        st.write("""
        Support Vector Regression tries to find a tube that fits the data points as closely as possible. Points outside this tube are penalized.

        Example: In predicting insurance charges, SVR might create a complex, potentially non-linear, function that tries to capture the relationship between various factors and the charges.

        The objective is to minimize:

        1/2 ||w||² + C Σ(ξᵢ + ξᵢ*)

        Subject to:
        - y - ⟨w, x⟩ - b ≤ ε + ξᵢ
        - ⟨w, x⟩ + b - y ≤ ε + ξᵢ*
        - ξᵢ, ξᵢ* ≥ 0

        Where w is the normal vector to the hyperplane, C > 0 is the trade-off parameter, and ε is the threshold.
        """)

with tab4:
    st.header("Make a Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", min_value=18, max_value=100, value=30)
        bmi = st.slider("BMI", min_value=10.0, max_value=50.0, value=25.0)
        children = st.slider("Number of Children", min_value=0, max_value=10, value=0)
    
    with col2:
        sex = st.radio("Sex", ["male", "female"])
        smoker = st.radio("Smoker", ["yes", "no"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

    if st.button("💰 Predict Charges", key="predict_button"):
        input_data = pd.DataFrame({
            'age': [age], 'sex': [sex], 'bmi': [bmi],
            'children': [children], 'smoker': [smoker], 'region': [region]
        })
        input_encoded = preprocess_data(input_data)
        missing_cols = set(X.columns) - set(input_encoded.columns)
        for col in missing_cols:
            input_encoded[col] = 0
        input_encoded = input_encoded[X.columns]
        input_scaled = scaler.transform(input_encoded)
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Insurance Charge: ${prediction[0]:,.2f}")

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
        Developed by Venugopal Adep | Data source: https://www.kaggle.com/datasets/simranjain17/insurance
    </div>
    """, unsafe_allow_html=True)