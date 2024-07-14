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
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Graduate Admission Predictor", layout="wide", page_icon="🎓")

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
    data = pd.read_csv('Admission_Predict.csv')
    return data

data = load_data()

# Column explanations
column_explanations = {
    "GRE Score": "Graduate Record Examination score",
    "TOEFL Score": "Test of English as a Foreign Language score",
    "University Rating": "Rating of the undergraduate university (1-5)",
    "SOP": "Statement of Purpose strength (1-5)",
    "LOR": "Letter of Recommendation strength (1-5)",
    "CGPA": "Cumulative Grade Point Average",
    "Research": "Research experience (0 = No, 1 = Yes)",
    "Chance of Admit": "Probability of admission (0-1)"
}

# Sidebar
st.sidebar.title("🎓 Graduate Admission Predictor")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Select Regression Algorithm",
    ["Linear Regression", "Ridge Regression", "Lasso Regression", 
     "Decision Tree", "Random Forest", "Support Vector Regression"]
)

# Main content
st.title("Graduate Admission Predictor")
st.write('**Developed by : Venugopal Adep**')
st.write("Predict the chance of admission based on various factors.")

# Prepare the data
@st.cache_data
def prepare_data(data):
    X = data.drop(['Chance of Admit'], axis=1)
    y = data['Chance of Admit']
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
    st.dataframe(data)
    
    # Column Explanations
    st.subheader("Column Explanations")
    for col, explanation in column_explanations.items():
        st.write(f"**{col}**: {explanation}")
    
    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(data.describe())
    
    # Univariate Analysis
    st.subheader("Univariate Analysis")
    feature_univariate = st.selectbox("Select a feature for univariate analysis", data.columns)
    fig_univariate = px.histogram(data, x=feature_univariate, marginal="box")
    st.plotly_chart(fig_univariate)
    
    # Bivariate Analysis
    st.subheader("Bivariate Analysis")
    feature_bivariate = st.selectbox("Select a feature for bivariate analysis with Chance of Admit", 
                                     [col for col in data.columns if col != 'Chance of Admit'])
    fig_bivariate = px.scatter(data, x=feature_bivariate, y='Chance of Admit', color="Research", hover_data=['GRE Score', 'TOEFL Score'])
    st.plotly_chart(fig_bivariate)
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = data.corr()
    fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(fig_corr)
    
    # Crosstab Analysis (for demonstration, we'll create a categorical column)
    st.subheader("Crosstab Analysis")
    data['GRE_Category'] = pd.cut(data['GRE Score'], bins=3, labels=['Low', 'Medium', 'High'])
    crosstab = pd.crosstab(data['GRE_Category'], data['Research'])
    st.write(crosstab)
    fig_crosstab = px.bar(crosstab, barmode='group')
    st.plotly_chart(fig_crosstab)
    
    # Word Cloud (for demonstration, we'll create a text column)
    st.subheader("Word Cloud")
    data['Comments'] = data.apply(lambda row: f"GRE:{row['GRE Score']} TOEFL:{row['TOEFL Score']} CGPA:{row['CGPA']}", axis=1)
    text = ' '.join(data['Comments'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig_wordcloud, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig_wordcloud)

with tab2:
    st.header("Model Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted Chance of Admit")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                 mode='lines', name='Ideal'))
        fig.update_layout(xaxis_title="Actual Chance of Admit", yaxis_title="Predicted Chance of Admit")
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
        importances = model.feature_importances_
        feature_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
        feature_imp = feature_imp.sort_values('importance', ascending=False).reset_index(drop=True)
        fig = px.bar(feature_imp, x='importance', y='feature', orientation='h')
        st.plotly_chart(fig)

with tab3:
    st.header("Model Explanation")
    
    if algorithm == "Linear Regression":
        st.write("""
        Linear Regression finds the best linear relationship between the input features and the target variable (Chance of Admit).
        It assumes that the chance of admission can be predicted as a weighted sum of the input features.

        The equation is: Chance of Admit = w1*x1 + w2*x2 + ... + wn*xn + b

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

        For example: If (GRE Score > 320) and (CGPA > 8.5), then predict high chance of admission.

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
        gre_score = st.number_input("GRE Score", min_value=200, max_value=340, value=300)
        toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
        university_rating = st.slider("University Rating", 1, 5, 3)
        sop = st.slider("SOP", 1.0, 5.0, 3.0, 0.5)
    
    with col2:
        lor = st.slider("LOR", 1.0, 5.0, 3.0, 0.5)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
        research = st.selectbox("Research Experience", [0, 1])

    if st.button("🎓 Predict Chance of Admission", key="predict_button"):
        input_data = pd.DataFrame([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]], 
                                  columns=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research'])
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Chance of Admission: {prediction[0]:.2%}")