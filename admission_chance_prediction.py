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
st.set_page_config(page_title="Graduate Admission Predictor", layout="wide", initial_sidebar_state="expanded")

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
    background-color: #e6e6fa;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎓 Graduate Admission Predictor")
st.markdown("**Developed by: Your Name**")
st.markdown("Explore factors influencing graduate admission chances!")

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('Admission_Predict.csv')
    return data

data = load_data()

if data is None or data.empty:
    st.error("Failed to load the dataset. Please check the data source.")
    st.stop()

# Print data info for debugging
st.write("Dataset Info:")
st.write(data.info())

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
tab1, tab2, tab3, tab4 = st.tabs(["📚 Learn", "📊 Data Analysis", "🧮 Model Performance", "🧠 Quiz"])

with tab1:
    st.header("📚 Learn About Graduate Admissions")
    
    st.markdown("""
    <div class="highlight">
    <h3>Key Factors in Graduate Admissions</h3>
    <ul>
        <li>GRE Score: Graduate Record Examination score</li>
        <li>TOEFL Score: Test of English as a Foreign Language score</li>
        <li>University Rating: Rating of the undergraduate university (1-5)</li>
        <li>Statement of Purpose (SOP): Strength of the SOP (1-5)</li>
        <li>Letter of Recommendation (LOR): Strength of LOR (1-5)</li>
        <li>CGPA: Cumulative Grade Point Average</li>
        <li>Research Experience: Whether the applicant has research experience (0 or 1)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>Why These Factors Matter</h3>
    <ul>
        <li>Academic Performance: GRE, TOEFL, and CGPA reflect your academic abilities</li>
        <li>Research Aptitude: Research experience can be crucial for research-oriented programs</li>
        <li>Soft Skills: SOP and LOR provide insights into your communication and interpersonal skills</li>
        <li>University Background: The rating of your undergraduate institution can influence admissions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>Tips for Improving Your Admission Chances</h3>
    <ul>
        <li>Focus on maintaining a high CGPA throughout your undergraduate studies</li>
        <li>Prepare thoroughly for GRE and TOEFL exams</li>
        <li>Gain research experience through internships or projects</li>
        <li>Craft a strong Statement of Purpose that highlights your goals and experiences</li>
        <li>Build relationships with professors for strong Letters of Recommendation</li>
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
    corr = data.corr()
    fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Feature Distribution
    st.subheader("Feature Distribution")
    feature = st.selectbox("Select a feature to visualize:", data.columns)
    fig_dist = px.histogram(data, x=feature, marginal="box", hover_data=data.columns)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Scatter Plot
    st.subheader("Scatter Plot")
    x_axis = st.selectbox("Select X-axis:", data.columns, index=0)
    y_axis = st.selectbox("Select Y-axis:", data.columns, index=min(1, len(data.columns) - 1))
    color_by = st.selectbox("Color by:", data.columns, index=min(2, len(data.columns) - 1))
    fig_scatter = px.scatter(data, x=x_axis, y=y_axis, color=color_by, hover_data=data.columns)
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.header("🧮 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted Chance of Admit")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                 mode='lines', name='Ideal'))
        fig.update_layout(xaxis_title="Actual Chance of Admit", yaxis_title="Predicted Chance of Admit")
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
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "Which of these factors is NOT typically considered in graduate admissions?",
            "options": [
                "GRE Score",
                "TOEFL Score",
                "High School GPA",
                "Research Experience"
            ],
            "correct": 2,
            "explanation": "High School GPA is typically not considered for graduate admissions. The focus is more on undergraduate performance and standardized test scores like GRE and TOEFL."
        },
        {
            "question": "What does CGPA stand for?",
            "options": [
                "College Grade Point Average",
                "Cumulative Grade Point Average",
                "Calculated Grade Point Assessment",
                "Complete Grade Performance Analysis"
            ],
            "correct": 1,
            "explanation": "CGPA stands for Cumulative Grade Point Average. It's a measure of a student's overall academic performance across all courses."
        },
        {
            "question": "Which of these is likely to have the strongest positive correlation with admission chances?",
            "options": [
                "Age of the applicant",
                "Distance of the university from home",
                "CGPA",
                "Number of extracurricular activities"
            ],
            "correct": 2,
            "explanation": "CGPA (Cumulative Grade Point Average) typically has a strong positive correlation with admission chances, as it directly reflects academic performance."
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
st.sidebar.info("This app demonstrates the factors influencing graduate admission chances. Adjust the parameters and explore the different tabs to learn more!")

# Prediction Section
st.sidebar.header("Make a Prediction")
gre_score = st.sidebar.slider("GRE Score", 200, 340, 300)
toefl_score = st.sidebar.slider("TOEFL Score", 0, 120, 100)
university_rating = st.sidebar.slider("University Rating", 1, 5, 3)
sop = st.sidebar.slider("SOP", 1.0, 5.0, 3.0, 0.5)
lor = st.sidebar.slider("LOR", 1.0, 5.0, 3.0, 0.5)
cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 8.0, 0.1)
research = st.sidebar.selectbox("Research Experience", [0, 1])

if st.sidebar.button("🎓 Predict Chance of Admission"):
    input_data = pd.DataFrame([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]], 
                              columns=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research'])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.sidebar.success(f"Predicted Chance of Admission: {prediction[0]:.2%}")
