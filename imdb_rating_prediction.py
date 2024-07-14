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
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="IMDB Movie Rating Predictor", layout="wide", page_icon="🎬")

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
    data = pd.read_csv('imdb_top.csv')
    return data

data = load_data()

# Column explanations
column_explanations = {
    "Series_Title": "Title of the movie",
    "Released_Year": "Year the movie was released",
    "Certificate": "Movie rating by the censor board",
    "Runtime": "Total runtime of the movie",
    "Genre": "Genre of the movie",
    "Overview": "Brief summary of the movie",
    "Meta_score": "Score assigned by metacritic",
    "Director": "Director of the movie",
    "Star1": "Lead actor/actress",
    "Star2": "Supporting actor/actress",
    "Star3": "Supporting actor/actress",
    "Star4": "Supporting actor/actress",
    "No_of_Votes": "Number of votes received",
    "IMDB_Rating": "Rating on IMDB (Target Variable)"
}

# Sidebar
st.sidebar.title("🎬 IMDB Movie Rating Predictor")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Select Regression Algorithm",
    ["Linear Regression", "Ridge Regression", "Lasso Regression", 
     "Decision Tree", "Random Forest", "Support Vector Regression"]
)

# Main content
st.title("IMDB Movie Rating Predictor")
st.write('**Developed by : Venugopal Adep**')
st.write("Predict the IMDB rating of movies based on various features.")

# Prepare the data
@st.cache_data
def prepare_data(data):
    # Convert Runtime to minutes
    data['Runtime'] = data['Runtime'].str.extract('(\d+)').astype(int)
    
    # Create a copy of the original data for the word cloud
    original_data = data.copy()
    
    # Separate features and target
    X = data[['Released_Year', 'Certificate', 'Runtime', 'Genre', 'Meta_score', 'Director', 'No_of_Votes']]
    y = data['IMDB_Rating']
    
    # Identify numeric and categorical columns
    numeric_features = ['Released_Year', 'Runtime', 'Meta_score', 'No_of_Votes']
    categorical_features = ['Certificate', 'Genre', 'Director']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return X, y, preprocessor, original_data, numeric_features, categorical_features

X, y, preprocessor, original_data, numeric_features, categorical_features = prepare_data(data)

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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Analysis", "🧮 Model Performance", "📘 Model Explanation", "🔮 Prediction"])

with tab1:
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
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    feature_univariate = st.selectbox("Select a feature for univariate analysis", numeric_columns)
    fig_univariate = px.histogram(data, x=feature_univariate, marginal="box")
    st.plotly_chart(fig_univariate)
    
    # Bivariate Analysis
    st.subheader("Bivariate Analysis")
    feature_bivariate = st.selectbox("Select a feature for bivariate analysis with IMDB Rating", 
                                     [col for col in numeric_columns if col != 'IMDB_Rating'])
    fig_bivariate = px.scatter(data, x=feature_bivariate, y='IMDB_Rating', color="Genre", hover_data=['Series_Title'])
    st.plotly_chart(fig_bivariate)
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = data[numeric_columns].corr()
    fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(fig_corr)
    
    # Word Cloud
    st.subheader("Word Cloud of Movie Genres")
    text = ' '.join(original_data['Genre'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig_wordcloud, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig_wordcloud)

with tab2:
    st.header("Model Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted IMDB Rating")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                 mode='lines', name='Ideal'))
        fig.update_layout(xaxis_title="Actual IMDB Rating", yaxis_title="Predicted IMDB Rating")
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
        feature_names = (numeric_features + 
                        model.named_steps['preprocessor']
                        .named_transformers_['cat']
                        .get_feature_names_out(categorical_features).tolist())
        feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_imp = feature_imp.sort_values('importance', ascending=False).reset_index(drop=True)
        fig = px.bar(feature_imp, x='importance', y='feature', orientation='h')
        st.plotly_chart(fig)

with tab3:
    st.header("Model Explanation")
    
    if algorithm == "Linear Regression":
        st.write("""
        Linear Regression finds the best linear relationship between the input features and the target variable (IMDB Rating).
        It assumes that the rating can be predicted as a weighted sum of the input features.

        The equation is: IMDB Rating = w1*x1 + w2*x2 + ... + wn*xn + b

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

        For example: If (Meta_score > 80) and (Runtime > 120), then predict high rating.

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
        released_year = st.number_input("Released Year", min_value=int(data['Released_Year'].min()), max_value=int(data['Released_Year'].max()), value=2020)
        certificate = st.selectbox("Certificate", original_data['Certificate'].unique())
        runtime = st.number_input("Runtime (minutes)", min_value=0, value=120)
        genre = st.selectbox("Genre", original_data['Genre'].unique())
    
    with col2:
        meta_score = st.number_input("Meta Score", min_value=0, max_value=100, value=70)
        director = st.selectbox("Director", original_data['Director'].unique())
        no_of_votes = st.number_input("Number of Votes", min_value=0, value=100000)

    if st.button("🎬 Predict Rating", key="predict_button"):
        # Create a DataFrame with all features
        input_data = pd.DataFrame({
            'Released_Year': [released_year],
            'Certificate': [certificate],
            'Runtime': [runtime],
            'Genre': [genre],
            'Meta_score': [meta_score],
            'Director': [director],
            'No_of_Votes': [no_of_votes]
        })
        
        prediction = model.predict(input_data)
        st.success(f"Predicted IMDB Rating: {prediction[0]:.2f}")

# Run the Streamlit app
if __name__ == "__main__":
    st.sidebar.info("This app predicts IMDB movie ratings based on various features.")
    st.sidebar.warning("Note: This is a demonstration model and may not reflect actual IMDB ratings.")

    # Add some additional information or insights
    st.sidebar.subheader("Did you know?")
    st.sidebar.write("The IMDB rating system is based on a 10-star scale.")
    st.sidebar.write("Factors like genre, director, and meta score can significantly influence a movie's rating.")

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
        Developed by Venugopal Adep | Data source: https://www.kaggle.com/datasets/adepvenugopal/imdb-top
    </div>
    """, unsafe_allow_html=True)