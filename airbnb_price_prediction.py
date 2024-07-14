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
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Airbnb Price Predictor", layout="wide", page_icon="🏠")

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
    data = pd.read_csv('airbnb_prices_data.csv')
    return data

data = load_data()

# Column explanations
column_explanations = {
    "city": "City where the Airbnb is located",
    "longitude": "Longitude coordinate of the Airbnb",
    "latitude": "Latitude coordinate of the Airbnb",
    "review_scores_rating": "Overall review score for the Airbnb",
    "number_of_reviews": "Number of reviews received",
    "minimum_nights": "Minimum number of nights required for booking",
    "security_deposit": "Security deposit amount",
    "cleaning_fee": "Cleaning fee charged",
    "accommodates": "Number of guests the Airbnb can accommodate",
    "bathrooms": "Number of bathrooms",
    "bedrooms": "Number of bedrooms",
    "beds": "Number of beds",
    "property_type": "Type of property (e.g., Apartment, House)",
    "room_type": "Type of room (e.g., Entire home/apt, Private room)",
    "availability_365": "Number of days the Airbnb is available in a year",
    "host_identity_verified": "Whether the host's identity is verified (0 or 1)",
    "host_is_superhost": "Whether the host is a superhost (0 or 1)",
    "cancellation_policy": "Cancellation policy type",
    "price": "Price per night (Target Variable)"
}

# Sidebar
st.sidebar.title("🏠 Airbnb Price Predictor")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Select Regression Algorithm",
    ["Linear Regression", "Ridge Regression", "Lasso Regression", 
     "Decision Tree", "Random Forest", "Support Vector Regression"]
)

# Main content
st.title("Airbnb Price Predictor")
st.write('**Developed by : Venugopal Adep**')
st.write("Predict the price of Airbnb listings based on various features.")

# Prepare the data
@st.cache_data
def prepare_data(data):
    # Convert categorical variables to numeric
    categorical_columns = ['city', 'property_type', 'room_type', 'cancellation_policy']
    for col in categorical_columns:
        data[col] = pd.Categorical(data[col]).codes
    
    X = data.drop(['price'], axis=1)
    y = data['price']
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return X, y, imputer

X, y, imputer = prepare_data(data)

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
    feature_bivariate = st.selectbox("Select a feature for bivariate analysis with Price", 
                                     [col for col in numeric_columns if col != 'price'])
    fig_bivariate = px.scatter(data, x=feature_bivariate, y='price', color="city", hover_data=['property_type'])
    st.plotly_chart(fig_bivariate)
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = data[numeric_columns].corr()
    fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(fig_corr)
    
    # Crosstab Analysis
    st.subheader("Crosstab Analysis")
    categorical_columns = ['city', 'property_type', 'room_type', 'cancellation_policy']
    cat_col1, cat_col2 = st.columns(2)
    with cat_col1:
        cat_feature1 = st.selectbox("Select first categorical feature", categorical_columns)
    with cat_col2:
        cat_feature2 = st.selectbox("Select second categorical feature", 
                                    [col for col in categorical_columns if col != cat_feature1])
    crosstab = pd.crosstab(data[cat_feature1], data[cat_feature2])
    st.write(crosstab)
    fig_crosstab = px.bar(crosstab, barmode='group')
    st.plotly_chart(fig_crosstab)
    
    # Word Cloud
    st.subheader("Word Cloud of Property Types")
    text = ' '.join(data['property_type'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig_wordcloud, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig_wordcloud)

with tab2:
    st.header("Model Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted Price")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                 mode='lines', name='Ideal'))
        fig.update_layout(xaxis_title="Actual Price", yaxis_title="Predicted Price")
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
        Linear Regression finds the best linear relationship between the input features and the target variable (Price).
        It assumes that the price can be predicted as a weighted sum of the input features.

        The equation is: Price = w1*x1 + w2*x2 + ... + wn*xn + b

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

        For example: If (review_scores_rating > 90) and (accommodates > 4), then predict high price.

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
        city = st.selectbox("City", data['city'].unique())
        longitude = st.number_input("Longitude", value=data['longitude'].mean())
        latitude = st.number_input("Latitude", value=data['latitude'].mean())
        review_scores_rating = st.slider("Review Score Rating", 0, 100, 90)
        number_of_reviews = st.number_input("Number of Reviews", value=0)
        minimum_nights = st.number_input("Minimum Nights", value=1)
        security_deposit = st.number_input("Security Deposit", value=0)
        cleaning_fee = st.number_input("Cleaning Fee", value=0)
        accommodates = st.number_input("Accommodates", value=2)
    
    with col2:
        bathrooms = st.number_input("Bathrooms", value=1.0, step=0.5)
        bedrooms = st.number_input("Bedrooms", value=1)
        beds = st.number_input("Beds", value=1)
        property_type = st.selectbox("Property Type", data['property_type'].unique())
        room_type = st.selectbox("Room Type", data['room_type'].unique())
        availability_365 = st.slider("Availability (days/year)", 0, 365, 365)
        host_identity_verified = st.selectbox("Host Identity Verified", [0, 1])
        host_is_superhost = st.selectbox("Host is Superhost", [0, 1])
        cancellation_policy = st.selectbox("Cancellation Policy", data['cancellation_policy'].unique())

    if st.button("🏠 Predict Price", key="predict_button"):
        # Create a DataFrame with all features, initialized with zeros
        input_data = pd.DataFrame(0, index=[0], columns=X.columns)
        
        # Fill in the values for the features we have
        input_data['city'] = data['city'].unique().tolist().index(city)
        input_data['longitude'] = longitude
        input_data['latitude'] = latitude
        input_data['review_scores_rating'] = review_scores_rating
        input_data['number_of_reviews'] = number_of_reviews
        input_data['minimum_nights'] = minimum_nights
        input_data['security_deposit'] = security_deposit
        input_data['cleaning_fee'] = cleaning_fee
        input_data['accommodates'] = accommodates
        input_data['bathrooms'] = bathrooms
        input_data['bedrooms'] = bedrooms
        input_data['beds'] = beds
        input_data['property_type'] = data['property_type'].unique().tolist().index(property_type)
        input_data['room_type'] = data['room_type'].unique().tolist().index(room_type)
        input_data['availability_365'] = availability_365
        input_data['host_identity_verified'] = host_identity_verified
        input_data['host_is_superhost'] = host_is_superhost
        input_data['cancellation_policy'] = data['cancellation_policy'].unique().tolist().index(cancellation_policy)
        
        # Impute any missing values in the input data
        input_data = pd.DataFrame(imputer.transform(input_data), columns=input_data.columns)
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Price: ${prediction[0]:.2f} per night")

# Run the Streamlit app
if __name__ == "__main__":
    st.sidebar.info("This app predicts Airbnb prices based on various features.")

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
        Developed by Venugopal Adep | Data source: https://www.kaggle.com/datasets/adepvenugopal/airbnb-price
    </div>
    """, unsafe_allow_html=True)
