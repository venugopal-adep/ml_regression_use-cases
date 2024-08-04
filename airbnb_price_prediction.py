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
st.set_page_config(page_title="Airbnb Price Predictor", layout="wide", page_icon="🏠")

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f0f8ff;
}
.stButton>button {
    background-color: #ff5a5f;
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
    background-color: #ff5a5f;
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
st.title("🏠 Airbnb Price Predictor")
st.markdown("**Developed by: Venugopal Adep**")
st.markdown("Explore factors influencing Airbnb prices and predict listing prices!")

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('airbnb_prices_data.csv')
    return data

data = load_data()

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
    st.header("📚 Learn About Airbnb Pricing")
    
    st.markdown("""
    <div class="highlight">
    <h3>Key Factors Influencing Airbnb Prices</h3>
    <ul>
        <li>Location: City and specific neighborhood</li>
        <li>Property Features: Type, number of rooms, amenities</li>
        <li>Capacity: Number of guests it can accommodate</li>
        <li>Reviews: Rating and number of reviews</li>
        <li>Host Status: Superhost status, identity verification</li>
        <li>Booking Policies: Minimum nights, cancellation policy</li>
        <li>Fees: Cleaning fee, security deposit</li>
        <li>Availability: Number of days available per year</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>Why These Factors Matter</h3>
    <ul>
        <li>Location: Affects demand and local market rates</li>
        <li>Property Features: Determine the overall value and appeal</li>
        <li>Capacity: Larger properties can often command higher prices</li>
        <li>Reviews: Influence guest trust and willingness to book</li>
        <li>Host Status: Can affect perceived reliability and quality</li>
        <li>Booking Policies: Impact guest flexibility and booking decisions</li>
        <li>Fees: Additional costs that affect the total price for guests</li>
        <li>Availability: Can indicate popularity and affect pricing strategy</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    <h3>Tips for Understanding Airbnb Pricing</h3>
    <ul>
        <li>Research local market rates and adjust prices accordingly</li>
        <li>Highlight unique features of your property to justify higher prices</li>
        <li>Maintain high ratings and respond to reviews to build trust</li>
        <li>Consider becoming a Superhost to potentially charge premium rates</li>
        <li>Adjust prices based on seasonality and local events</li>
        <li>Be transparent about all fees to avoid guest surprises</li>
        <li>Use dynamic pricing to optimize occupancy and revenue</li>
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
    
    # Price by City
    st.subheader("Average Price by City")
    fig_city = px.bar(data.groupby('city')['price'].mean().reset_index(), x='city', y='price')
    st.plotly_chart(fig_city, use_container_width=True)

with tab3:
    st.header("🧮 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted Price")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                 mode='lines', name='Ideal'))
        fig.update_layout(xaxis_title="Actual Price", yaxis_title="Predicted Price")
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
        city = st.selectbox("City", data['city'].unique())
        property_type = st.selectbox("Property Type", data['property_type'].unique())
        room_type = st.selectbox("Room Type", data['room_type'].unique())
        accommodates = st.number_input("Accommodates", value=2, min_value=1)
        bathrooms = st.number_input("Bathrooms", value=1.0, step=0.5)
        bedrooms = st.number_input("Bedrooms", value=1)
        beds = st.number_input("Beds", value=1)
    
    with col2:
        review_scores_rating = st.slider("Review Score Rating", 0, 100, 90)
        number_of_reviews = st.number_input("Number of Reviews", value=0)
        minimum_nights = st.number_input("Minimum Nights", value=1)
        availability_365 = st.slider("Availability (days/year)", 0, 365, 365)
        host_is_superhost = st.selectbox("Host is Superhost", [0, 1])
        cancellation_policy = st.selectbox("Cancellation Policy", data['cancellation_policy'].unique())

    if st.button("🏠 Predict Price"):
        input_data = pd.DataFrame(0, index=[0], columns=X.columns)
        
        input_data['city'] = data['city'].unique().tolist().index(city)
        input_data['property_type'] = data['property_type'].unique().tolist().index(property_type)
        input_data['room_type'] = data['room_type'].unique().tolist().index(room_type)
        input_data['accommodates'] = accommodates
        input_data['bathrooms'] = bathrooms
        input_data['bedrooms'] = bedrooms
        input_data['beds'] = beds
        input_data['review_scores_rating'] = review_scores_rating
        input_data['number_of_reviews'] = number_of_reviews
        input_data['minimum_nights'] = minimum_nights
        input_data['availability_365'] = availability_365
        input_data['host_is_superhost'] = host_is_superhost
        input_data['cancellation_policy'] = data['cancellation_policy'].unique().tolist().index(cancellation_policy)
        
        input_data = pd.DataFrame(imputer.transform(input_data), columns=input_data.columns)
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Price: ${prediction[0]:.2f} per night")
        
        # Radar chart for input features
        features = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'review_scores_rating', 'number_of_reviews', 'minimum_nights', 'availability_365']
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
            "question": "Which of these factors is typically NOT considered in Airbnb price prediction?",
            "options": [
                "Number of bedrooms",
                "Host's age",
                "Review score rating",
                "City location"
            ],
            "correct": 1,
            "explanation": "Host's age is typically not considered in Airbnb price prediction. The focus is more on property features, location, and booking-related factors."
        },
        {
            "question": "What does 'Superhost' status indicate?",
            "options": [
                "The host owns multiple properties",
                "The host has achieved high standards of hospitality",
                "The property is a luxury listing",
                "The host has been on Airbnb for over 10 years"
            ],
            "correct": 1,
            "explanation": "Superhost status indicates that the host has achieved and maintained high standards of hospitality, as recognized by Airbnb."
        },
        {
            "question": "Which of these is likely to have a strong positive influence on an Airbnb's price?",
            "options": [
                "A strict cancellation policy",
                "Being located far from the city center",
                "A high number of positive reviews",
                "Requiring a long minimum stay"
            ],
            "correct": 2,
            "explanation": "A high number of positive reviews typically has a strong positive influence on an Airbnb's price. It builds trust with potential guests and can justify higher rates."
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
st.sidebar.info("This app demonstrates the factors influencing Airbnb prices. Adjust the parameters and explore the different tabs to learn more!")

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
    Developed by Venugopal Adep | Data source: Airbnb Prices Dataset
</div>
""", unsafe_allow_html=True)

# Add some spacing at the bottom to prevent content from being hidden by the footer
st.write("<br><br><br>", unsafe_allow_html=True)
