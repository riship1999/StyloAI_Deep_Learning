import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Fashion Recommender",
    page_icon="👔",
    layout="wide"
)

st.title("Fashion Recommender System")
st.write("Welcome to our AI-powered Fashion Recommendation System!")

# Sample data for demonstration
sample_data = {
    'Category': ['Shirts', 'Pants', 'Dresses', 'Shoes', 'Accessories'],
    'Popularity': [85, 72, 90, 68, 45]
}
df = pd.DataFrame(sample_data)

# Create a simple bar chart
fig = px.bar(
    df,
    x='Category',
    y='Popularity',
    title='Popular Fashion Categories',
    color='Category'
)

st.plotly_chart(fig, use_container_width=True)

# Add some interactive elements
st.sidebar.header("Preferences")
style_preference = st.sidebar.selectbox(
    "Select your style preference:",
    ["Casual", "Formal", "Sporty", "Vintage", "Trendy"]
)

price_range = st.sidebar.slider(
    "Price Range ($)",
    min_value=0,
    max_value=1000,
    value=(100, 500)
)

if st.button("Get Recommendations"):
    st.success(f"Showing recommendations for {style_preference} style in price range ${price_range[0]} - ${price_range[1]}")
    # Placeholder for recommendations
    st.info("This is a demo version. ML-powered recommendations coming soon!")
