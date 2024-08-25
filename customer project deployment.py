import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the customer data
data = pd.read_excel("marketing_campaign1.xlsx")
data = data.dropna()
data.isnull().sum()

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa; /* Light blue-grey background color */
        color: #333333; /* Dark grey font color */
        font-family: 'Arial', sans-serif; /* Default font style */
    }
    .stButton button {
        background-color: #4CAF50; /* Green background color for buttons */
        color: white; /* White text on buttons */
        border-radius: 5px; /* Rounded corners for buttons */
        border: none; /* No border */
        font-size: 16px; /* Larger font size */
        padding: 10px 20px; /* Padding around the button */
        cursor: pointer; /* Pointer cursor on hover */
    }
    .stHeader {
        font-family: 'Courier New', Courier, monospace; /* Monospace font for headers */
        color: #4B0082; /* Indigo font color for headers */
        text-align: center; /* Center alignment for headers */
    }
    .stSubheader {
        font-family: 'Comic Sans MS', 'Comic Sans', cursive; /* Fun, casual font style for subheaders */
        color: #FF6347; /* Tomato red font color for subheaders */
    }
    .stMarkdown h3 {
        color: #20B2AA; /* Light sea green color for markdown headers */
    }
    </style>
""", unsafe_allow_html=True)
# Preprocess the data
data['Age'] = 2015 - data['Year_Birth']
data['Spent'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']
data['Living_With'] = data['Marital_Status'].replace({'Married': 'Partner', 'Together': 'Partner', 'Absurd': 'Alone', 'Widow': 'Alone', 
                                                      'YOLO': 'Alone', 'Divorced': 'Alone', 'Single': 'Alone'})
data['Children'] = data['Kidhome'] + data['Teenhome']
data['Family_Size'] = data['Living_With'].replace({'Alone': 1, 'Partner': 2}) + data['Children']
data['Is_Parent'] = np.where(data.Children > 0, 1, 0)

# Features used for clustering
features = data[['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                 'MntSweetProducts', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response', 'Age', 'Spent', 'Children', 
                 'Family_Size', 'Is_Parent']]

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train the KMeans clustering model
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(features_scaled)

# Save the model for later use
joblib.dump(kmeans, 'model.joblib')

# Define cluster descriptions
cluster_descriptions = {0: {"title": "Younger Families with One Young Child","messaging": "Emphasize products or services that support early childhood development, parenting hacks, and convenience.","channels": "Social media platforms like Instagram and Facebook.","promotions": "Bundle deals for family-friendly products. Time-sensitive discounts.","content_ideas": "Share tips on balancing work and parenting. Create engaging content like DIY projects for toddlers."},1: {"title": "Older Parents with Teenagers", "messaging": "Focus on products and services that cater to the needs of teenagers and their parents.","channels": "Facebook and LinkedIn.","promotions": "Offers on educational tools, extracurricular activities, or family trips.","content_ideas": "Blog posts or webinars on managing teenage behavior or preparing for college."}, 2: {"title": "High-Income, Child-Free Couples and Singles", "messaging": "Emphasize luxury, exclusivity, and experiences over products.","channels": "Premium platforms like LinkedIn and Instagram.","promotions": "Exclusive membership offers, luxury product lines, or premium service tiers.", "content_ideas": "Showcase high-end experiences, such as luxury travel, gourmet food, and exclusive events."},3: {"title": "Older Parents with Larger Families","messaging": "Focus on products and services that cater to the needs of a larger family.", "channels": "Facebook, family-oriented community events, or local advertising.","promotions": "Family packs or bulk discounts on essential products.","content_ideas": "Content that promotes family activities and togetherness." }}

# Streamlit app layout
st.title("Customer Segmentation and Product Recommendation System")

# Sidebar for customer details input
st.sidebar.header("Input Customer Details")

# Select a customer or input details manually
selected_customer_id = st.sidebar.selectbox("Select Customer ID", data.index)
input_type = st.sidebar.radio("Input Type", ["Use Selected Customer", "Manual Input"])

if input_type == "Use Selected Customer":
    customer = data.loc[selected_customer_id]
else:
    # Manual input for new customer
    customer = {"Income": st.sidebar.slider("Income", min_value=int(data["Income"].min()), max_value=int(data["Income"].max()), step=1000),
        "Kidhome": st.sidebar.slider("Number of Young Children at Home", min_value=0, max_value=3, step=1),
        "Teenhome": st.sidebar.slider("Number of Teenagers at Home", min_value=0, max_value=3, step=1),
        "Recency": st.sidebar.slider("Recency of Last Purchase", min_value=0, max_value=100, step=1),
        "MntWines": st.sidebar.slider("Amount Spent on Wines", min_value=0, max_value=1000, step=10),
        "MntFruits": st.sidebar.slider("Amount Spent on Fruits", min_value=0, max_value=1000, step=10),
        "MntMeatProducts": st.sidebar.slider("Amount Spent on Meat Products", min_value=0, max_value=1000, step=10),
        "MntFishProducts": st.sidebar.slider("Amount Spent on Fish Products", min_value=0, max_value=1000, step=10),
        "MntSweetProducts": st.sidebar.slider("Amount Spent on Sweet Products", min_value=0, max_value=1000, step=10),
        "AcceptedCmp1": st.sidebar.selectbox("Accepted Campaign 1", [0, 1]),
        "AcceptedCmp2": st.sidebar.selectbox("Accepted Campaign 2", [0, 1]),
        "Complain": st.sidebar.selectbox("Has Complain", [0, 1]),
        "Response": st.sidebar.selectbox("Response", [0, 1]),
        "Age": st.sidebar.slider("Age", min_value=18, max_value=100, step=1),
        "Spent": st.sidebar.slider("Total Spent", min_value=0, max_value=10000, step=100),
        "Children": st.sidebar.slider("Number of Children", min_value=0, max_value=5, step=1),
        "Family_Size": st.sidebar.slider("Family Size", min_value=1, max_value=10, step=1),
        "Is_Parent": st.sidebar.selectbox("Is Parent", [0, 1])}
    customer = pd.Series(customer)

# Button to predict the cluster
if st.sidebar.button("Predict Cluster"):
    # Preprocess and predict cluster
    features = customer[['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
                         'MntFishProducts', 'MntSweetProducts', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response', 
                         'Age', 'Spent', 'Children', 'Family_Size', 'Is_Parent']]

    features_scaled = scaler.transform([features])
    cluster = kmeans.predict(features_scaled)[0]

    # Display cluster information and recommendations
    st.header(f"Predicted Customer Cluster: {cluster}")
    st.subheader(f"{cluster_descriptions[cluster]['title']}")
    st.write(f"**Messaging:** {cluster_descriptions[cluster]['messaging']}")
    st.write(f"**Channels:** {cluster_descriptions[cluster]['channels']}")
    st.write(f"**Promotions:** {cluster_descriptions[cluster]['promotions']}")
    st.write(f"**Content Ideas:** {cluster_descriptions[cluster]['content_ideas']}")

    
    
    # Show the customer distribution in the clusters
    st.write("### Cluster Visualization")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Income', y='Spent', hue='Cluster', data=data, palette='viridis', s=100)
    plt.title("Customer Clusters Visualization")
    plt.xlabel("Income")
    plt.ylabel("Total Spent")
    st.pyplot(plt)

