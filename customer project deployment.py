import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# Initialize the Label Encoder and Standard Scaler
le = LabelEncoder()
scaler = StandardScaler()

st.title('Customer Segmentation')
st.sidebar.header('User Input')

def user_input_features():
    Age=st.sidebar.number_input('Insert the Age', min_value=18, max_value=100, value=30)
    Education=st.sidebar.selectbox('Graduate',('Undergraduate','Postgraduate'))
    Earning=st.sidebar.number_input('Insert the Amount', min_value=0, max_value=1000000, value=50000)
    Expenditure=st.sidebar.number_input('Insert the Amount', min_value=0, max_value=1000000, value=20000)
    Children st.sidebar.number_input('Insert number', min_value=0, max_value=10, value=0)
    
    data = {'Customer Age': Age,
            'Earning': Earning, 
            'Expenditure': Expenditure,
            'Children': Children}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input')
st.write(df)

# Encode the Education feature
df['Education'] = le.fit_transform(df['Education'])

# Scaling the features
df_scaled = scaler.fit_transform(df)

# Sidebar for selecting number of clusters
st.sidebar.header('Clustering Parameters')
n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=3)

# Generate a dummy dataset for training (replace with your actual dataset)
# Example data
data = {'Customer Age': [25, 45, 35, 50, 23, 40],'Education': [0, 1, 0, 1, 0, 1], 'Earning': [30000, 60000, 40000, 80000, 35000, 70000],'Expenditure': [15000, 20000, 12000, 25000, 10000, 22000],'Children': [1, 2, 1, 3, 0, 2]}
df_train = pd.DataFrame(data)

# Encode and scale the training data
df_train['Education'] = le.fit_transform(df_train['Education'])
df_train_scaled = scaler.fit_transform(df_train)

# Perform clustering on the training data
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_train_scaled)

# Predict the cluster for the user input
user_cluster = kmeans.predict(df_scaled)
df['Predicted Cluster'] = user_cluster

# Display the resulting cluster
st.subheader('Predicted Cluster for User Input')
st.write(f"The customer is predicted to belong to cluster: {user_cluster[0]}")