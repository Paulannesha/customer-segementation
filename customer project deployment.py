# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:21:15 2024

@author: jalli
"""

# prompt: i want to deployment code for streamlit that should take input from user as age,education,relationship,income,kids,

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering

# Load the trained model (replace with your actual model loading)
# ... (Your model loading code here) ...

# Example model loading (replace with your actual model)
# Assuming you have a trained model saved as 'trained_model.pkl'
# import pickle
# with open('trained_model.pkl', 'rb') as file:
#     model = pickle.load(file)

# Function to predict cluster based on user input
def predict_cluster(age, education, relationship, income, kids):
    # Create a DataFrame with user input
    input_data = pd.DataFrame({
        'Customer_Age': [age],
        'Education': [education],
        'Marital_Status': [relationship],
        'Income': [income],
        'kids': [kids],
        # Add other necessary columns with default values if needed
        # ...
    })
    le = LabelEncoder()
    input_data['Education'] = le.fit_transform(input_data['Education'])
    input_data['Marital_Status'] = le.fit_transform(input_data['Marital_Status'])
     # Make prediction
    predicted_cluster = model.predict(input_data)
    return predicted_cluster[0]



df=pd.read_excel("C:\\Users\\jalli\\Downloads\\marketing_campaign.xlsx")
# Preprocess the input data
df['Income'] = df['Income'].fillna(df['Income'].median())
df = df.drop(columns=['Z_CostContact' , 'Z_Revenue'],axis=1)
df['Education'] = df['Education'].replace(['PhD','2n Cycle','Graduation','Master'],'Post Graduate')
df['Education'] = df['Education'].replace(['Basic'],'Under Graduate')
df['Marital_Status'] = df['Marital_Status'].replace(['Married','Together'],'Relationship')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced','Widow','Alone','YOLO','Absurd'],'Single')
df['kids'] = df['Kidhome'] + df['Teenhome']
df['Expenses'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
df['TotalAcceptedCmp'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5']
df['TotalAcceptedCmp'] = df['TotalAcceptedCmp'].apply(lambda x: 1 if x > 0 else 0)
df['Customer_Age'] = (pd.Timestamp('now').year) - df['Year_Birth']
df_old=df.copy()
col_del = ['Year_Birth','AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','NumCatalogPurchases','NumDealsPurchases','NumStorePurchases','NumWebPurchases','NumWebVisitsMonth','MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds','Kidhome','Teenhome']
df=df.drop(columns=col_del,axis=1)
import pandas as pd
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
df['Dt_Customer'] = df['Dt_Customer'].apply(lambda x: x.toordinal())



df['Education'] = le.fit_transform(df['Education'])
df['Marital_Status'] = le.fit_transform(df['Marital_Status'])
AC = AgglomerativeClustering(n_clusters=4,affinity='euclidean', linkage='single')
yhat_AC = AC.fit_predict(df)
# df['Clusters'] = yhat_AC
#Adding the Clusters feature to the orignal dataframe.
df['Clusters'] = yhat_AC




X = df.drop('Clusters', axis=1)
y = df['Clusters']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a model (e.g., Logistic Regression)
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)





# Streamlit app
st.title("Customer Segmentation Prediction")

# Get user input
age = st.number_input("Age", min_value=18, max_value=100)
education = st.selectbox("Education", ["Under Graduate", "Post Graduate"])
relationship = st.selectbox("Relationship", ["Relationship", "Single"])
income = st.number_input("Income")
kids = st.number_input("Number of Kids", min_value=0)



# Make prediction
if st.button("Predict Cluster"):
    cluster = predict_cluster(age, education, relationship, income, kids)
    st.write(f"Predicted Cluster: {cluster}")
    if cluster in [0, 1]:
      st.write("We recommend the following products:")
      st.write("1) Wines")
      st.write("2) Meat")
      st.write("3) Fish Products")
      # Add image and link for Wines
      st.image("https://www.bing.com/images/search?view=detailV2&ccid=K7he1nWu&id=C234FBB34DFC0B29D328C84850F224734220AE02&thid=OIP.K7he1nWuc18tktfyuxG8wwHaHa&mediaurl=https%3A%2F%2Fwww.rankandstyle.com%2Fmedia%2Fproducts%2F2%2F2015-apothic-inferno-red-blend-wines-on-amazon.jpg&exph=700&expw=700&q=wines+product+amazon&simid=608004655119756863&FORM=IRPRST&ck=31103DB4509030906746E8B5AC49CFB4&selectedIndex=1&itb=0&cw=1152&ch=569&ajaxhist=0&ajaxserp=0", width=200)
      st.markdown("[Buy Now](https://www.amazon.in/Vega-Rica-Non-Alcoholic-Wine/dp/B07DSCBGXW/ref=sr_1_1_sspa?dib=eyJ2IjoiMSJ9.k2dI4SRchzhMYVjfBmnRv7uM54nsN04g00-pMGSiKUFtOs-W5zUV9bg0zRxyxCMkrTiKsylk4JMg4ac90YZM6IpVKY1rnSHWkYD1NDGjNp1tGcACZYq6NvfJxRgNrms-SAxQQE4zrB2vSWbnHIcLq7vv3bu4U_q-IHuF7V_ks17ANrsNHYpTCrZV5MVdR8KfMavNrQF0C5OGjYAfzhM5X-2cmNvKNX-WSixaGEHLXoILiw-ivOYA7sFxY2hsRZCDjR_ZHk5vLhzvgjIO804lRgcn7_shOg79NTC2ytKvxjE.URiIPojaHOzk038YTZVQtOPKF-clBZgRbk8zKlo8pX8&dib_tag=se&keywords=wine&qid=1724315810&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1)")
      # Add image and link for Meat
      st.image("https://www.bing.com/images/search?view=detailV2&ccid=2WJPFSVI&id=123B85E5C1AC4258BDD4B19DBF5FBF99928A7FFA&thid=OIP.2WJPFSVI3g6GJmeJpPgHFAHaHD&mediaurl=https%3A%2F%2Fm.media-amazon.com%2Fimages%2FI%2F71nsPK88NwL._SL1298_.jpg&exph=1236&expw=1298&q=meat+product+amazon&simid=608029746318486138&form=IRPRST&ck=587D72B4812E3F946464A254EA9F7E1D&selectedindex=0&itb=0&cw=1152&ch=569&ajaxhist=0&ajaxserp=0&vt=0&sim=11", width=200)
      st.markdown("[Buy Now](https://www.amazon.com/Beyond-Meat-Plant-Ground-Gluten/dp/B07SPB8BTP)")
      # Add image and link for Fish Products
      st.image("https://www.bing.com/images/search?view=detailV2&ccid=0Krk0NRx&id=A4D65808A2F05C9D6CF3D70EC6819C10EBFA13C7&thid=OIP.0Krk0NRxPerZbXFv6wvh0wHaHk&mediaurl=https%3A%2F%2Fimages-na.ssl-images-amazon.com%2Fimages%2FI%2F91IP1D2F9HL._AC_SL1500_.jpg&exph=1500&expw=1469&q=fish+product+amazon&simid=607993964945173782&FORM=IRPRST&ck=A97418470DDE1E50CE29ED1E92FEF521&selectedIndex=0&itb=0&cw=1152&ch=569&ajaxhist=0&ajaxserp=0", width=200)
      st.markdown("[Buy Now](https://www.amazon.ca/Koller-Products-BettaTank-Gallon-Fish/dp/B07BJ99WM4)") 
