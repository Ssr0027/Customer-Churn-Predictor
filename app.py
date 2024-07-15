import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib
import base64

# Load the dataset
df = pd.read_csv("Churn_Modelling.csv")

# Drop irrelevant columns
data = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Convert Geography text input into numeric data
data = pd.get_dummies(data, drop_first=True)

# Store dependent variables and independent variables
X = data.drop('Exited', axis=1)
Y = data['Exited']

# Balance the data using SMOTE
X_res, Y_res = SMOTE().fit_resample(X, Y)

# Split the dataset into Training & Testing data
X_train, X_test, Y_train, Y_test = train_test_split(X_res, Y_res, test_size=0.20, random_state=42)

# Scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the logistic regression model
log = LogisticRegression()
log.fit(X_train, Y_train)

# save the model 
joblib.dump(log, 'churn_predict_model')

# Load the saved model
model = joblib.load('churn_predict_model')

# Load background image and set it as page background
background_image = "background_image.jpg"
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/jpeg;base64,{background_image});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Page title and subtitle
st.title("Customer Churn Predictor")
st.subheader("Enter customer details:")

# Create input fields and labels
p1 = st.number_input("Credit Score", value=None, step=1)
p2 = st.number_input("Age of the Customer", value=None, step=1)
p3 = st.number_input("Tenure (Working Time Period)", value=None, step=1)
p4 = st.number_input("Balance in the Account ($)", value=None, step=0.01, format="%.2f")
p5 = st.number_input("Number of Products", value=None, step=1)
p6 = st.selectbox("Has Credit Card", [1, 0])
p7 = st.selectbox("Is Active Member", [1, 0])
p8 = st.number_input("Estimated Salary ($)", value=None, step=0.01, format="%.2f")
p9 = st.selectbox("Geography", ['Germany', 'Spain', 'France'])
p10 = st.selectbox("Gender", ['Male', 'Female'])

if p9 == 'Germany':
    Geography_Germany = 1
    Geography_Spain = 0
    Geography_France = 0
elif p9 == 'Spain':
    Geography_Germany = 0
    Geography_Spain = 1
    Geography_France = 0
elif p9 == 'France':
    Geography_Germany = 0
    Geography_Spain = 0
    Geography_France = 1

# Use the loaded model to make predictions
result = model.predict(sc.transform([[p1, p2, p3, p4, p5, p6, p7, p8, Geography_Germany, Geography_Spain, 1 if p10 == 'Female' else 0]]))

# Display the prediction result
st.subheader("Churn Prediction:")
if result[0] == 1:
    st.write("He/She will Churn")
else:
    st.write("No Churn")

# Footer text or additional information
st.text("© 2024 Your Company Name. All rights reserved.")
