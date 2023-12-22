import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tkinter as tk
from tkinter import *
import joblib

# Ignore future warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the dataset
df = pd.read_csv("/Users/shubham/Desktop/ChurnProject/Churn_Modelling.csv")

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

# Make a prediction using the loaded model
prediction = model.predict([[619, 42, 2, 0.0, 0, 0, 0, 101348.88, 0, 0, 0]])
print(prediction)

# Create a tkinter window
root = Tk()
root.title("Customer Churn Predictor")
root.geometry("500x450")

# Function to predict churn and display the result
def show_entry_fields():
    # Retrieve user input from GUI elements
    p1=int (e1.get () ) 
    p2=int (e2.get () ) 
    p3=int (e3.get () )
    p4=float(e4.get () )
    p5=int (e5.get () ) 
    p6=int (e6.get () ) 
    p7=int (e7.get () )
    p8=float(e8.get () ) 
    p9=int(e9.get () )
    p10=int(e10.get () )
    if p9 == 1:
        Geography_Germany=1
        Geography_Spain=0
        Geography_France=0
    elif p9==2:
        Geography_Germany=0
        Geography_Spain=1
        Geography_France=0
    elif p9==3:
        Geography_Germany=0
        Geography_Spain=0
        Geography_France=1
 
    # model = joblib.load ('churn model')
    # ... (Retrieve other inputs)
    
    # Use the loaded model to make predictions
    model = joblib.load('churn_predict_model')
    result = model.predict(sc.transform([[p1, p2, p3, p4, p5, p6, p7, p8, Geography_Germany, Geography_Spain, p10]]))
    
    # Display the prediction result in the GUI
    prediction_label.config(text=f"Churn Prediction: {result[0]}")

# Create input fields and labels
e1 = Entry(root)
e2 = Entry(root)
e3 = Entry(root)
e4 = Entry(root)
e5 = Entry(root)
e6 = Entry(root)
e7 = Entry(root)
e8 = Entry(root)
e9 = Entry(root)
e10 = Entry(root)
# ... (Create other input fields and labels)

# Create a button for prediction
predict_button = Button(root, text='Predict', command=show_entry_fields)


# Place input fields, labels, and buttons in the window using grid or pack as needed
padding_value = 5

e1.grid(row=1, column=1, padx=padding_value, pady=padding_value, sticky="w")
e2.grid(row=2, column=1, padx=padding_value, pady=padding_value, sticky="w")
e3.grid(row=3, column=1, padx=padding_value, pady=padding_value, sticky="w")
e4.grid(row=4, column=1, padx=padding_value, pady=padding_value, sticky="w")
e5.grid(row=5, column=1, padx=padding_value, pady=padding_value, sticky="w")
e6.grid(row=6, column=1, padx=padding_value, pady=padding_value, sticky="w")
e7.grid(row=7, column=1, padx=padding_value, pady=padding_value, sticky="w")
e8.grid(row=8, column=1, padx=padding_value, pady=padding_value, sticky="w")
e9.grid(row=9, column=1, padx=padding_value, pady=padding_value, sticky="w")
e10.grid(row=10, column=1, padx=padding_value, pady=padding_value, sticky="w")



label1 = Label(root, text="Credit Score")
label1.grid(row=1, column=2, padx=padding_value, pady=padding_value, sticky="w")

label2 = Label(root, text="Age of the Coustomer")
label2.grid(row=2, column=2, padx=padding_value, pady=padding_value, sticky="w")

label3 = Label(root, text="Tenure: Working Time Period ")
label3.grid(row=3, column=2, padx=padding_value, pady=padding_value, sticky="w")

label4 = Label(root, text="Balance in the Account")
label4.grid(row=4, column=2, padx=padding_value, pady=padding_value, sticky="w")

label5 = Label(root, text="Num of Products")
label5.grid(row=5, column=2, padx=padding_value, pady=padding_value, sticky="w")

label6 = Label(root, text="Has Credit Card: 1 for Y, 0 for N")
label6.grid(row=6, column=2, padx=padding_value, pady=padding_value, sticky="w")

label7 = Label(root, text="Is Active Member: 1 for Y, 0 for N")
label7.grid(row=7, column=2, padx=padding_value, pady=padding_value, sticky="w")

label8 = Label(root, text="Estimates Salary")
label8.grid(row=8, column=2, padx=padding_value, pady=padding_value, sticky="w")

label9 = Label(root, text="Geography: 1 for G, 2 for S, 3 for F")#use 1 for germany 2 for spain and 3 for france
label9.grid(row=9, column=2, padx=padding_value, pady=padding_value, sticky="w")

label10 = Label(root, text="Gender: 1 for M, 2 for F")
label10.grid(row=10, column=2, padx=padding_value, pady=padding_value, sticky="w")
# ... (Place other input fields and labels)
predict_button.grid(row=11, column=1)

# Create a label to display the prediction result
prediction_label = Label(root, text="Result: 1 He/She will Churn, 0 No Churn")
prediction_label.grid(row=12, column=1)

root.mainloop()
