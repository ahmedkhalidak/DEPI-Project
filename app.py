import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the pre-trained XGBoost model
xgb_model = joblib.load('models/XGBoost_model.sav')

# Create a dictionary to store the loaded models
loaded_models = {
    'XGBoost': xgb_model
}

# Decode predictions
def decode_prediction(pred):
    return 'Customer Exits' if pred == 1 else 'Customer Stays'

# Streamlit app layout
def main():
    # Set the title of the app
    st.title("Customer Churn Prediction")

    # User input section
    st.subheader("Enter User Information")
    
    # Input fields for customer data
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
    geography = st.selectbox("Country", ['France', 'Germany', 'Spain'])
    gender = st.selectbox("Gender", ['Female', 'Male'])
    age = st.number_input("Age", min_value=18, step=1)
    tenure = st.number_input("Years as Bank Customer", step=1)
    balance = st.number_input("Account Balance", step=0.01)
    num_of_products = st.selectbox("Number of Products Used", [1, 2, 3, 4, 5])
    has_credit_card = st.selectbox("Has an Active Credit Card", ['Yes', 'No'])
    is_active_member = st.selectbox("Is an Active Bank Member", ['Yes', 'No'])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=0.01)

    # Convert Yes/No responses to 1/0 for the model
    has_credit_card = 1 if has_credit_card == 'Yes' else 0
    is_active_member = 1 if is_active_member == 'Yes' else 0

    # Prediction logic on form submission
    if st.button("Submit"):
        # Prepare the input data for the model
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],  # Match column name
            'Geography': [geography],        # Match column name
            'Gender': [gender],              # Match column name
            'Age': [age],                    # Match column name
            'Tenure': [tenure],              # Match column name
            'Balance': [balance],            # Match column name
            'NumOfProducts': [num_of_products],  # Match column name
            'HasCrCard': [has_credit_card],  # Match column name
            'IsActiveMember': [is_active_member],  # Match column name
            'EstimatedSalary': [estimated_salary]  # Match column name
        })
        
        # Display the customer data entered
        st.subheader("Customer Data")
        st.write(input_data)

        # Make predictions using the loaded models
        for model_name, model in loaded_models.items():
            prediction = model.predict(input_data)[0]
            prediction_decoded = decode_prediction(prediction)
            
            # Display the model prediction result
            st.subheader(f"{model_name} Prediction")
            st.write(f"The model predicts that the customer will: **{prediction_decoded}**")

# Run the app
if __name__ == "__main__":
    main()
