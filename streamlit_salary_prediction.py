import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return model

# Load the model
model = load_model()

# Set up the label encoders for categorical data
def encode_data(data, encoders):
    for column in encoders:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
    return data

# Collect user input
st.title("AI Job Market Salary Prediction")
st.write("Provide the job details to predict the salary.")

job_title = st.text_input("Job Title")
industry = st.text_input("Industry")
company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
location = st.text_input("Location")
ai_adoption_level = st.selectbox("AI Adoption Level", ["Low", "Medium", "High"])
automation_risk = st.selectbox("Automation Risk", ["Low", "High"])
required_skills = st.text_input("Required Skills")
remote_friendly = st.selectbox("Remote Friendly", ["Yes", "No"])
job_growth_projection = st.selectbox("Job Growth Projection", ["Growth", "Decline"])

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Job_Title': [job_title],
    'Industry': [industry],
    'Company_Size': [company_size],
    'Location': [location],
    'AI_Adoption_Level': [ai_adoption_level],
    'Automation_Risk': [automation_risk],
    'Required_Skills': [required_skills],
    'Remote_Friendly': [remote_friendly],
    'Job_Growth_Projection': [job_growth_projection]
})

# Encode the input data using the same encoding process as training
encoders = ['Job_Title', 'Industry', 'Company_Size', 'Location', 'AI_Adoption_Level', 'Automation_Risk', 'Required_Skills', 'Remote_Friendly', 'Job_Growth_Projection']
encoded_data = encode_data(input_data.copy(), encoders)

# Make predictions when the 'Predict Salary' button is clicked
if st.button('Predict Salary'):
    salary_prediction = model.predict(encoded_data)
    st.write(f"The predicted salary is: ${salary_prediction[0]:,.2f}")
