import streamlit as st
import numpy as np
import pickle

# Load the pre-trained model
with open('Model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# App title
st.title('House Price Prediction')

# Input fields
bedrooms = st.number_input('Number of Bedrooms', min_value=0, step=1)
bathrooms = st.number_input('Number of Bathrooms', min_value=0, step=1)
location = st.number_input('Location Code', min_value=0, step=1)
size = st.number_input('Size of the House (sq ft)', min_value=0)
status = st.number_input('Status Code', min_value=0, step=1)
facing = st.number_input('Facing Code', min_value=0, step=1)
type_ = st.number_input('Type Code', min_value=0, step=1)

# Predict button
if st.button('Predict'):
    # Create input data array
    input_data = np.array([[bedrooms, bathrooms, location, size, status, facing, type_]])

    # Scale the input data
    input_df = scaler.transform(input_data)

    # Predict the price using the pre-trained model
    predicted_price = model.predict(input_df)[0]

    # Display the result
    st.write(f'Predicted House Price: {predicted_price}')

# Run the app
if __name__ == '__main__':
    st._main_run_clExplicit('streamlit run your_script_name.py', 'streamlit')
