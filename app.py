%%writefile app.py
import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('house_price_model.pkl', 'rb'))

st.set_page_config(page_title="House Price Prediction", page_icon="🏡")

st.title("🏡 House Price Predictor")
st.markdown("Enter property details to estimate the price")

# Input fields
sqft = st.number_input("Square Footage", min_value=300, max_value=10000, step=10)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1.0, max_value=10.0, step=0.5)

# Predict button
if st.button("Predict Price"):
    input_data = np.array([[sqft, bedrooms, bathrooms]])
    prediction = model.predict(input_data)[0]

    # Add error margin using MAE
    mae = 36569  # Approx from previous model evaluation
    lower_bound = max(0, prediction - mae)  # Prevent negative prices
    upper_bound = prediction + mae

    # Display result
    if prediction < 0:
        st.warning("Prediction seems unrealistic for given inputs. Please enter realistic values.")
    else:
        st.success(f"Estimated Price: ${prediction:,.2f}")
        st.write(f"Estimated variation: ± ${mae:,.2f}")


