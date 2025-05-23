import streamlit as st
import pandas as pd
import numpy as np
import joblib


model_path = "RF.pkl"
# Load trained model
model = joblib.load(model_path)

st.set_page_config(page_title="Airbnb NYC Predictor", layout="wide")
st.title("Advanced Airbnb Price Predictor")

tab1, tab2 = st.tabs(["Predict Price", "Info"])

with tab1:
    st.subheader("Fill in the listing details:")

    col1, col2 = st.columns(2)

    with col1:
        neighbourhood = st.selectbox("Neighbourhood Group", ['Brooklyn', 'Manhattan', 'Queens', 'Bronx', 'Staten Island'])
        room_type = st.selectbox("Room Type", ['Entire home/apt', 'Private room', 'Shared room'])

    with col2:
        min_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=2)

    # Location feature
    dist_center = st.slider("Distance to Times Square (km)", 0.1, 30.0, 3.0)

    if st.button("Predict Price"):
        input_data = pd.DataFrame([{
            'neighbourhood_group': neighbourhood,
            'room_type': room_type,
            'minimum_nights': min_nights,
            'distance_to_center_km': dist_center
        }])

        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Price: ${prediction:,.2f}")

        # Confidence interval approximation
        preds_all = np.stack([
            tree.predict(model.named_steps['preprocessor'].transform(input_data))
            for tree in model.named_steps['regressor'].estimators_
        ])
        std_dev = preds_all.std()

        st.info(f"Approx. Confidence Interval: ${prediction - 1.96*std_dev:,.2f} to ${prediction + 1.96*std_dev:,.2f}")

        st.download_button(
            "Download Prediction",
            data=input_data.assign(predicted_price=prediction).to_csv(index=False),
            file_name="airbnb_prediction.csv",
            mime="text/csv"
        )

with tab2:
    st.subheader("Model Information")
    st.markdown("""
    This app uses a machine learning model trained on NYC Airbnb listings data to predict prices.
    The features used include:
    - Neighbourhood group
    - Room type
    - Minimum nights
    - Distance to Times Square (in km)

    The model is a Random Forest Regressor that has been trained on preprocessed and scaled data.
    """)
