import streamlit as st
import pandas as pd
from joblib import load

# Load the trained Random Forest model
model = load("D:/project/wqd7001/Model/rf_model.pkl")

# Set page config
st.set_page_config(page_title="Traffic Congestion Predictor", layout="centered")
st.title("🚦 Traffic Congestion Prediction System")
st.write("This tool predicts traffic congestion level in Kuala Lumpur based on weather and time features.")

# Sidebar inputs
st.sidebar.header("Input Features")

temp = st.sidebar.slider("Temperature (°C)", 20.0, 40.0, 29.0)
humidity = st.sidebar.slider("Humidity (%)", 20.0, 100.0, 70.0)
windspeed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 15.0, 3.0)
aqi = st.sidebar.slider("Air Quality Index (AQI)", 0, 200, 60)
precip = st.sidebar.slider("Rainfall (mm)", 0.0, 10.0, 0.0)

hour = st.sidebar.selectbox("Hour of Day", list(range(0, 24)), index=8)
weekday = st.sidebar.selectbox("Day of Week (0 = Mon, 6 = Sun)", list(range(7)), index=2)

street_choice = st.sidebar.radio("Street", ("Jalan Sultan Salahuddin", "Jalan Lapangan Terbang"))
street = 1 if street_choice == "Jalan Sultan Salahuddin" else 0
is_weekend = 1 if weekday in [5, 6] else 0

# Construct input sample
sample = pd.DataFrame([{
    "temp": temp,
    "humidity": humidity,
    "windspeed": windspeed,
    "aqi": aqi,
    "precip": precip,
    "street_Jalan Sultan Salahuddin": street,
    "hour": hour,
    "weekday": weekday,
    "is_weekend": is_weekend
}])

# Predict when user clicks button
if st.button("Predict Congestion Level"):
    prediction = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]
    class_names = model.classes_

    # Display prediction result
    st.subheader("📍 Prediction Result")
    st.success(f"Predicted Congestion Level: **{prediction}**")

    # Show prediction probabilities as bar chart
    st.subheader("📊 Prediction Probabilities")
    proba_df = pd.DataFrame({
        "Congestion Level": class_names,
        "Probability": proba
    })
    st.bar_chart(proba_df.set_index("Congestion Level"))
