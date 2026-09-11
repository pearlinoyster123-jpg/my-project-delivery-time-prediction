import streamlit as st
import requests

st.set_page_config(
    page_title="Delivery Time Prediction",
    page_icon="🚚",
    layout="centered"
)

st.title("🚚 Delivery Time Prediction")
st.write("Enter the delivery details below to predict delivery time.")

Order_Hour = st.number_input(
    "Order Hour",
    min_value=0,
    max_value=23,
    value=12
)

Is_Weekend = st.selectbox(
    "Is Weekend?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

Weather = st.selectbox(
    "Weather",
    ["Clear", "Rainy", "Cloudy", "Snowy"]
)

Vehicle_Type = st.selectbox(
    "Vehicle Type",
    ["Bike", "Scooter", "Car"]
)

Rider_Experience_Years = st.number_input(
    "Rider Experience (Years)",
    min_value=0.0,
    value=2.0,
    step=0.5
)

Restaurant_Load = st.selectbox(
    "Restaurant Load",
    ["Low", "Medium", "High"]
)

Preparation_Time_Min = st.number_input(
    "Preparation Time (Minutes)",
    min_value=0.0,
    value=15.0
)

Road_Distance_km = st.number_input(
    "Road Distance (km)",
    min_value=0.0,
    value=5.0
)

Traffic_Level = st.selectbox(
    "Traffic Level",
    ["Low", "Medium", "High"]
)

Average_Speed_kmph = st.number_input(
    "Average Speed (km/h)",
    min_value=0.0,
    value=30.0
)

if st.button("🚀 Predict Delivery Time"):

    API_URL = "YOUR_VERCEL_API_URL/predict"

    params = {
        "Order_Hour": Order_Hour,
        "Is_Weekend": Is_Weekend,
        "Weather": Weather,
        "Vehicle_Type": Vehicle_Type,
        "Rider_Experience_Years": Rider_Experience_Years,
        "Restaurant_Load": Restaurant_Load,
        "Preparation_Time_Min": Preparation_Time_Min,
        "Road_Distance_km": Road_Distance_km,
        "Traffic_Level": Traffic_Level,
        "Average_Speed_kmph": Average_Speed_kmph
    }

    try:
        response = requests.post(API_URL, params=params)

        if response.status_code == 200:
            result = response.json()

            st.success(
                f"Predicted Delivery Time: "
                f"{result['predicted_delivery_time_min']} minutes"
            )
        else:
            st.error(f"API Error: {response.status_code}")
            st.write(response.text)

    except Exception as e:
        st.error(f"Connection Error: {e}")