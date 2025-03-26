import streamlit as st
import pickle
import pandas as pd
import os

# List of IPL Teams
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

# List of Host Cities
cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Load the model safely
if os.path.exists("pipe.pkl"):
    with open("pipe.pkl", "rb") as file:
        pipe = pickle.load(file)
else:
    st.error("Model file 'pipe.pkl' not found. Please check your file path.")
    st.stop()

# Streamlit App Title
st.title('🏏 IPL Win Predictor')

# Team Selection
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('🏏 Select the Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('🎯 Select the Bowling Team', sorted(teams))

# Select Host City
selected_city = st.selectbox('📍 Select Host City', sorted(cities))

# Input for Target Score
target = st.number_input('🎯 Target Score', min_value=1, step=1)

# Score, Overs, Wickets Columns
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('🏏 Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('⏳ Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets_out = st.number_input('❌ Wickets Lost', min_value=0, max_value=10, step=1)

# Prediction Button
if st.button('🔮 Predict Probability'):
    runs_left = target - score
    balls_left = int(120 - (overs * 6))  # Convert overs to balls
    wickets_left = 10 - wickets_out

    # Handle division by zero safely
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # Create input DataFrame
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Predict probabilities
    result = pipe.predict_proba(input_df)
    loss_prob = result[0][0] * 100
    win_prob = result[0][1] * 100

    # Display results
    st.header(f"🏆 {batting_team} - {round(win_prob, 1)}%")
    st.header(f"⚡ {bowling_team} - {round(loss_prob, 1)}%")
