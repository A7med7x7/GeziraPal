import streamlit as st 
import pandas as pd 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder

import pickle
model =tf.keras.models.load_model('/Users/ahmed/Desktop/Python /my_model (1).h5')

scaler = StandardScaler()
encoder = LabelEncoder()

with open("/Users/ahmed/Desktop/Python /html & css/style.css", "r") as file:
    css = file.read()

st.image("/Users/ahmed/Desktop/Python /Images/pal-01.png",width=100)
st.title('GeziraPal Crop :blue[Recommender]', ':sunglasses:') 
st.header('About us', divider='rainbow')

st.markdown('''
    :orange[Welcome to our innovative tool portal, showcasing a 
    cutting-edge demo of a crop recommender system designed to assist you in
    predicting the optimal crop for cultivation. Our dedicated team, comprising
    Reem Khogaly, Asmaa Farouq, Maab Jafar, and Ahmed Alghali, has poured their expertise
    and passion into creating this platform for your benefit. We invite you to explore, learn,] :green[Have Fun!] .''')

st.divider()

district_options = ['Khartoum', 'ALfashir', 'Algazira', 'Shendi', 'Niyala']
soil_color_options = ['Black', 'Red', 'Medium Brown', 'Dark Brown', 'Light Brown', 'Reddish Brown']

district = st.selectbox('Select your district', district_options)
soil_color = st.selectbox('Select soil color', soil_color_options)
nitrogen = st.number_input('nitrogen level (min=20,max=150)',min_value=10, max_value=40, value=25)
phosphorus = st.number_input('phosphorus level, min=10, max=90',min_value=10, max_value=90)
potassium = st.number_input('potassium, min=5,max=150',min_value=5,max_value=150)
ph = st.slider('pH level', min_value=0.5, max_value=8.5, value=5.0)
rainfall = st.number_input('rainfall',min_value=300, max_value=1700)
temperature = st.number_input('Temperature', min_value=10, max_value=40)


dataset=pd.read_csv('/Users/ahmed/Desktop/Python /my_data/crop_and_fertilizer.csv')

seri = pd.DataFrame(dataset['District_string'].unique())
seri['encoded'] = encoder.fit_transform(seri)
new = pd.DataFrame(dataset['Soil_color'].unique())
new['encoded_soil'] = encoder.fit_transform(new)

x = dataset[['District_Name','Soil_color','Nitrogen','Phosphorus','Potassium','pH','Rainfall','Temperature']]
transformed = scaler.fit_transform(x)

encoded_to_label = {
    0: 'Cotton',
    1: 'Ginger',
    2: 'Gram',
    3: 'Grapes',
    4: 'Groundnut',
    5: 'Jowar',
    6: 'Maize',
    7: 'Masoor',
    8: 'Moong',
    9: 'Rice',
    10: 'Soybean',
    11: 'Sugarcane',
    12: 'Tur',
    13: 'Turmeric',
    14: 'Urad',
    15: 'Wheat'
}

def decode_label(encoded_value):
    return encoded_to_label.get(encoded_value, "Label not found")
  
district_to_encoded = {
    'Khartoum': 2,
    'ALfashir': 0,
    'Algazira': 1,
    'Shendi': 4,
    'Niyala': 3
}
soil_color_to_encoded = {
    'Black': 0,
    'Red': 5,
    'Medium Brown': 3,
    'Dark Brown': 1,
    'Light Brown': 2,
    'Reddish Brown': 6
}

def predict(district, soil_color, nitrogen, phosphorus, potassium, ph, rainfall, temperature): 
    try:

        encoded_district = district_to_encoded.get(district, -1) 

       
        if encoded_district == -1:
            return "Invalid district", None


        encoded_soil_color = soil_color_to_encoded.get(soil_color, -1)  

        
        if encoded_soil_color == -1:
            return "Invalid soil color", None
            
        features = [[encoded_district, encoded_soil_color, nitrogen, phosphorus, potassium, ph, rainfall, temperature]]
        scaled_features = scaler.transform(features)


        prediction = model.predict(scaled_features)
        predicted_class_index = np.argmax(prediction)

        predicted_crop = decode_label(predicted_class_index)
        return predicted_crop
    except Exception as e:
        print(f"Error: {e}")
        return "Error occurred during prediction", None


if st.button('Predict'):
    result = predict(district, soil_color, nitrogen, phosphorus, potassium, ph, rainfall, temperature)
    if result != "Error occurred during prediction":
        st.success(f'Recommended Crop: {result}')
    else:
        st.error('Error occurred during prediction. Please check your inputs.')
        