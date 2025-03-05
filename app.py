import streamlit as st 
import pandas as pd 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

model =tf.keras.models.load_model('models/my_model.h5')

scaler = StandardScaler()
encoder = LabelEncoder()


st.image("../GeziraPal/web/assets/Images/pal-01.png",width=100)
st.title('GeziraPal Crop Recommender') 
st.header('About us')

# Team member information (including LinkedIn URLs)
team_members = [
    {"name": "Reem Khogaly", "linkedin_url": "https://www.linkedin.com/in/reem-m-khogaly-730a391a8/"},
    {"name": "Asmaa Farouq", "linkedin_url": "https://www.linkedin.com/in/asmaa-farouq-b41733264/"},
    {"name": "Maab Jafar", "linkedin_url": "https://www.linkedin.com/in/maab-taha/"},
    {"name": "Ahmed Alghali", "linkedin_url": "https://www.linkedin.com/in/ahmed-alghali-4997a5229/"}
]


content = """
Welcome to our innovative tool portal, showcasing a cutting-edge demo of a crop recommender system designed to assist you in predicting the  crop for cultivation. Our dedicated team, comprising:

"""

for member in team_members:
    content += f"- [{member['name']}]({member['linkedin_url']})\n" 

content += """
has poured their expertise and passion into creating this platform for your benefit. We invite you to explore, learn, and have fun!
**Explore the Code on GitHub:** [GitHub Repository](https://github.com/A7med7x7/Tesnor-Flow-SC-Fertilizer-and-Crop-Recommendation-Recomm)

"""

st.markdown(content)


st.write("This app recommed possible crops based on district, fertilizer, and soil using basic data filtering.")
st.header('basic recommender')


dataset=pd.read_csv('../data/crop_and_fertilizer.csv') #set to change 


def predict_possible_crops(district, fertilizer, soil):
    filtered_data = dataset[(dataset['District_string'] == district) &
                            (dataset['Fertilizer'] == fertilizer) &
                            (dataset['soil_string'] == soil)]
    possible_crops = filtered_data['Crop_string'].unique().tolist()
    return "Possible crops:", possible_crops

district_options = ['Khartoum', 'ALfashir', 'Algazira', 'Shendi', 'Niyala']
soil_color_options = ['Black', 'Red', 'Medium Brown', 'Dark Brown', 'Light Brown', 'Reddish Brown']
fertilizer_options = ['Urea', 'DAP', 'MOP', '10:26:26 NPK', 'SSP', 'Magnesium Sulphate',
                      '13:32:26 NPK', '12:32:16 NPK', '50:26:26 NPK', '19:19:19 NPK',
                      'Chilated Micronutrient', '18:46:00 NPK', 'Sulphur',
                      '20:20:20 NPK', 'Ammonium Sulphate', 'Ferrous Sulphate',
                      'White Potash', '10:10:10 NPK', 'Hydrated Lime']

district_basic = st.selectbox('Select your district ', district_options)
fertilizer_basic = st.selectbox("Enter fertilizer ", fertilizer_options)
soil_basic = st.selectbox("Your soil color ", soil_color_options)


crop = predict_possible_crops(district_basic, fertilizer_basic, soil_basic)

if st.button("Recommend Possible Crops Basic Recommender"):
    if len(crop[1]) >= 1:
        st.success(f'Recommended Crop: {crop}')
    else:
        st.warning("No recommendation available based on current selections. Try different combinations of district, fertilizer, and soil type.")


st.header('AI recommender')

district = st.selectbox('Select your district', district_options)
soil_color = st.selectbox('Select soil color', soil_color_options)
nitrogen = st.number_input('nitrogen level (min=20,max=150)',min_value=20, max_value=150)
phosphorus = st.number_input('phosphorus level, min=10, max=90',min_value=10, max_value=90)
potassium = st.number_input('potassium, min=5,max=150',min_value=5,max_value=150)
ph = st.slider('pH level', min_value=0.5, max_value=8.5, value=5.0)
rainfall = st.number_input('rainfall',min_value=300, max_value=1700)
temperature = st.number_input('Temperature', min_value=10, max_value=40)


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
def corresponding_youtube_tutorial(district, soil_color, nitrogen, phosphorus, potassium, ph, rainfall, temperature):
    try:
        predicted_crop = predict(district, soil_color, nitrogen, phosphorus, potassium, ph, rainfall, temperature)
        if predicted_crop == "Invalid district" or predicted_crop == "Invalid soil color":
            return predicted_crop, None

        if predicted_crop in encoded_to_label.values():
            predicted_label = next(key for key, value in encoded_to_label.items() if value == predicted_crop)
        else:
            return "Invalid crop prediction", None
        link = dataset.loc[dataset['Crop'] == predicted_label, 'Link'].values[0]
        return predicted_crop, link
    except Exception as e:
        print(f"Error: {e}")
        return "Error occurred during prediction", None

if st.button("Recommend YouTube Tutorial"):
    predicted_crop, tutorial_link = corresponding_youtube_tutorial(district, soil_color, nitrogen, phosphorus, potassium, ph, rainfall, temperature)
    if tutorial_link:
        st.markdown(f"Here is the recommended tutorial: [{predicted_crop} Tutorial]({tutorial_link})")
    else:
        st.error("Error: No tutorial found for the recommended crop.")
