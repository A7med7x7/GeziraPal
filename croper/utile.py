"""def predict_crop_s(district, soil_color, nitrogen, phosphorus, potassium, ph, rainfall, temperature):
    encoded_district = encoder.fit_transform([district])[0]
    encoded_soil_color = encoder.fit_transform([soil_color])[0]

    features = scaler.transform([[encoded_district,encoded_soil_color, nitrogen, phosphorus, potassium, ph, rainfall, temperature]])
    prediction = model.predict(features)
    predicted_class_index = prediction.argmax(axis=1)[0]
    #predicted_crop = encoder.inverse_transform([predicted_class_index])[0]


    return predicted_class_index


def decode_label(encoded_value):
  return encoded_to_label.get(encoded_value, "Label not found")



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
    15: 'Wheat'}



# Example:
district = 'Khartoum'
soil_color = 'Black'
nitrogen = 75
phosphorus = 50
potassium = 100
ph = 6.5
rainfall = 100
temperature = 20

crop = predict_crop_s(district,soil_color, nitrogen, phosphorus, potassium, ph, rainfall, temperature)
label = decode_label(crop)
print("Predicted Crop:", label)"""

