
![salah-1](https://github.com/A7med7x7/Tesnor-Flow-SC-Fertilizer-and-Crop-Recommendation-Recomm/assets/95593096/da0fc393-61ab-43b0-9bc6-060f6b9386fd)

# Tesnor-Flow-SC-Fertilizer-and-Crop-Recommendation-system
GeziraPal is a project that aimed at assisting individuals with knowledge sharing in agriculture & crop cultivation. The project utilizes TensorFlow, a popular machine learning framework, to develop a recommendation system for suggesting suitable crops to cultivate based on various factors. and deployed in streamlit 

### How to Use the Code

#### Loading and Installing Dependencies

Before running the application, ensure you have installed the necessary dependencies. You can find all the required dependencies along with their corresponding versions in the `requirements.txt` file.

```bash
pip install -r requirements.txt
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn import *
import tensorflow as tf
import tensorflow.keras as keras
import streamlit as st
```

### To use GeziraPal, follow these steps:

1/ Clone the repository to your local machine using the following command:

git clone https://github.com/your-username/Tesnor-Flow-SC-Fertilizer-and-Crop-Recommendation-Recomm.git

2/ Install the necessary dependencies listed in the requirements.txt file using:

pip install -r requirements.txt

3/ Run the Streamlit web application by executing:

streamlit run app.py

4/ Access the web interface by opening a browser and navigating to the provided URL.

5/ Choose between the Basic Recommender and AI Recommender options based on your preference.

6/ Input the relevant parameters such as district, soil color, nutrient levels, and weather conditions.

7/ Click the "Predict" button to receive crop recommendations.

8/ Optionally, explore the provided YouTube tutorial recommendations for further learning.

### Usage Details
NumPy & Pandas: Use these libraries for numerical computations and data manipulation within your code.
Joblib: Use it as an alternative for loading transformers, encoders, and scalers saved during preprocessing or model training.
Sci-kit learn: Utilize this library for machine learning algorithms and additional functionalities.
TensorFlow: Employ TensorFlow for machine learning model development and deployment.
TensorFlow.keras: Specifically used for loading models created with Keras.
Streamlit: Utilized for streamlining the web application development process.


### About the Data

#### Source
The dataset was originally obtained from Kaggle: [Crop and Fertilizer Dataset for Western Maharashtra](https://www.kaggle.com/datasets/sanchitagholap/crop-and-fertilizer-dataset-for-westernmaharashtra).

#### Overview
- The dataset is generated and contains 11 features with more than 4k instances.
- After analysis and data generation, it now has over 12k instances.
- The target variable is the crop type.
- In this repository, you will have access to the cleaned version of the dataset.
  1. `crop_and_fertilizer.csv`: The full dataset.
  2. `x_train_resampled.csv`: The training variables.
  3. `y_train_resampled.csv`: Target variable.

### Notebook Contents

The notebook consists of:

1. Data reading and cleaning.
2. Data preprocessing.
3. Modeling.
4. Evaluation and systemizing functions.

### String Value Input Function

In order to enable users to input string values instead of labeled numbers to represent their information, a function is built using transformers (scaler and encoder).

### Prediction Functions in the App

The app contains 3 prediction functions:

1. **Basic Recommendation Function**:
   - It's a filtering method that uses only Pandas as a tool without any prediction capabilities or modeling.
   - It returns the available crops to grow from the database that match your inputs.

2. **Advanced Recommendation Function**:
   - Using TensorFlow, we build a multi-class classification Keras model that is able to recommend the best crop.

3. **Corresponding YouTube Tutorial Function**:
   - This function returns the predicted label, converts it, and then searches in the corresponding YouTube tutorial from the 'link' column to suggest what to watch.

