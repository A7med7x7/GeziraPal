
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

### Usage Details
NumPy & Pandas: Use these libraries for numerical computations and data manipulation within your code.
Joblib: Use it as an alternative for loading transformers, encoders, and scalers saved during preprocessing or model training.
Sci-kit learn: Utilize this library for machine learning algorithms and additional functionalities.
TensorFlow: Employ TensorFlow for machine learning model development and deployment.
TensorFlow.keras: Specifically used for loading models created with Keras.
Streamlit: Utilized for streamlining the web application development process.
