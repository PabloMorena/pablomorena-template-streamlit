# Your code here
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

from pickle import dump

import plotly.express as px
import streamlit as st

import warnings

from pickle import load
import streamlit as st

model = load(open("../src/random_forest_classifier_default_42.sav", "rb"))
class_dict = {
    "0": "No tiene diabetes",
    "1": "Sí tiene diabetes",
}

st.title("Modelo de predicción sobre la diabetes")

val1 = st.slider("Pregnancies", min_value = 0.0, max_value = 9.0, step = 0.1)
val2 = st.slider("Glucose", min_value = 0.0, max_value = 198.0, step = 0.1)
val3 = st.slider("SkinThickness", min_value = 0.0, max_value = 60.0, step = 0.1)
val4 = st.slider("Insulin", min_value = 0.0, max_value = 846.0, step = 0.1)
val5 = st.slider("BMI", min_value = 0.0, max_value = 67.0, step = 0.1)
val6 = st.slider("Age", min_value = 0.0, max_value = 99.0, step = 0.1)

if st.button("Predict"):
    prediction = str(model.predict([[val1, val2, val3, val4, val5, val6]])[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)