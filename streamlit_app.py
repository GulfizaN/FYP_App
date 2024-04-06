import streamlit as st
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a K-Nearest Neighbors (KNN) model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Streamlit app starts here
st.title('Iris Flower Species Prediction')

# User input features
st.sidebar.header('Input Features')
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    return np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

input_features = user_input_features()

# Predict
prediction = model.predict(input_features)
prediction_proba = model.predict_proba(input_features)

# Display results
st.subheader('Prediction')
st.write(f"Species: {iris.target_names[prediction][0]}")
st.subheader('Prediction Probability')
st.write(prediction_proba)
