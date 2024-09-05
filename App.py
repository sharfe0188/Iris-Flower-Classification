import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = load_iris()
data = pd.read_csv("iris_data.csv")
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

# Train a simple model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit app
st.title("Iris Flower Classification")

st.sidebar.header("Input Features")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()), float(X['sepal length (cm)'].mean()))
    sepal_width = st.sidebar.slider('Sepal width (cm)', float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()), float(X['sepal width (cm)'].mean()))
    petal_length = st.sidebar.slider('Petal length (cm)', float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()), float(X['petal length (cm)'].mean()))
    petal_width = st.sidebar.slider('Petal width (cm)', float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()), float(X['petal width (cm)'].mean()))
    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input parameters')
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction][0])

st.subheader('Prediction Probability')
st.write(prediction_proba)
