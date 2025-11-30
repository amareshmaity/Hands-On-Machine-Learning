import streamlit as st
import numpy as np
from modules.data_loader import load_data
from modules.model_trainer import train_model
from modules.visualizer import plot_decision_boundary, plot_tree_structure
from sklearn.metrics import accuracy_score

## Configure streamlit app layout
st.set_page_config(layout="wide", page_title="Decision Tree Hyperparameter Tuner")

st.title("Decision Tree Hyperparameter Tuner and Visualizer")

## Load data using module
X_train, y_train, X_test, y_test, df = load_data()
feature_names = df.columns[:2].tolist()
class_names = [str(c) for c in np.unique(y_train)]


## Side bar for controlling hyperparameters
st.sidebar.header("Hyperparameter Tuning")
criterion=st.sidebar.selectbox("Select Criterion", ['gini', 'entropy', 'log_loss'])
max_depth=st.sidebar.slider("Max Depth", 1, 50, 50) # default 5
min_samples_split=st.sidebar.slider("Min Samples Split", 2, 50, 2)
min_samples_leaf=st.sidebar.slider("Min Samples Leaf", 1, 50, 1)
max_features=st.sidebar.slider("Max Features", 1, 3, 3)
splitter=st.sidebar.selectbox("Splitter", ["best", "random"])
max_leaf_nodes=st.sidebar.slider("Max Leaf Nodes", 2, 50, 50) 
min_impurity_decrease=st.sidebar.slider("Min Impurity Decrease", 0.0, 1.0, 0.00)


## Train the model
model = train_model(criterion, max_depth, min_samples_split, min_samples_leaf, max_features, splitter, max_leaf_nodes, min_impurity_decrease)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)


## Layout for visualization
with st.container():
    st.subheader("Decision Region Visualization")
    plot_decision_boundary(model, X_train, y_train, feature_names)
    st.markdown(f"**Train Accuracy: {train_accuracy:.4f}**")
    st.markdown(f"**Test Accuracy: {test_accuracy:.4f}**")
    

with st.container():
    st.subheader("Decision Tree Structure Visualization")
    plot_tree_structure(model, feature_names, class_names)