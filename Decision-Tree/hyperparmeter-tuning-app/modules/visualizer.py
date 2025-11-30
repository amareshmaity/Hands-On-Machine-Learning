import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import plot_tree

import streamlit as st


## Function to display decision boundary
def plot_decision_boundary(model, X, y, feature_names):

    """Generate a plot of the decision region"""
    plt.figure(figsize=(3,3))

    disp = DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        response_method='predict',
        xlabel=feature_names[0], 
        ylabel=feature_names[1], 
        alpha=0.2, 
        cmap=plt.cm.RdYlBu
    )

    disp.ax_.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, marker="s", edgecolor="k", s=20, cmap=plt.cm.RdYlBu)
    st.pyplot(plt)


## Funciton to display decision tree
def plot_tree_structure(model, feature_names, class_names):

    """Generates a visualization of the decision tree structure."""
    plt.figure(figsize=(16,12))

    plot_tree(
        model, 
        feature_names=feature_names, 
        class_names=class_names, 
        filled=True, 
        rounded=True, 
        fontsize=10
    )
    st.pyplot(plt)