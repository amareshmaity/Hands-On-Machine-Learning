from sklearn.tree import DecisionTreeClassifier

## Function that represent model
def train_model(criterion, max_depth, min_samples_split, min_samples_leaf, max_features, splitter, max_leaf_nodes, min_impurity_decrease):
    """Trains a Decision Tree model with given hyperparameters."""
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf, 
        max_features=max_features, 
        splitter=splitter, 
        max_leaf_nodes=max_leaf_nodes, 
        min_impurity_decrease=min_impurity_decrease,
        random_state=42
    )

    return model