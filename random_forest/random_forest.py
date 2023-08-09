from joblib import load
import numpy as np
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def infer(data, model_path):
    model = load(model_path)
    predict = model.predict(data)
    return predict

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    test_labels = test_labels.squeeze().tolist()
    # print([item for row in test_labels for item in row])
    correct = (predictions == test_labels).sum()
    # print("correct",correct)
    accuracy = correct / len(predictions) * 100
    # print('Model Performance')
    
    return accuracy

def train_random_forest(x_train,y_train):


    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 20)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 500, num = 15)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 20]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 8]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    # pprint(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(x_train, y_train)

    print(rf_random.best_params_)

    best_random = rf_random.best_estimator_
    return best_random