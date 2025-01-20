def xgboost_classification(X, y):
    
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import numpy as np
    
    """
    Perform classification using XGBoost with grid search for model selection.
    
    Args:
        X: Features dataset (numpy array or pandas DataFrame).
        y: Target labels (numpy array or pandas Series).
    
    Returns:
        best_model: The best XGBoost model found by grid search.
        best_params: Best hyperparameters.
        report: Classification report for the test data.
    """
    # Step 1: Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 2: Define the model
    xgb_model = XGBClassifier(eval_metric='logloss')
    
    # Step 3: Set up hyperparameter grid
    param_grid = {
        'n_estimators': np.linspace(50, 300, 6).astype('uint8'),
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.4, 0.8, 1.0]
    }
    
    # Step 4: Perform Grid Search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    # Step 5: Evaluate the best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Best Parameters: {best_params}")
    print(f"Accuracy on Test Data: {acc}")
    print("Classification Report:")
    print(report)
    
    return best_model, best_params, report

def svmTraining(training_set_filename, cv, n_jobs, cs, xgboost_model_filename_out):
    import matplotlib as mpl
    mpl.use('Agg')  # Use Agg backend for matplotlib (no GUI)
    import matplotlib.pyplot as plt
    from sklearn import preprocessing
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    import pickle
    import numpy as np
    from sklearn.metrics import cohen_kappa_score, make_scorer
    


    # This script selects the best model for an SVM using a brute-force grid search approach 
    # exploring the space generated by gamma and C. Once the best parameters are found, 
    # the script trains an SVM that can be applied for an online prediction.
    # The saved SVM model is a dict formed as follows:
    # 'svmModel': svm
    # 'normalizer': normalizer
    # 'feature_names': feature_names
    # 'target_names': target_names

    # Load the training set from the pickle file
    training_set = pickle.load(open(training_set_filename, 'rb'), encoding='latin1')
    feature_names = training_set['feature_names']
    target_names = training_set['target_names']
    Samples_train = training_set['data']
    Labels_train = training_set['target']

    # Normalize the training samples
    print("Normalization ...")
    normalizer = preprocessing.StandardScaler().fit(Samples_train)
    Samples_train_normalized = normalizer.transform(Samples_train)

    # Grid search to find the best hyperparameters
    print("Grid Search ...")
    
    best_model, best_params, report = xgboost_classification(Samples_train_normalized, np.array(Labels_train) - 1)
    

    # Save the trained SVM model to a pickle file
    model = {'xgboostModel': best_model, 'normalizer': normalizer, 'feature_names': feature_names, 'target_names': target_names}
    pickle.dump(model, open(xgboost_model_filename_out, "wb"))
