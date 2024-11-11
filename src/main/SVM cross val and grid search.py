import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import itertools


# load data
df = pd.read_csv('src/main/resources/StudentPerformanceFactors.csv')

# check for missing values
if df.isnull().sum().sum() > 0:
    print("Missing values found. Filling missing values...")

    # fill missing values for numerical with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # fill missing values for categorical with mode
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# convert categorical features
category_col = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Internet_Access',
                'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', 'Gender']
df = pd.get_dummies(df, columns=category_col, drop_first=True)

remaining_category_col = ['Motivation_Level', 'Family_Income', 'Teacher_Quality', 'Distance_from_Home']
df = pd.get_dummies(df, columns=remaining_category_col, drop_first=True)

# convert exam score to binary class
df['Exam_Score_Binary'] = df['Exam_Score'].apply(lambda x: 1 if x >= 65 else 0)

# separate feature and target
X = df.drop(['Exam_Score', 'Exam_Score_Binary'], axis=1)
y = df['Exam_Score_Binary']

# scale numeric columns
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# convert to numpy arrays
X = X.values
y = y.values

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# n-fold cross validation
def cross_validation(model, X, y, n_folds=5):
    fold_size = len(X) // n_folds
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    accuracy_scores = []

    for fold in range(n_folds):
        val_indices = indices[fold * fold_size: (fold + 1) * fold_size]
        train_indices = np.setdiff1d(indices, val_indices)

        # stratified class dist in train and val
        if len(np.unique(y[train_indices])) < 2 or len(np.unique(y[val_indices])) < 2:
            print(f"Skipping fold {fold + 1} due to lack of class representation.")
            continue

        X_train_fold, X_val_fold = X[train_indices], X[val_indices]
        y_train_fold, y_val_fold = y[train_indices], y[val_indices]

        # train model and evaluate on validation set
        model.fit(X_train_fold, y_train_fold)
        y_val_pred = model.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, y_val_pred)
        accuracy_scores.append(accuracy)

    return np.mean(accuracy_scores) if accuracy_scores else 0


# grid search
def grid_search_svm(X, y, param_grid, n_folds=5):
    best_params = None
    best_score = 0
    best_model = None

    # combinations of hyperparams
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"\n--- SVM Grid Search ({len(param_combinations)} combinations) ---")
    for i, params in enumerate(param_combinations):
        try:
            model = SVC(**params)
            score = cross_validation(model, X, y, n_folds)

            print(f"Combination {i + 1}/{len(param_combinations)}: {params}, Cross-Validation Accuracy: {score:.4f}")

            if score == 0:
                print(f"Warning, cross validation returned 0 for parameters {params}")

            if score > best_score:
                best_score = score
                best_params = params
                best_model = model
        except Exception as e:
            print(f"Error with parameters {params}: {e}")

    print(f"\nBest Score from Grid Search: {best_score:.4f}")
    if best_params:
        print(f"Best Params: {best_params}")

    return best_model, best_params, best_score


# param grid for SVM
svm_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
}

print("Class Distribution:")
print(df['Exam_Score_Binary'].value_counts())
# perform grid search for SVM
best_svm_model, best_svm_params, best_svm_score = grid_search_svm(X_train, y_train, svm_param_grid)


# evaluate on training
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print("\n--- Training Set Evaluation ---")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Training F1 Score: {train_f1:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

    return train_accuracy, train_f1, test_accuracy, test_f1


# train best on full training set
if best_svm_model:
    best_svm_model.fit(X_train, y_train)
    print(f"Best Cross-Validation Accuracy: {best_svm_score:.4f}")

    evaluate_model(best_svm_model, X_train, y_train, X_test, y_test)
else:
    print("No suitable model found during grid search.")

