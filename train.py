import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# The top 8 most important features for this dataset
SELECTED_FEATURES = [
    'Type of Travel', 
    'Class', 
    'Customer Type', 
    'Inflight wifi service', 
    'Online boarding', 
    'Inflight entertainment', 
    'Seat comfort', 
    'Leg room service'
]

def train_and_save_model():
    df = pd.read_csv('data/train.csv')
    
    # Filter dataset to only use the selected features
    X = df[SELECTED_FEATURES]
    y = df['satisfaction']
    
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = X.select_dtypes(include=['object']).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_columns),
        ('cat', categorical_pipeline, categorical_columns)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(
            criterion='gini', max_depth=15, min_samples_leaf=5, 
            min_samples_split=20, random_state=42
        ))
    ])

    print("Training streamlined model on top 8 features...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Accuracy with reduced features:", accuracy_score(y_test, y_pred))
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model_pipeline.pkl')
    print("Model saved to models/model_pipeline.pkl")

if __name__ == "__main__":
    train_and_save_model()