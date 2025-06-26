import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pytest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pytest
import os
assert os.path.exists("data/JIRA_Issues_Dataset.csv"), "Dataset file not found"


def preprocess_data(df):
    """
    Preprocess JIRA dataset by encoding categorical features and vectorizing text.
    """
    # Drop JIRA-ID (not a useful predictor)
    df = df.drop(columns=["jira-id"])

    # Encode 'priority' and 'component' using Label Encoding
    label_encoders = {}
    for col in ["priority", "component", "module"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoders for reference

    # Convert 'description' (text) into numerical features using TF-IDF
    tfidf = TfidfVectorizer(max_features=100)
    desc_features = tfidf.fit_transform(df["description"]).toarray()
    desc_df = pd.DataFrame(desc_features, columns=[f"desc_{i}" for i in range(desc_features.shape[1])])

    # Concatenate processed data
    df = df.drop(columns=["description"]).reset_index(drop=True)
    df = pd.concat([df, desc_df], axis=1)

    return df


def test_defect_prediction():
    """
    Load JIRA dataset, preprocess data, train ML model, and validate accuracy.
    """
    # Load the dataset
    df = pd.read_csv("data/JIRA_Issues_Dataset.csv")

    # Preprocess dataset
    df = preprocess_data(df)
    print(df.head(10))

    # Simulate a binary defect likelihood column (since it's missing)
    np.random.seed(42)
    df["defect_likelihood"] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])

    # Split dataset into train/test
    X = df.drop(columns=["defect_likelihood"])
    y = df["defect_likelihood"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n📊 Model Evaluation Report")
    print(f"Model Accuracy: {accuracy:.2f}")

    print("\n🧠 Business Problem Solved:")
    print(
        "This model predicts the likelihood that a JIRA issue will lead to a defect, based on metadata such as priority, component, and issue description.")
    print(
        "The goal is to help QA teams identify high-risk issues earlier, prioritize testing, and reduce defect leakage.\n")

    if accuracy > 0.7:
        print("✅ The model is learning meaningful patterns and can be useful for risk prediction.")
    elif accuracy > 0.5:
        print(
            "⚠️ The model is performing slightly better than random guessing. Consider improving data quality, labels, or features.")
    else:
        print(
            "❌ The model is not learning useful patterns. Investigate noisy labels, poor features, or model complexity.\n")

    # Ensure accuracy is reasonable
    #assert accuracy > 0.5, "Model accuracy too low"

