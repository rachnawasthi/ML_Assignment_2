import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(page_title="Breast Cancer ML Dashboard", layout="wide")

st.title("Breast Cancer Classification Dashboard")

st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "target" not in data.columns:
        st.error("CSV must contain a 'target' column")
    else:
        X = data.drop("target", axis=1)
        y = data["target"]

        scaler = joblib.load("models/scaler.pkl")

        model_dict = {
            "Logistic Regression": joblib.load("models/logistic_model.pkl"),
            "Decision Tree": joblib.load("models/decision_tree_model.pkl"),
            "KNN": joblib.load("models/knn_model.pkl"),
            "Naive Bayes": joblib.load("models/naive_bayes_model.pkl"),
            "Random Forest": joblib.load("models/random_forest_model.pkl"),
            "XGBoost": joblib.load("models/xgboost_model.pkl")
        }

        model = model_dict[model_choice]

        if model_choice in ["Logistic Regression", "KNN"]:
            X = scaler.transform(X)

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        st.subheader("Evaluation Metrics")
        st.write("Accuracy:", acc)
        st.write("AUC:", auc)
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("F1 Score:", f1)
        st.write("MCC:", mcc)

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        st.pyplot(fig)

else:
    st.info("Upload a test dataset to begin.")