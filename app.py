import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Streamlit Page Config
st.set_page_config(page_title="Employee Salary Prediction", layout="wide")

# Title
st.title("👩‍💼 Employee Salary Prediction Dashboard")
st.markdown("Upload a CSV file to train machine learning models and classify employee income levels.")

# Sidebar file uploader
uploaded_file = st.sidebar.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    # Clean column names
    df.columns = df.columns.str.replace("-", "_").str.lower()

    # Check if 'income' column exists
    if 'income' not in df.columns:
        st.error("❌ 'income' column not found in dataset.")
    else:
        # Strip whitespace and clean missing data
        df['income'] = df['income'].str.strip()
        df.replace("?", np.nan, inplace=True)
        df.dropna(inplace=True)

        # Label encoding for categorical variables
        label_encoders = {}
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        # Split features and target
        X = df.drop("income", axis=1)
        y = df["income"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Sidebar model selection
        st.sidebar.subheader("🧠 Choose ML Model")
        model_name = st.sidebar.selectbox("Model", ["Random Forest", "Logistic Regression", "Decision Tree", "SVM"])

        if model_name == "Random Forest":
            model = RandomForestClassifier()
        elif model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier()
        else:
            model = SVC()

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Results
        st.subheader(f"📈 Results: {model_name}")
        st.text(classification_report(y_test, y_pred))

        st.metric(label="✅ Accuracy", value=f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.dataframe(pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]))

        # Optional: Income distribution chart
        st.subheader("📌 Income Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='income', data=df, ax=ax)
        st.pyplot(fig)
else:
    st.info("📥 Upload a CSV file to get started.")
