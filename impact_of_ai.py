import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# --- Streamlit UI Setup ---
st.set_page_config(page_title="AI Impact Analysis 2030", layout="wide")
st.title("🤖 AI Impact on Jobs 2030: ML Analysis")

# 1. Streamlit File Uploader (Replaces google.colab.files)
uploaded_file = st.file_uploader("Upload your dataset (AI_Impact_on_Jobs_2030.csv)", type="csv")

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")
    
    # Optional: Preview data
    if st.checkbox("Show raw data preview"):
        st.write(df.head())

    # --- 2. Configuration & Preprocessing ---
    target_column = 'Risk_Category'

    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found. Please check your CSV.")
    else:
        # Drop columns not useful for prediction
        cols_to_drop = ['Job_Title', 'Transaction_ID', 'Customer_ID', 'Name', 'Email']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        df = df.dropna(subset=[target_column])

        # Feature and Target Split
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Convert Categorical to Numeric
        X = pd.get_dummies(X, drop_first=True)

        # Train/Test Split
        X_train_raw, X_test_raw, y_train_text, y_test_text = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Encode Target
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_text)
        y_test = le.transform(y_test_text)
        class_names = le.classes_

        # Impute and Scale
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        
        X_train = imputer.fit_transform(X_train_raw)
        X_test = imputer.transform(X_test_raw)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # --- 3. Model Training ---
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        }

        results = []
        cms = {}
        rf_model = None

        st.subheader("Model Evaluation")
        with st.spinner('Training models...'):
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                results.append({'Model': name, 'Accuracy': acc})
                cms[name] = confusion_matrix(y_test, y_pred)

                if name == "Random Forest":
                    rf_model = model

        # --- 4. Visualization ---
        results_df = pd.DataFrame(results)
        
        # Display Performance Chart
        st.write("### Accuracy Comparison")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=results_df, x='Model', y='Accuracy', palette='viridis', ax=ax1)
        ax1.set_ylim(0, 1.1)
        st.pyplot(fig1)

        # Display Confusion Matrices
        st.write("### Confusion Matrices")
        fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (name, cm) in zip(axes, cms.items()):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=class_names, yticklabels=class_names)
            ax.set_title(name)
        st.pyplot(fig2)

        # Feature Importance
        if rf_model is not None:
            st.write("### Top Influencing Factors (Random Forest)")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            importances = pd.Series(rf_model.feature_importances_, index=X.columns)
            importances.nlargest(10).plot(kind='barh', color='skyblue', ax=ax3)
            plt.gca().invert_yaxis()
            st.pyplot(fig3)
