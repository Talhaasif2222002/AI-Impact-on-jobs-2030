import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# --- CONFIG ---
GITHUB_URL = "https://raw.githubusercontent.com/talhaasif2222002/AI-Impact-on-jobs-2030/main/AI_Impact_on_Jobs_2030.csv"

@st.cache_data(show_spinner="Fetching data from GitHub...")
def get_data_automatically(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text))
        else:
            st.error(f"Error: Could not find file at {url}")
            return None
    except Exception as e:
        st.error(f"Failed to connect: {e}")
        return None

# --- APP START ---
st.set_page_config(page_title="AI Impact Analysis 2030", layout="wide")
st.title("🤖 AI Impact on Jobs 2030")

df = get_data_automatically(GITHUB_URL)

if df is not None:
    st.success("✅ Data loaded successfully!")
    
    # 1. Sidebar Controls
    st.sidebar.header("Dashboard Settings")
    show_raw_data = st.sidebar.checkbox("Show Raw Data Preview", value=True)
    
    # 2. Layout Columns
    top_col1, top_col2 = st.columns([1, 2])

    # --- MODELING LOGIC ---
    # We assume 'AI_Impact_Level' or 'Risk_Category' is the target. 
    # Adjust 'Risk_Level' to match your actual CSV column name.
    target_col = 'Risk_Level' 
    if target_col not in df.columns:
        # Fallback if the column name is different
        target_col = df.select_dtypes(include=['object']).columns[-1]

    # Clean data for modeling
    model_df = df.dropna()
    X = model_df.select_dtypes(include=[np.number])
    
    if not X.empty:
        y = LabelEncoder().fit_transform(model_df[target_col])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        
        with top_col1:
            st.metric("Model Prediction Accuracy", f"{acc:.2%}")
            st.write("The model predicts job risk categories based on automation scores.")

    if show_raw_data:
        with top_col2:
            st.write("### Data Preview")
            st.dataframe(df.head(10))

    st.divider()

    # --- VISUALIZATIONS ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 Correlation Heatmap")
        st.write("Understanding the relationships between numerical variables.")
        
        # Generating the Heatmap
        fig_heat, ax_heat = plt.subplots(figsize=(10, 8))
        corr = df.select_dtypes(include=[np.number]).corr()
        
        sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ax=ax_heat, fmt=".2f")
        plt.xticks(rotation=45)
        st.pyplot(fig_heat)

    with col_right:
        st.subheader("📈 Feature Importance")
        st.write("Which factors contribute most to AI displacement risk?")
        
        if not X.empty:
            fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
            feat_importances = pd.Series(model.feature_importances_, index=X.columns)
            feat_importances.nlargest(10).plot(kind='barh', ax=ax_imp, color='#3498db')
            plt.gca().invert_yaxis() # Highest importance on top
            st.pyplot(fig_imp)
            
    st.divider()
    
    # Optional: Distribution Plot
    st.subheader("Distribution of AI Impact Scores")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_col = st.selectbox("Select a metric to view distribution:", numeric_cols)
    
    fig_dist, ax_dist = plt.subplots(figsize=(12, 4))
    sns.histplot(df[selected_col], kde=True, color='green', ax=ax_dist)
    st.pyplot(fig_dist)

else:
    st.warning("Waiting for data connection...")
