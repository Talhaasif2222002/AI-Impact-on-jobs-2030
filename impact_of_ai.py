import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# --- AUTOMATIC DATA LOADING CONFIG ---
# Replace with the details from your GitHub screen!
USERNAME = "YOUR_USERNAME" 
REPO = "YOUR_REPO"
FILE_NAME = "AI_Impact_on_Jobs_2030.csv"

# Construct the Raw URL
GITHUB_URL = f"https://raw.githubusercontent.com/Talhaasif2222002/AI-Impact-on-jobs-2030/main/AI_Impact_on_Jobs_2030"

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
st.set_page_config(page_title="AI Impact Analysis", layout="wide")
st.title("🤖 AI Impact on Jobs 2030")

df = get_data_automatically(GITHUB_URL)

if df is not None:
    st.success("Data loaded automatically!")
    
    # 3. Quick Preprocessing
    target = 'Risk_Category'
    # Drop rows without target
    df = df.dropna(subset=[target])
    
    # Simple Features (drop text columns)
    X = df.select_dtypes(include=[np.number])
    y = LabelEncoder().fit_transform(df[target])

    # 4. Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    
    # 5. Dashboard Visuals
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Accuracy", f"{accuracy_score(y_test, model.predict(X_test)):.2%}")
    
    with col2:
        st.write("### Data Preview")
        st.dataframe(df.head(5))

    # Feature Importance Plot
    st.write("### Top Factors Driving AI Risk")
    fig, ax = plt.subplots()
    pd.Series(model.feature_importances_, index=X.columns).nlargest(10).plot(kind='barh', ax=ax)
    st.pyplot(fig)
