import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_ai_impact():
    # 1. Load the dataset
    file_path = 'AI_Impact_on_Jobs_2030.csv'
    
    try:
        df = pd.read_csv(file_path)
        print("--- Dataset Successfully Loaded ---")
        print(df.head())
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the CSV is in the same folder.")
        return

    # 2. Basic Data Cleaning & Analysis
    # Assuming columns like 'Job_Title', 'AI_Impact_Score', and 'Automation_Risk' exist
    # Let's create a summary of impact by industry or job type
    print("\n--- Summary Statistics ---")
    print(df.describe())

    # 3. Visualization: Top 10 Jobs by AI Impact Score
    plt.figure(figsize=(12, 6))
    
    # Sorting values for a better chart
    # Note: Replace 'AI_Impact_Score' and 'Job_Title' with the exact column names in your CSV
    top_impact = df.sort_values(by='AI_Impact_Score', ascending=False).head(10)
    
    sns.barplot(data=top_impact, x='AI_Impact_Score', y='Job_Title', palette='viridis')
    
    plt.title('Top 10 Jobs Predicted to be Impacted by AI (2030)')
    plt.xlabel('Impact Score (0-100)')
    plt.ylabel('Job Title')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('ai_impact_chart.png')
    print("\nChart saved as 'ai_impact_chart.png'")
    plt.show()

if __name__ == "__main__":
    analyze_ai_impact()
