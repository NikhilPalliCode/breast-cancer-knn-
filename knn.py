# breast_cancer_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

def load_and_preprocess_data():
    """Load and preprocess the data"""
    df = pd.read_csv('data.csv')
    
    # Basic info
    print("Dataset shape:", df.shape)
    print("\nClass distribution:")
    print(df['diagnosis'].value_counts())
    
    return df

def create_visualizations(df):
    """Create 4 clear visualizations"""
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Breast Cancer Wisconsin Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Diagnosis Distribution
    ax1 = axes[0, 0]
    diagnosis_counts = df['diagnosis'].value_counts()
    colors = ['#2ecc71', '#e74c3c']  # Green for benign, red for malignant
    wedges, texts, autotexts = ax1.pie(diagnosis_counts.values, labels=['Benign', 'Malignant'], 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Diagnosis Distribution', fontweight='bold', pad=20)
    
    # Make labels bold
    for text in texts:
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    # 2. Feature Comparison (Radius Mean)
    ax2 = axes[0, 1]
    benign = df[df['diagnosis'] == 'B']['radius_mean']
    malignant = df[df['diagnosis'] == 'M']['radius_mean']
    
    ax2.hist(benign, alpha=0.7, label='Benign', bins=20, color='#2ecc71')
    ax2.hist(malignant, alpha=0.7, label='Malignant', bins=20, color='#e74c3c')
    ax2.set_xlabel('Radius Mean')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Radius Mean Distribution by Diagnosis', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Correlation Heatmap (Top 8 features)
    ax3 = axes[1, 0]
    # Select only numeric columns and compute correlation with diagnosis
    df_numeric = df.copy()
    df_numeric['diagnosis'] = df_numeric['diagnosis'].map({'B': 0, 'M': 1})
    
    # Get top 8 features most correlated with diagnosis
    correlation = df_numeric.corr()['diagnosis'].abs().sort_values(ascending=False)
    top_features = correlation.index[1:9]  # Skip diagnosis itself
    
    corr_matrix = df_numeric[top_features].corr()
    im = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax3.set_xticks(range(len(top_features)))
    ax3.set_yticks(range(len(top_features)))
    ax3.set_xticklabels([f.split('_')[0] for f in top_features], rotation=45, ha='right')
    ax3.set_yticklabels([f.split('_')[0] for f in top_features])
    ax3.set_title('Top Feature Correlations', fontweight='bold')
    
    # Add correlation values to heatmap
    for i in range(len(top_features)):
        for j in range(len(top_features)):
            text = ax3.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    # 4. Feature Importance from Random Forest
    ax4 = axes[1, 1]
    
    # Prepare data for model
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train simple Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y_encoded)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True).tail(10)  # Top 10 features
    
    ax4.barh(feature_importance['feature'], feature_importance['importance'], 
             color='#3498db')
    ax4.set_xlabel('Importance')
    ax4.set_title('Top 10 Most Important Features', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('breast_cancer_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def statistical_analysis(df):
    """Perform quick statistical analysis"""
    print("\n" + "="*50)
    print("STATISTICAL ANALYSIS")
    print("="*50)
    
    # Compare means of key features
    features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
    
    for feature in features:
        benign = df[df['diagnosis'] == 'B'][feature]
        malignant = df[df['diagnosis'] == 'M'][feature]
        
        print(f"\n{feature}:")
        print(f"  Benign:   Mean = {benign.mean():.2f}, Std = {benign.std():.2f}")
        print(f"  Malignant: Mean = {malignant.mean():.2f}, Std = {malignant.std():.2f}")
        print(f"  Difference: {malignant.mean() - benign.mean():.2f}")

def main():
    """Main function"""
    print("Loading and analyzing Breast Cancer Wisconsin dataset...")
    
    # Load data
    df = load_and_preprocess_data()
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df)
    
    # Perform statistical analysis
    statistical_analysis(df)
    
    print("\nAnalysis complete! Check 'breast_cancer_analysis.png' for the visualizations.")

if __name__ == "__main__":
    main()