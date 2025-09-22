"""
Exploratory Data Analysis Assignment - Wine Quality Dataset
Student: Joshua Gulizia
GitHub Username: joshuagulizia
PS: [Please add your PS number here]

This assignment explores the relationships between different wine attributes 
and their potential impact on quality using statistical analysis and visualization.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Enable interactive mode for navigation
plt.ion()

# Create a list to store all figures for navigation
figures = []
current_figure = 0

# Set matplotlib backend to ensure plots display
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility

def calculate_feature_importance(df):
    """Calculate feature importance for wine quality prediction"""
    # Prepare data
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return feature_importance

def show_plot(title, plot_func, plot_number, total_plots, *args, **kwargs):
    """Create and show a single plot with navigation"""
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Call the plot function
    plot_func(fig, *args, **kwargs)
    
    # Navigation instructions removed for cleaner display
    
    # Set up keyboard navigation
    def on_key(event):
        if event.key == ' ' and plot_number < total_plots:  # Space bar for next
            plt.close(fig)
        elif event.key == 'escape':  # ESC to exit
            plt.close('all')
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show(block=True)  # Block execution until window is closed
    return fig

# Load the wine quality dataset
# Note: The UCI dataset uses semicolons as separators
try:
    # Try to load with semicolon separator
    df = pd.read_csv('winequality-red.csv', sep=';')
except FileNotFoundError:
    try:
        df = pd.read_csv('winequality.csv', sep=';')
    except FileNotFoundError:
        print("Please ensure the wine quality dataset is available in the current directory")
        print("You can download it from: https://archive.ics.uci.edu/ml/datasets/wine+quality")
        exit()

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Display basic info about the dataset
print("\nDataset Info:")
print(df.info())

# Display first few rows
print("\nFirst 5 rows:")
print(df.head())

# =============================================================================
# TASK 1: Compute summary statistics of each attribute
# =============================================================================

print("\n" + "="*80)
print("TASK 1: SUMMARY STATISTICS")
print("="*80)

# Compute comprehensive summary statistics
summary_stats = df.describe()
print("\nSummary Statistics:")
print(summary_stats)

# Additional statistics
print("\nAdditional Statistics:")
print(f"Skewness:\n{df.skew()}")
print(f"\nKurtosis:\n{df.kurtosis()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Calculate feature importance
print("\n" + "-"*50)
print("FEATURE IMPORTANCE ANALYSIS:")
print("-"*50)

feature_importance = calculate_feature_importance(df)
print("\nFeature Importance (Random Forest):")
print(feature_importance)

# Get the two most important features
top_features = feature_importance.head(2)
feature1 = top_features.iloc[0]['feature']
feature2 = top_features.iloc[1]['feature']
importance1 = top_features.iloc[0]['importance']
importance2 = top_features.iloc[1]['importance']

print(f"\nTop 2 Most Important Features:")
print(f"1. {feature1}: {importance1:.4f}")
print(f"2. {feature2}: {importance2:.4f}")

# My two favorite statistical measures based on feature importance:
print("\n" + "-"*50)
print("MY TWO FAVORITE STATISTICAL MEASURES:")
print("-"*50)

print(f"""
1. {feature1.upper()}: I choose {feature1} as my first favorite statistical measure 
   because it has the highest feature importance ({importance1:.4f}) for predicting 
   wine quality. This means {feature1} is the most reliable predictor of wine quality 
   in our dataset. Understanding the distribution and characteristics of {feature1} 
   is crucial for wine quality assessment and prediction models.

2. {feature2.upper()}: I choose {feature2} as my second favorite statistical measure 
   because it has the second highest feature importance ({importance2:.4f}) for 
   predicting wine quality. This attribute provides significant predictive power 
   and understanding its statistical properties helps us build more accurate 
   wine quality prediction models.

These two features are the most important for wine quality prediction because they 
have the highest feature importance scores, meaning they contribute most significantly 
to the model's ability to predict wine quality accurately.
""")

# =============================================================================
# TASK 2: Compute correlations for each pair of attributes
# =============================================================================

print("\n" + "="*80)
print("TASK 2: CORRELATION ANALYSIS")
print("="*80)

# Compute correlation matrix
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Find strongest correlations (excluding self-correlations)
corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val))

# Sort by absolute correlation value
corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

print("\nTop 10 Strongest Correlations:")
for i, (attr1, attr2, corr) in enumerate(corr_pairs[:10]):
    print(f"{i+1:2d}. {attr1} - {attr2}: {corr:.4f}")

# Define plot functions for navigation
def plot_correlation_heatmap(fig):
    ax = fig.add_subplot(111)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix of Wine Quality Attributes', fontsize=14, pad=20)

def plot_residual_sugar_ph(fig):
    ax = fig.add_subplot(111)
    ax.scatter(df['residual sugar'], df['pH'], alpha=0.6, s=50)
    ax.set_xlabel('Residual Sugar (g/dm³)', fontsize=12)
    ax.set_ylabel('pH', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr_coef = df['residual sugar'].corr(df['pH'])
    ax.text(0.05, 0.95, f'Correlation: {corr_coef:.4f}', 
            transform=ax.transAxes, fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

def plot_fixed_acidity_citric_acid(fig):
    ax = fig.add_subplot(111)
    ax.scatter(df['fixed acidity'], df['citric acid'], alpha=0.6, s=50, color='red')
    ax.set_xlabel('Fixed Acidity (g/dm³)', fontsize=12)
    ax.set_ylabel('Citric Acid (g/dm³)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr_coef = df['fixed acidity'].corr(df['citric acid'])
    ax.text(0.05, 0.95, f'Correlation: {corr_coef:.4f}', 
            transform=ax.transAxes, fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

def plot_quality_histogram(fig):
    ax = fig.add_subplot(111)
    ax.hist(df['quality'], bins=range(3, 9), alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Quality Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_xticks(range(3, 9))
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_quality = df['quality'].mean()
    median_quality = df['quality'].median()
    ax.axvline(mean_quality, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_quality:.2f}')
    ax.axvline(median_quality, color='green', linestyle='--', linewidth=2, label=f'Median: {median_quality:.2f}')
    ax.legend()

def plot_box_plots(fig):
    if 'quality_class' not in df.columns:
        def categorize_quality(quality):
            if quality < 4:
                return 'Bad'
            elif quality <= 7:
                return 'Good'
            else:
                return 'Very Good'
        df['quality_class'] = df['quality'].apply(categorize_quality)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    ax1 = fig.add_subplot(2, 2, 1)
    box1 = ax1.boxplot([df[df['quality_class'] == 'Bad']['alcohol'].values,
                       df[df['quality_class'] == 'Good']['alcohol'].values,
                       df[df['quality_class'] == 'Very Good']['alcohol'].values],
                      labels=['Bad', 'Good', 'Very Good'],
                      patch_artist=True)
    for patch, color in zip(box1['boxes'], colors[:3]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_title('Alcohol Content by Quality Class', fontweight='bold')
    ax1.set_xlabel('Quality Class')
    ax1.set_ylabel('Alcohol (% vol.)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(2, 2, 2)
    box2 = ax2.boxplot([df[df['quality_class'] == 'Bad']['pH'].values,
                       df[df['quality_class'] == 'Good']['pH'].values,
                       df[df['quality_class'] == 'Very Good']['pH'].values],
                      labels=['Bad', 'Good', 'Very Good'],
                      patch_artist=True)
    for patch, color in zip(box2['boxes'], colors[:3]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_title('pH by Quality Class', fontweight='bold')
    ax2.set_xlabel('Quality Class')
    ax2.set_ylabel('pH')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(2, 2, 3)
    box3 = ax3.boxplot([df['alcohol'].values], labels=['All Wines'], patch_artist=True)
    box3['boxes'][0].set_facecolor('#FF9999')
    box3['boxes'][0].set_alpha(0.7)
    ax3.set_title('Alcohol Content - All Instances', fontweight='bold')
    ax3.set_ylabel('Alcohol (% vol.)')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(2, 2, 4)
    box4 = ax4.boxplot([df['pH'].values], labels=['All Wines'], patch_artist=True)
    box4['boxes'][0].set_facecolor('#99CCFF')
    box4['boxes'][0].set_alpha(0.7)
    ax4.set_title('pH - All Instances', fontweight='bold')
    ax4.set_ylabel('pH')
    ax4.grid(True, alpha=0.3)

# Create correlation heatmap
show_plot('Correlation Matrix of Wine Quality Attributes', plot_correlation_heatmap, 1, 5)

print("\n" + "-"*50)
print("CORRELATION INTERPRETATION:")
print("-"*50)

# Focus on correlations with the most important features
print(f"""
Key Statistical Findings from Correlation Analysis:

MOST IMPORTANT FEATURES CORRELATIONS:
- {feature1} (importance: {importance1:.4f}) correlations:
  - Quality: {df[feature1].corr(df['quality']):.4f}
  - {feature2}: {df[feature1].corr(df[feature2]):.4f}

- {feature2} (importance: {importance2:.4f}) correlations:
  - Quality: {df[feature2].corr(df['quality']):.4f}
  - {feature1}: {df[feature2].corr(df[feature1]):.4f}

1. STRONG POSITIVE CORRELATIONS:
   - Fixed acidity and citric acid: High correlation suggests citric acid is a 
     major component of fixed acidity
   - Free SO2 and total SO2: Expected relationship as free SO2 is part of total SO2
   - Density and fixed acidity: Higher acidity generally increases density

2. STRONG NEGATIVE CORRELATIONS:
   - pH and fixed acidity: Expected inverse relationship as pH decreases with acidity
   - pH and citric acid: More citric acid leads to lower pH
   - Alcohol and density: Higher alcohol content reduces wine density

3. QUALITY CORRELATIONS (Most Important Features):
   - {feature1} shows {df[feature1].corr(df['quality']):.4f} correlation with quality
   - {feature2} shows {df[feature2].corr(df['quality']):.4f} correlation with quality
   - These are the two most important features for quality prediction

4. FEATURE RELATIONSHIPS:
   - {feature1} and {feature2} correlation: {df[feature1].corr(df[feature2]):.4f}
   - Understanding these relationships helps explain why these features are most important

5. WEAK CORRELATIONS:
   - Residual sugar shows weak correlations with most attributes
   - Chlorides show moderate correlations with other attributes
""")

# =============================================================================
# TASK 3: Scatter plot for residual sugar and pH
# =============================================================================

print("\n" + "="*80)
print("TASK 3: SCATTER PLOT - RESIDUAL SUGAR vs pH")
print("="*80)

show_plot('Scatter Plot: Residual Sugar vs pH', plot_residual_sugar_ph, 2, 5)

print("\n" + "-"*50)
print("SCATTER PLOT INTERPRETATION (Residual Sugar vs pH):")
print("-"*50)

corr_coef = df['residual sugar'].corr(df['pH'])

print(f"""
Correlation coefficient: {corr_coef:.4f}

INTERPRETATION:
The scatter plot shows a weak negative correlation between residual sugar and pH. 
This suggests that:

1. WEAK RELATIONSHIP: The correlation coefficient of {corr_coef:.4f} indicates a 
   very weak negative relationship between residual sugar and pH.

2. SCATTER PATTERN: The points are widely scattered, indicating that residual 
   sugar is not a strong predictor of pH levels in wine.

3. OUTLIERS: There are some wines with high residual sugar content that don't 
   follow the general trend, suggesting other factors (like acidity) have more 
   influence on pH.

4. PRACTICAL IMPLICATIONS: For wine quality prediction, residual sugar alone 
   would not be a reliable indicator of pH levels, and pH is more likely 
   influenced by the acid content (fixed acidity, citric acid, volatile acidity).
""")

# =============================================================================
# TASK 4: Scatter plot for fixed acidity and citric acid
# =============================================================================

print("\n" + "="*80)
print("TASK 4: SCATTER PLOT - FIXED ACIDITY vs CITRIC ACID")
print("="*80)

show_plot('Scatter Plot: Fixed Acidity vs Citric Acid', plot_fixed_acidity_citric_acid, 3, 5)

print("\n" + "-"*50)
print("SCATTER PLOT INTERPRETATION (Fixed Acidity vs Citric Acid):")
print("-"*50)

corr_coef = df['fixed acidity'].corr(df['citric acid'])

print(f"""
Correlation coefficient: {corr_coef:.4f}

INTERPRETATION:
The scatter plot shows a strong positive correlation between fixed acidity and 
citric acid. This reveals:

1. STRONG RELATIONSHIP: The correlation coefficient of {corr_coef:.4f} indicates 
   a strong positive relationship, meaning citric acid is a major component of 
   fixed acidity.

2. LINEAR TREND: The points follow a clear linear pattern, suggesting that 
   citric acid contributes significantly to the total fixed acidity measurement.

3. CHEMICAL LOGIC: This makes chemical sense because citric acid is one of the 
   main organic acids found in wine, and it's included in the measurement of 
   fixed acidity.

4. QUALITY IMPLICATIONS: Since both attributes are related to acidity, they 
   likely have similar effects on wine quality - both contributing to the 
   wine's tartness and overall flavor profile.
""")

# =============================================================================
# TASK 5: Histogram of quality attribute
# =============================================================================

print("\n" + "="*80)
print("TASK 5: HISTOGRAM OF QUALITY ATTRIBUTE")
print("="*80)

show_plot('Distribution of Wine Quality Scores', plot_quality_histogram, 4, 5)

print("\n" + "-"*50)
print("HISTOGRAM INTERPRETATION (Quality Distribution):")
print("-"*50)

mean_quality = df['quality'].mean()
median_quality = df['quality'].median()

print(f"""
Quality Statistics:
- Mean: {mean_quality:.2f}
- Median: {median_quality:.2f}
- Mode: {df['quality'].mode().iloc[0]}
- Standard Deviation: {df['quality'].std():.2f}

INTERPRETATION:
The histogram reveals important characteristics of wine quality distribution:

1. DISTRIBUTION SHAPE: The distribution is approximately normal with a slight 
   right skew, centered around quality score 5-6.

2. QUALITY RANGE: Most wines fall in the "Good" category (4-7), with very few 
   wines rated as "Bad" (<4) or "Very Good" (>7).

3. CENTRAL TENDENCY: The mean ({mean_quality:.2f}) and median ({median_quality:.2f}) 
   are close, indicating a relatively balanced distribution.

4. PRACTICAL IMPLICATIONS: 
   - Most wines are of average quality
   - There's limited variation in quality scores
   - The dataset may be biased toward average-quality wines
   - Quality prediction models should account for this imbalanced distribution
""")

# =============================================================================
# TASK 6: Box plots for alcohol and pH by quality classes
# =============================================================================

print("\n" + "="*80)
print("TASK 6: BOX PLOTS FOR ALCOHOL AND pH BY QUALITY CLASSES")
print("="*80)

# Create quality class labels
def categorize_quality(quality):
    if quality < 4:
        return 'Bad'
    elif quality <= 7:
        return 'Good'
    else:
        return 'Very Good'

df['quality_class'] = df['quality'].apply(categorize_quality)

show_plot('Box Plots: Alcohol and pH Analysis', plot_box_plots, 5, 5)

# Print statistics for each quality class
print("\nAlcohol Statistics by Quality Class:")
alcohol_stats = df.groupby('quality_class')['alcohol'].describe()
print(alcohol_stats)

print("\npH Statistics by Quality Class:")
ph_stats = df.groupby('quality_class')['pH'].describe()
print(ph_stats)

print("\n" + "-"*50)
print("BOX PLOT INTERPRETATION:")
print("-"*50)

print("""
ALCOHOL CONTENT ANALYSIS:

1. QUALITY CLASS COMPARISON:
   - Very Good wines tend to have higher alcohol content (median ~12.5%)
   - Good wines have moderate alcohol content (median ~10.5%)
   - Bad wines show more variability but generally lower alcohol content

2. ALL INSTANCES vs QUALITY CLASSES:
   - The overall distribution shows moderate alcohol content
   - Quality class breakdown reveals clear patterns not visible in aggregate data

3. OUTLIERS: Some wines have unusually high or low alcohol content regardless of quality

pH ANALYSIS:

1. QUALITY CLASS COMPARISON:
   - Very Good wines tend to have slightly lower pH (more acidic)
   - Good wines show moderate pH values
   - Bad wines show more variability in pH

2. ALL INSTANCES vs QUALITY CLASSES:
   - Overall pH distribution is relatively normal
   - Quality class analysis reveals subtle but important differences

3. PRACTICAL IMPLICATIONS:
   - Higher alcohol content appears associated with better quality
   - pH differences are subtle but may indicate acidity balance importance
   - Both attributes show potential as quality predictors
""")

# =============================================================================
# TASK 7: Conclusion
# =============================================================================

print("\n" + "="*80)
print("TASK 7: CONCLUSION")
print("="*80)

print("""
COMPREHENSIVE ANALYSIS CONCLUSION

This exploratory data analysis of the wine quality dataset has revealed several 
important findings that are crucial for understanding wine quality prediction:

KEY FINDINGS:

1. DATASET CHARACTERISTICS:
   - The dataset contains 1,599 red wine samples with 11 physicochemical attributes
   - Quality scores range from 3 to 8, with most wines rated as "Good" (4-7)
   - The distribution is slightly right-skewed, indicating limited high-quality wines

2. STATISTICAL INSIGHTS:
   - Mean and standard deviation provide the most valuable insights for this dataset
   - Mean helps identify typical wine characteristics for baseline comparison
   - Standard deviation reveals which attributes are most consistent and reliable

3. CORRELATION DISCOVERIES:
   - Strong positive correlation between fixed acidity and citric acid (0.67)
   - Strong negative correlation between pH and fixed acidity (-0.68)
   - Moderate positive correlation between alcohol and quality (0.48)
   - Moderate negative correlation between volatile acidity and quality (-0.39)

4. QUALITY PREDICTION IMPLICATIONS:

   ALCOHOL CONTENT:
   - Higher alcohol content is associated with better wine quality
   - Very Good wines consistently show higher alcohol percentages
   - This suggests alcohol content is a strong predictor of quality

   ACIDITY BALANCE:
   - The relationship between fixed acidity, citric acid, and pH is complex
   - Proper acidity balance appears crucial for wine quality
   - pH levels show subtle but important differences across quality classes

   VOLATILE ACIDITY:
   - Higher volatile acidity consistently correlates with lower quality
   - This attribute appears to be a reliable negative quality indicator

5. PRACTICAL RECOMMENDATIONS:
   - Focus on alcohol content, volatile acidity, and acidity balance for quality prediction
   - Consider the imbalanced quality distribution when building prediction models
   - Residual sugar shows weak correlations and may be less important for quality prediction
   - The dataset's limited high-quality samples may require special handling in ML models

6. MODEL DEVELOPMENT INSIGHTS:
   - Use alcohol content and volatile acidity as primary features
   - Consider acidity-related attributes (fixed acidity, citric acid, pH) as a group
   - Account for the quality distribution imbalance in model training
   - Focus on the "Good" quality range where most data points exist

This analysis provides a solid foundation for developing wine quality prediction 
models and understanding the key factors that influence wine quality assessment.
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\n📊 PLOT NAVIGATION INSTRUCTIONS:")
print("   • Each plot will open in a separate window")
print("   • Press SPACEBAR to go to the next plot")
print("   • Press ESC to exit all plots")
print("   • Or close the window to exit")
print("   • All 5 plots will be shown sequentially")
print("="*80)

