"""
COMPREHENSIVE DATA ANALYSIS TUTORIAL
====================================

This tutorial teaches you how data analysis works step by step.
We'll use the Iris dataset as our example throughout.

Author: Data Science Tutorial
Purpose: Educational - Learn how to analyze data with Python
"""

# =============================================================================
# STEP 1: IMPORTING LIBRARIES
# =============================================================================
"""
WHY WE IMPORT THESE LIBRARIES:

1. pandas (pd): 
   - The Swiss Army knife of data analysis
   - Creates DataFrames (like Excel spreadsheets in Python)
   - Handles missing data, data types, and data manipulation
   - Think of it as your data container and manipulator

2. numpy (np):
   - The foundation of scientific computing in Python
   - Provides arrays and mathematical functions
   - pandas is built on top of numpy
   - Handles numerical computations efficiently

3. matplotlib.pyplot (plt):
   - The basic plotting library
   - Creates static, interactive, and animated visualizations
   - Think of it as your canvas for drawing graphs

4. seaborn (sns):
   - Built on top of matplotlib
   - Makes statistical visualizations beautiful and easy
   - Provides high-level functions for complex plots
   - Think of it as matplotlib's stylish cousin

5. sklearn.datasets:
   - Provides sample datasets for learning
   - load_iris() gives us the famous Iris flower dataset
   - Perfect for learning because it's clean and well-documented
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

print("=" * 60)
print("STEP 1: LIBRARIES IMPORTED SUCCESSFULLY")
print("=" * 60)
print("✓ pandas: For data manipulation and analysis")
print("✓ numpy: For numerical computations")
print("✓ matplotlib: For basic plotting")
print("✓ seaborn: For beautiful statistical plots")
print("✓ sklearn: For sample datasets")
print()

# =============================================================================
# STEP 2: LOADING DATA
# =============================================================================
"""
WHAT IS A DATASET?
A dataset is a collection of data points (rows) with features (columns).
Think of it like a spreadsheet where:
- Each row = one observation (e.g., one flower)
- Each column = one feature (e.g., petal length, sepal width)
- The target = what we want to predict (e.g., flower species)

THE IRIS DATASET:
- 150 flowers from 3 species: setosa, versicolor, virginica
- 4 measurements per flower: sepal length/width, petal length/width
- This is a CLASSIFICATION problem (predicting categories)
- It's famous because it's perfect for learning - clean, small, well-understood
"""

print("=" * 60)
print("STEP 2: LOADING THE IRIS DATASET")
print("=" * 60)

# Load the dataset
iris = load_iris()

print("Dataset loaded! Let's explore its structure:")
print(f"✓ Number of samples: {iris.data.shape[0]}")
print(f"✓ Number of features: {iris.data.shape[1]}")
print(f"✓ Feature names: {iris.feature_names}")
print(f"✓ Target classes: {iris.target_names}")
print()

# =============================================================================
# STEP 3: CONVERTING TO PANDAS DATAFRAME
# =============================================================================
"""
WHY CONVERT TO PANDAS DATAFRAME?
- sklearn returns numpy arrays (just numbers)
- pandas DataFrames are more user-friendly
- We get column names, data types, and powerful methods
- Think of it as upgrading from a basic list to a smart spreadsheet
"""

print("=" * 60)
print("STEP 3: CONVERTING TO PANDAS DATAFRAME")
print("=" * 60)

# Create DataFrame from the data
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target (species) as a new column
iris_df['species'] = iris.target

# Add species names for easier reading
iris_df['species_name'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("DataFrame created! Here's what it looks like:")
print(iris_df.head())
print()
print("DataFrame info:")
print(f"✓ Shape: {iris_df.shape} (rows, columns)")
print(f"✓ Data types: {iris_df.dtypes.tolist()}")
print(f"✓ Memory usage: {iris_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print()

# =============================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
"""
WHAT IS EXPLORATORY DATA ANALYSIS?
EDA is the process of investigating datasets to:
1. Understand the data structure
2. Find patterns and relationships
3. Detect anomalies or problems
4. Generate hypotheses
5. Prepare for further analysis

KEY EDA TECHNIQUES:
- Summary statistics (mean, median, std, etc.)
- Data visualization (plots and charts)
- Correlation analysis
- Distribution analysis
"""

print("=" * 60)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("4.1 BASIC STATISTICS:")
print("The describe() method gives us key statistics:")
print(iris_df.describe())
print()

print("4.2 DATA DISTRIBUTION:")
print("Let's see how many samples we have of each species:")
print(iris_df['species_name'].value_counts())
print()

print("4.3 MISSING DATA CHECK:")
print("Are there any missing values?")
print(iris_df.isnull().sum())
print()

# =============================================================================
# STEP 5: DATA VISUALIZATION - HISTOGRAMS
# =============================================================================
"""
WHAT ARE HISTOGRAMS?
Histograms show the distribution of a single variable:
- X-axis: Values of the variable (e.g., petal length)
- Y-axis: Frequency (how many times each value appears)
- Bars: Show how data is distributed

WHY USE HISTOGRAMS?
- See if data is normally distributed
- Identify outliers
- Understand data spread
- Compare distributions between groups
"""

print("=" * 60)
print("STEP 5: CREATING HISTOGRAMS")
print("=" * 60)

# Set up the plotting style
plt.style.use('default')  # Clean, professional look
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Histograms: Distribution of Each Feature', fontsize=16, fontweight='bold')

# Create histograms for each feature
features = iris.feature_names
colors = ['skyblue', 'lightgreen', 'salmon', 'gold']

for i, (feature, color) in enumerate(zip(features, colors)):
    row, col = i // 2, i % 2
    axes[row, col].hist(iris_df[feature], bins=20, alpha=0.7, color=color, edgecolor='black')
    axes[row, col].set_title(f'{feature.title()}', fontweight='bold')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Histograms created! What do we see?")
print("  - Sepal length/width: Roughly normal distributions")
print("  - Petal length/width: Bimodal (two peaks) - suggests two groups!")
print()

# =============================================================================
# STEP 6: DATA VISUALIZATION - SCATTER PLOTS
# =============================================================================
"""
WHAT ARE SCATTER PLOTS?
Scatter plots show relationships between two variables:
- X-axis: One variable (e.g., petal length)
- Y-axis: Another variable (e.g., petal width)
- Points: Each point represents one data sample
- Colors: Can represent categories (e.g., species)

WHY USE SCATTER PLOTS?
- See correlations between variables
- Identify clusters or groups
- Spot outliers
- Understand relationships
"""

print("=" * 60)
print("STEP 6: CREATING SCATTER PLOTS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Scatter Plots: Relationships Between Features', fontsize=16, fontweight='bold')

# Define colors for each species
species_colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}

# Create scatter plots
plot_combinations = [
    ('sepal length (cm)', 'sepal width (cm)'),
    ('petal length (cm)', 'petal width (cm)'),
    ('sepal length (cm)', 'petal length (cm)'),
    ('sepal width (cm)', 'petal width (cm)')
]

for i, (x_feat, y_feat) in enumerate(plot_combinations):
    row, col = i // 2, i % 2
    
    # Plot each species with different colors
    for species in iris.target_names:
        species_data = iris_df[iris_df['species_name'] == species]
        axes[row, col].scatter(species_data[x_feat], species_data[y_feat], 
                              c=species_colors[species], label=species, alpha=0.7, s=50)
    
    axes[row, col].set_xlabel(x_feat)
    axes[row, col].set_ylabel(y_feat)
    axes[row, col].set_title(f'{x_feat.split()[0].title()} vs {y_feat.split()[0].title()}')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Scatter plots created! What do we see?")
print("  - Setosa (red) is clearly separated from others")
print("  - Versicolor and Virginica overlap but are distinguishable")
print("  - Petal measurements show better separation than sepal measurements")
print()

# =============================================================================
# STEP 7: DATA VISUALIZATION - BOX PLOTS
# =============================================================================
"""
WHAT ARE BOX PLOTS?
Box plots show the distribution of data across categories:
- Box: Contains 50% of the data (25th to 75th percentile)
- Line in box: Median (50th percentile)
- Whiskers: Extend to show data range
- Dots: Outliers (unusual values)

WHY USE BOX PLOTS?
- Compare distributions between groups
- Identify outliers
- See data spread and skewness
- Understand group differences
"""

print("=" * 60)
print("STEP 7: CREATING BOX PLOTS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Box Plots: Feature Distributions by Species', fontsize=16, fontweight='bold')

for i, feature in enumerate(features):
    row, col = i // 2, i % 2
    
    # Create box plot using pandas
    iris_df.boxplot(column=feature, by='species_name', ax=axes[row, col])
    axes[row, col].set_title(f'{feature.title()} by Species')
    axes[row, col].set_xlabel('Species')
    axes[row, col].set_ylabel(feature)
    axes[row, col].grid(True, alpha=0.3)

# Remove the default title that pandas adds
plt.suptitle('')  # This removes the automatic title

plt.tight_layout()
plt.show()

print("✓ Box plots created! What do we see?")
print("  - Setosa has smallest petals, largest sepal width")
print("  - Virginica has largest petals")
print("  - Clear differences between species")
print()

# =============================================================================
# STEP 8: CORRELATION ANALYSIS
# =============================================================================
"""
WHAT IS CORRELATION?
Correlation measures how two variables move together:
- +1: Perfect positive correlation (as one increases, so does the other)
- 0: No correlation (variables are independent)
- -1: Perfect negative correlation (as one increases, the other decreases)

WHY ANALYZE CORRELATIONS?
- Understand feature relationships
- Identify redundant features
- Find the most important predictors
- Guide feature selection for machine learning
"""

print("=" * 60)
print("STEP 8: CORRELATION ANALYSIS")
print("=" * 60)

# Calculate correlation matrix
correlation_matrix = iris_df[iris.feature_names].corr()

print("Correlation Matrix:")
print(correlation_matrix.round(3))
print()

# Create correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.show()

print("✓ Correlation analysis complete! Key findings:")
print("  - Petal length and petal width: 0.96 (very high correlation)")
print("  - Sepal length and petal length: 0.87 (high correlation)")
print("  - Sepal width shows negative correlations")
print()

# =============================================================================
# STEP 9: ADVANCED VISUALIZATIONS WITH SEABORN
# =============================================================================
"""
WHY USE SEABORN?
- Beautiful default styles
- Statistical plots built-in
- Easy to create complex visualizations
- Integrates well with pandas DataFrames
"""

print("=" * 60)
print("STEP 9: ADVANCED SEABORN VISUALIZATIONS")
print("=" * 60)

# Pair plot - shows all pairwise relationships
print("Creating pair plot...")
plt.figure(figsize=(12, 10))
sns.pairplot(iris_df, hue='species_name', diag_kind='hist', palette='husl')
plt.suptitle('Pair Plot: All Feature Relationships', y=1.02, fontsize=16, fontweight='bold')
plt.show()

# Violin plots - show distribution shape
print("Creating violin plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Violin Plots: Distribution Shapes by Species', fontsize=16, fontweight='bold')

for i, feature in enumerate(features):
    row, col = i // 2, i % 2
    sns.violinplot(data=iris_df, x='species_name', y=feature, ax=axes[row, col])
    axes[row, col].set_title(f'{feature.title()} Distribution')
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Advanced visualizations complete!")
print()

# =============================================================================
# STEP 10: DRAWING INSIGHTS AND CONCLUSIONS
# =============================================================================
"""
HOW TO INTERPRET YOUR DATA:
1. Look for patterns in visualizations
2. Check statistical summaries
3. Consider correlations
4. Think about what makes sense biologically/logically
5. Form hypotheses for further testing
"""

print("=" * 60)
print("STEP 10: DATA INSIGHTS AND CONCLUSIONS")
print("=" * 60)

print("🔍 WHAT WE LEARNED FROM THE IRIS DATASET:")
print()

print("1. 📊 DATA STRUCTURE:")
print("   • 150 samples, 4 features, 3 classes")
print("   • No missing data - clean dataset")
print("   • Balanced classes (50 samples each)")
print()

print("2. 📈 DISTRIBUTIONS:")
print("   • Sepal measurements: roughly normal")
print("   • Petal measurements: bimodal (two peaks)")
print("   • Bimodality suggests natural grouping")
print()

print("3. 🔗 RELATIONSHIPS:")
print("   • Petal length and width are highly correlated (0.96)")
print("   • Sepal length and petal length are correlated (0.87)")
print("   • Sepal width is negatively correlated with others")
print()

print("4. 🎯 SPECIES CHARACTERISTICS:")
print("   • Setosa: Small petals, wide sepals (easily distinguishable)")
print("   • Virginica: Largest petals")
print("   • Versicolor: Intermediate characteristics")
print()

print("5. 🤖 MACHINE LEARNING IMPLICATIONS:")
print("   • Perfect for classification algorithms")
print("   • Petal measurements are most discriminative")
print("   • Setosa can be classified with simple rules")
print("   • Versicolor vs Virginica needs more sophisticated methods")
print()

print("6. 🧠 BIOLOGICAL INSIGHTS:")
print("   • Petal size is the main distinguishing feature")
print("   • Sepal width varies inversely with other measurements")
print("   • Clear evolutionary differences between species")
print()

# =============================================================================
# STEP 11: NEXT STEPS IN DATA SCIENCE
# =============================================================================
"""
WHAT COMES NEXT?
1. Feature Engineering: Create new features from existing ones
2. Machine Learning: Build predictive models
3. Model Evaluation: Test how well models perform
4. Deployment: Use models in real applications
5. Monitoring: Track model performance over time
"""

print("=" * 60)
print("STEP 11: NEXT STEPS IN DATA SCIENCE")
print("=" * 60)

print("🚀 WHERE TO GO FROM HERE:")
print()
print("1. 📚 LEARN MORE:")
print("   • Machine Learning algorithms (classification)")
print("   • Feature engineering techniques")
print("   • Model evaluation metrics")
print("   • Cross-validation methods")
print()

print("2. 🛠️ PRACTICE WITH:")
print("   • Other datasets (wine, digits, breast cancer)")
print("   • Real-world data from Kaggle")
print("   • Your own data collection projects")
print()

print("3. 🔧 TOOLS TO EXPLORE:")
print("   • scikit-learn for machine learning")
print("   • Jupyter Notebooks for interactive analysis")
print("   • Plotly for interactive visualizations")
print("   • Streamlit for data apps")
print()

print("4. 📖 CONCEPTS TO STUDY:")
print("   • Statistical hypothesis testing")
print("   • Data preprocessing and cleaning")
print("   • Dimensionality reduction (PCA)")
print("   • Ensemble methods")
print()

print("=" * 60)
print("🎉 CONGRATULATIONS! YOU'VE COMPLETED THE TUTORIAL!")
print("=" * 60)
print("You now understand:")
print("✓ How to load and explore datasets")
print("✓ How to create meaningful visualizations")
print("✓ How to interpret data and draw insights")
print("✓ How to prepare for machine learning")
print()
print("Keep practicing with different datasets and you'll become")
print("a data science expert in no time! 🚀")
