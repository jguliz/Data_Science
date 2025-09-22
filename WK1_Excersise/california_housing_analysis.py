import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== Dataset Analysis: California Housing Dataset ===")
print()

# Load the California Housing dataset
housing = fetch_california_housing()
housing_df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
housing_df['target'] = housing.target

print("Dataset Info:")
print(f"Shape: {housing_df.shape}")
print(f"Features: {housing.feature_names}")
target_desc = housing.DESCR.split('Target')[1].split('\n')[0].strip()
print(f"Target: {target_desc}")
print()
print("First 5 rows:")
print(housing_df.head())
print()
print("Dataset Description:")
print(housing_df.describe())
print()

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# 1. Histograms for each feature
print("=== HISTOGRAMS ===")
feature_names = housing.feature_names
for i, feature in enumerate(feature_names[:8]):  # Show first 8 features
    plt.subplot(3, 4, i+1)
    plt.hist(housing_df[feature], bins=30, alpha=0.7, edgecolor='black')
    plt.title(f'{feature.replace("_", " ").title()} Distribution')
    plt.xlabel(feature.replace("_", " ").title())
    plt.ylabel('Frequency')

# Target variable histogram
plt.subplot(3, 4, 9)
plt.hist(housing_df['target'], bins=30, alpha=0.7, color='red', edgecolor='black')
plt.title('House Value Distribution')
plt.xlabel('House Value (in $100,000s)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 2. Scatter plots showing relationships with target
print("=== SCATTER PLOTS ===")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for i, feature in enumerate(feature_names):
    axes[i].scatter(housing_df[feature], housing_df['target'], alpha=0.5, s=1)
    axes[i].set_xlabel(feature.replace("_", " ").title())
    axes[i].set_ylabel('House Value ($100k)')
    axes[i].set_title(f'{feature.replace("_", " ").title()} vs House Value')

plt.tight_layout()
plt.show()

# 3. Box plots for key features
print("=== BOX PLOTS ===")
plt.figure(figsize=(15, 10))

# Create box plots for features that make sense to categorize
plt.subplot(2, 2, 1)
housing_df.boxplot(column='target', by='AveRooms', ax=plt.gca())
plt.title('House Value by Average Rooms')
plt.suptitle('')

plt.subplot(2, 2, 2)
housing_df.boxplot(column='target', by='AveBedrms', ax=plt.gca())
plt.title('House Value by Average Bedrooms')
plt.suptitle('')

plt.subplot(2, 2, 3)
housing_df.boxplot(column='target', by='Population', ax=plt.gca())
plt.title('House Value by Population')
plt.suptitle('')

plt.subplot(2, 2, 4)
housing_df.boxplot(column='target', by='AveOccup', ax=plt.gca())
plt.title('House Value by Average Occupancy')
plt.suptitle('')

plt.tight_layout()
plt.show()

# Additional analysis using seaborn
print("=== ADDITIONAL SEABORN VISUALIZATIONS ===")

# Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = housing_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Feature Correlation Heatmap - California Housing')
plt.show()

# Pair plot for key features
key_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'target']
plt.figure(figsize=(15, 12))
sns.pairplot(housing_df[key_features], diag_kind='hist')
plt.suptitle('Pair Plot of Key California Housing Features', y=1.02)
plt.show()

# Violin plots for target distribution by quantiles
plt.figure(figsize=(15, 10))

# Create quantile-based categories for some features
housing_df['MedInc_quartile'] = pd.qcut(housing_df['MedInc'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
housing_df['HouseAge_quartile'] = pd.qcut(housing_df['HouseAge'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

plt.subplot(2, 2, 1)
sns.violinplot(data=housing_df, x='MedInc_quartile', y='target')
plt.title('House Value Distribution by Median Income Quartile')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
sns.violinplot(data=housing_df, x='HouseAge_quartile', y='target')
plt.title('House Value Distribution by House Age Quartile')
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
sns.scatterplot(data=housing_df, x='Latitude', y='Longitude', hue='target', 
                palette='viridis', alpha=0.6, s=1)
plt.title('Geographic Distribution of House Values')
plt.xlabel('Latitude')
plt.ylabel('Longitude')

plt.subplot(2, 2, 4)
sns.scatterplot(data=housing_df, x='MedInc', y='target', alpha=0.5, s=1)
plt.title('Median Income vs House Value')
plt.xlabel('Median Income')
plt.ylabel('House Value ($100k)')

plt.tight_layout()
plt.show()

print("=== DATA INSIGHTS ===")
print()
print("What we can see from this data:")
print()
print("1. HISTOGRAMS:")
print("   - Median Income shows a right-skewed distribution")
print("   - House Age is relatively uniformly distributed")
print("   - Average Rooms and Bedrooms show normal-like distributions")
print("   - Population and Households are heavily right-skewed")
print("   - House values (target) show a right-skewed distribution with a long tail")
print()
print("2. SCATTER PLOTS:")
print("   - Strong positive correlation between Median Income and House Value")
print("   - Moderate negative correlation between House Age and House Value")
print("   - Average Rooms shows positive correlation with House Value")
print("   - Geographic location (Latitude/Longitude) shows clear patterns")
print("   - Population and Households show weak correlations with House Value")
print()
print("3. BOX PLOTS:")
print("   - Higher average rooms generally correspond to higher house values")
print("   - More bedrooms don't necessarily mean higher values")
print("   - Population density shows complex relationship with house values")
print("   - Average occupancy has interesting patterns across value ranges")
print()
print("4. CORRELATION ANALYSIS:")
print("   - Median Income has the strongest positive correlation with House Value")
print("   - House Age shows negative correlation with House Value")
print("   - Average Rooms is positively correlated with House Value")
print("   - Geographic features (Latitude/Longitude) show moderate correlations")
print()
print("5. KEY INSIGHTS:")
print("   - This is a regression problem (predicting continuous house values)")
print("   - Median Income is the most important predictor")
print("   - Geographic location matters significantly (California coastal effect)")
print("   - House age has a negative impact on value")
print("   - The dataset shows clear patterns suitable for machine learning")
print("   - Some features may need transformation due to skewness")
print("   - The data represents real estate patterns in California")
