import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== Dataset Analysis: Iris Dataset ===")
print()

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species_name'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Dataset Info:")
print(f"Shape: {iris_df.shape}")
print(f"Features: {iris.feature_names}")
print(f"Target classes: {iris.target_names}")
print()
print("First 5 rows:")
print(iris_df.head())
print()
print("Dataset Description:")
print(iris_df.describe())
print()

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# 1. Histograms for each feature
print("=== HISTOGRAMS ===")
plt.subplot(3, 4, 1)
plt.hist(iris_df['sepal length (cm)'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Sepal Length Distribution')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')

plt.subplot(3, 4, 2)
plt.hist(iris_df['sepal width (cm)'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('Sepal Width Distribution')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')

plt.subplot(3, 4, 3)
plt.hist(iris_df['petal length (cm)'], bins=20, alpha=0.7, color='salmon', edgecolor='black')
plt.title('Petal Length Distribution')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')

plt.subplot(3, 4, 4)
plt.hist(iris_df['petal width (cm)'], bins=20, alpha=0.7, color='gold', edgecolor='black')
plt.title('Petal Width Distribution')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')

# 2. Scatter plots showing relationships
print("=== SCATTER PLOTS ===")
plt.subplot(3, 4, 5)
colors = ['red', 'green', 'blue']
for i, species in enumerate(iris.target_names):
    species_data = iris_df[iris_df['species'] == i]
    plt.scatter(species_data['sepal length (cm)'], species_data['sepal width (cm)'], 
               c=colors[i], label=species, alpha=0.7)
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()

plt.subplot(3, 4, 6)
for i, species in enumerate(iris.target_names):
    species_data = iris_df[iris_df['species'] == i]
    plt.scatter(species_data['petal length (cm)'], species_data['petal width (cm)'], 
               c=colors[i], label=species, alpha=0.7)
plt.title('Petal Length vs Petal Width')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()

plt.subplot(3, 4, 7)
for i, species in enumerate(iris.target_names):
    species_data = iris_df[iris_df['species'] == i]
    plt.scatter(species_data['sepal length (cm)'], species_data['petal length (cm)'], 
               c=colors[i], label=species, alpha=0.7)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()

plt.subplot(3, 4, 8)
for i, species in enumerate(iris.target_names):
    species_data = iris_df[iris_df['species'] == i]
    plt.scatter(species_data['sepal width (cm)'], species_data['petal width (cm)'], 
               c=colors[i], label=species, alpha=0.7)
plt.title('Sepal Width vs Petal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()

# 3. Box plots
print("=== BOX PLOTS ===")
plt.subplot(3, 4, 9)
iris_df.boxplot(column='sepal length (cm)', by='species_name', ax=plt.gca())
plt.title('Sepal Length by Species')
plt.suptitle('')  # Remove default title

plt.subplot(3, 4, 10)
iris_df.boxplot(column='sepal width (cm)', by='species_name', ax=plt.gca())
plt.title('Sepal Width by Species')
plt.suptitle('')

plt.subplot(3, 4, 11)
iris_df.boxplot(column='petal length (cm)', by='species_name', ax=plt.gca())
plt.title('Petal Length by Species')
plt.suptitle('')

plt.subplot(3, 4, 12)
iris_df.boxplot(column='petal width (cm)', by='species_name', ax=plt.gca())
plt.title('Petal Width by Species')
plt.suptitle('')

plt.tight_layout()
plt.show()

# Additional analysis using seaborn
print("=== ADDITIONAL SEABORN VISUALIZATIONS ===")

# Pair plot
plt.figure(figsize=(12, 10))
sns.pairplot(iris_df, hue='species_name', diag_kind='hist')
plt.suptitle('Pair Plot of Iris Dataset Features', y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = iris_df[iris.feature_names].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# Violin plots
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.violinplot(data=iris_df, x='species_name', y='sepal length (cm)')
plt.title('Sepal Length Distribution by Species')

plt.subplot(2, 2, 2)
sns.violinplot(data=iris_df, x='species_name', y='sepal width (cm)')
plt.title('Sepal Width Distribution by Species')

plt.subplot(2, 2, 3)
sns.violinplot(data=iris_df, x='species_name', y='petal length (cm)')
plt.title('Petal Length Distribution by Species')

plt.subplot(2, 2, 4)
sns.violinplot(data=iris_df, x='species_name', y='petal width (cm)')
plt.title('Petal Width Distribution by Species')

plt.tight_layout()
plt.show()

print("=== DATA INSIGHTS ===")
print()
print("What we can see from this data:")
print()
print("1. HISTOGRAMS:")
print("   - Sepal length and width show relatively normal distributions")
print("   - Petal length and width show bimodal distributions, suggesting two distinct groups")
print("   - This hints at the separability of the iris species")
print()
print("2. SCATTER PLOTS:")
print("   - Setosa (red) is clearly separated from the other two species")
print("   - Versicolor (green) and Virginica (blue) show some overlap")
print("   - Petal measurements show better separation than sepal measurements")
print("   - Petal length vs petal width shows the clearest separation between species")
print()
print("3. BOX PLOTS:")
print("   - Setosa has the smallest petal measurements and largest sepal width")
print("   - Virginica has the largest petal measurements")
print("   - Sepal length shows the least variation between species")
print("   - Petal width shows the most distinct differences between species")
print()
print("4. CORRELATION ANALYSIS:")
print("   - Petal length and petal width are highly correlated (0.96)")
print("   - Sepal length and petal length are moderately correlated (0.87)")
print("   - Sepal width shows negative correlation with other features")
print()
print("5. KEY INSIGHTS:")
print("   - The dataset is well-suited for classification tasks")
print("   - Petal measurements are more discriminative than sepal measurements")
print("   - Setosa is easily distinguishable from the other species")
print("   - Versicolor and Virginica require more sophisticated classification")
print("   - The data shows clear patterns that would work well with machine learning algorithms")