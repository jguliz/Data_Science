"""
DISCRIMINATIVE POWER ANALYSIS
============================

This script demonstrates how to measure which features have the most
"discriminative power" - i.e., which features are most reliable for
making inferences about species classification.

The concept: Features with tighter spreads (lower variance) within each
class are more reliable for making predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the data
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species_name'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("=" * 70)
print("DISCRIMINATIVE POWER ANALYSIS")
print("=" * 70)
print("Question: Which features have the highest 'discriminative power'?")
print("Answer: Features with the tightest spreads (lowest variance) within each class")
print()

# =============================================================================
# STEP 1: Calculate variance within each species for each feature
# =============================================================================
print("STEP 1: CALCULATING VARIANCE WITHIN EACH SPECIES")
print("-" * 50)

# Calculate variance for each feature within each species
variance_analysis = {}

for feature in iris.feature_names:
    variance_analysis[feature] = {}
    for species in iris.target_names:
        species_data = iris_df[iris_df['species_name'] == species][feature]
        variance_analysis[feature][species] = species_data.var()
        
        print(f"{feature:20} | {species:10} | Variance: {species_data.var():.4f}")

print()

# =============================================================================
# STEP 2: Calculate average variance across species for each feature
# =============================================================================
print("STEP 2: AVERAGE VARIANCE ACROSS SPECIES (Lower = More Discriminative)")
print("-" * 50)

feature_discriminative_power = {}

for feature in iris.feature_names:
    avg_variance = np.mean([variance_analysis[feature][species] for species in iris.target_names])
    feature_discriminative_power[feature] = avg_variance
    
    print(f"{feature:20} | Average Variance: {avg_variance:.4f}")

print()

# =============================================================================
# STEP 3: Rank features by discriminative power
# =============================================================================
print("STEP 3: RANKING BY DISCRIMINATIVE POWER")
print("-" * 50)
print("(Lower variance = Higher discriminative power = More reliable for inference)")
print()

# Sort by variance (ascending - lower variance is better)
sorted_features = sorted(feature_discriminative_power.items(), key=lambda x: x[1])

for i, (feature, variance) in enumerate(sorted_features, 1):
    discriminative_score = 1 / variance  # Higher score = more discriminative
    print(f"{i}. {feature:20} | Variance: {variance:.4f} | Discriminative Score: {discriminative_score:.2f}")

print()

# =============================================================================
# STEP 4: Visualize discriminative power
# =============================================================================
print("STEP 4: CREATING VISUALIZATIONS")
print("-" * 50)

# Create a comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Discriminative Power Analysis: Which Features Are Most Reliable?', 
             fontsize=16, fontweight='bold')

# 1. Box plots showing spread within each species
for i, feature in enumerate(iris.feature_names):
    row, col = i // 2, i % 2
    
    # Create box plot
    iris_df.boxplot(column=feature, by='species_name', ax=axes[row, col])
    axes[row, col].set_title(f'{feature.title()}\n(Variance: {feature_discriminative_power[feature]:.4f})')
    axes[row, col].set_xlabel('Species')
    axes[row, col].set_ylabel(feature)
    axes[row, col].grid(True, alpha=0.3)

plt.suptitle('')  # Remove default title
plt.tight_layout()
plt.show()

# 2. Bar chart of discriminative power
plt.figure(figsize=(12, 6))

features = [item[0] for item in sorted_features]
discriminative_scores = [1/item[1] for item in sorted_features]  # Higher score = more discriminative

bars = plt.bar(range(len(features)), discriminative_scores, 
               color=['gold', 'lightgreen', 'lightcoral', 'lightblue'])
plt.xlabel('Features')
plt.ylabel('Discriminative Score (1/Variance)')
plt.title('Discriminative Power Ranking\n(Higher Score = More Reliable for Inference)')
plt.xticks(range(len(features)), [f.replace(' (cm)', '').replace(' ', '\n') for f in features])

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, discriminative_scores)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{score:.1f}', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# STEP 5: Statistical measures of discriminative power
# =============================================================================
print("STEP 5: STATISTICAL MEASURES OF DISCRIMINATIVE POWER")
print("-" * 50)

# Calculate coefficient of variation (CV) for each feature
print("Coefficient of Variation (CV = std/mean):")
print("Lower CV = More consistent = Higher discriminative power")
print()

cv_analysis = {}

for feature in iris.feature_names:
    cv_scores = []
    for species in iris.target_names:
        species_data = iris_df[iris_df['species_name'] == species][feature]
        cv = species_data.std() / species_data.mean()
        cv_scores.append(cv)
    
    avg_cv = np.mean(cv_scores)
    cv_analysis[feature] = avg_cv
    print(f"{feature:20} | Average CV: {avg_cv:.4f}")

print()

# =============================================================================
# STEP 6: Practical implications
# =============================================================================
print("STEP 6: PRACTICAL IMPLICATIONS")
print("-" * 50)

print("🎯 WHAT THIS MEANS FOR MAKING INFERENCES:")
print()

best_feature = sorted_features[0][0]
worst_feature = sorted_features[-1][0]

print(f"1. MOST RELIABLE FEATURE: {best_feature}")
print(f"   • Lowest variance across species")
print(f"   • Most consistent within each species")
print(f"   • Highest confidence in predictions")
print(f"   • Saying: '{best_feature} carries the most weight'")
print()

print(f"2. LEAST RELIABLE FEATURE: {worst_feature}")
print(f"   • Highest variance across species")
print(f"   • Most inconsistent within each species")
print(f"   • Lowest confidence in predictions")
print(f"   • Saying: '{worst_feature} has the least discriminative power'")
print()

print("3. REAL-WORLD ANALOGY:")
print("   • Imagine you're a detective trying to identify a suspect")
print("   • Height (consistent) vs. mood (variable) as identifying features")
print("   • Height has higher discriminative power because it's more consistent")
print("   • Mood has lower discriminative power because it varies too much")
print()

print("4. MACHINE LEARNING IMPLICATIONS:")
print("   • Features with high discriminative power should be weighted more")
print("   • Features with low discriminative power might be removed or downweighted")
print("   • This is the basis for feature selection and feature importance")
print()

# =============================================================================
# STEP 7: Common sayings and terminology
# =============================================================================
print("STEP 7: COMMON SAYINGS AND TERMINOLOGY")
print("-" * 50)

print("📚 TERMS FOR HIGH DISCRIMINATIVE POWER:")
print("   • 'Carries more weight'")
print("   • 'Has higher predictive value'")
print("   • 'More reliable indicator'")
print("   • 'Stronger signal'")
print("   • 'Better discriminator'")
print("   • 'More informative feature'")
print()

print("📚 TERMS FOR LOW DISCRIMINATIVE POWER:")
print("   • 'Carries less weight'")
print("   • 'Has lower predictive value'")
print("   • 'Less reliable indicator'")
print("   • 'Weaker signal'")
print("   • 'Poor discriminator'")
print("   • 'Less informative feature'")
print()

print("📚 STATISTICAL SAYINGS:")
print("   • 'The feature with the tightest spread has the highest discriminative power'")
print("   • 'Lower variance means higher confidence'")
print("   • 'Consistency breeds reliability'")
print("   • 'The signal-to-noise ratio determines reliability'")
print()

print("=" * 70)
print("🎉 SUMMARY: DISCRIMINATIVE POWER ANALYSIS COMPLETE!")
print("=" * 70)
print(f"Most reliable feature for species classification: {best_feature}")
print(f"Least reliable feature for species classification: {worst_feature}")
print()
print("Key takeaway: Features with tighter spreads (lower variance) within")
print("each class are more reliable for making inferences and predictions!")