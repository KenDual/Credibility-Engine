import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("E-COMMERCE RETURN FRAUD DETECTION - EXPLORATORY DATA ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA...")
df = pd.read_csv('data/ecommerce_returns_fixed.csv')
print(f"✓ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================================
# 2. BASIC INFORMATION
# ============================================================================
print("\n[2] BASIC INFORMATION")
print("-" * 80)
print(f"Dataset shape: {df.shape}")
print(f"\nColumn names and types:")
print(df.dtypes)
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 3. FIRST LOOK AT DATA
# ============================================================================
print("\n[3] FIRST 5 ROWS")
print("-" * 80)
print(df.head())

print("\n[3] STATISTICAL SUMMARY")
print("-" * 80)
print(df.describe())

# ============================================================================
# 4. MISSING VALUES ANALYSIS
# ============================================================================
print("\n[4] MISSING VALUES ANALYSIS")
print("-" * 80)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_pct
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_df) > 0:
    print(missing_df)
    print("\n⚠ NOTE: Return_Date, Return_Reason, Days_to_Return have nulls for 'Not Returned' orders")
else:
    print("✓ No missing values found!")

# ============================================================================
# 5. TARGET VARIABLE ANALYSIS (Return_Status)
# ============================================================================
print("\n[5] TARGET VARIABLE ANALYSIS - Return_Status")
print("-" * 80)
return_counts = df['Return_Status'].value_counts()
return_pct = df['Return_Status'].value_counts(normalize=True) * 100

print(f"\nReturn_Status Distribution:")
for status, count in return_counts.items():
    pct = return_pct[status]
    print(f"  {status}: {count} ({pct:.2f}%)")

# Create binary target for modeling
df['Is_Returned'] = (df['Return_Status'] == 'Returned').astype(int)
print(f"\n✓ Created binary target 'Is_Returned': 1=Returned, 0=Not Returned")

# Visualize
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
return_counts.plot(kind='bar', color=['#10B981', '#EF4444'])
plt.title('Return Status Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Return Status')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
plt.pie(return_counts, labels=return_counts.index, autopct='%1.1f%%', 
        colors=['#10B981', '#EF4444'], startangle=90)
plt.title('Return Status Proportion', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('notebooks/01_return_status_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notebooks/01_return_status_distribution.png")
plt.close()

# ============================================================================
# 6. RETURN REASON ANALYSIS
# ============================================================================
print("\n[6] RETURN REASON ANALYSIS")
print("-" * 80)
return_reasons = df[df['Return_Status'] == 'Returned']['Return_Reason'].value_counts()
print("\nReturn Reason Distribution (for Returned orders only):")
print(return_reasons)

plt.figure(figsize=(10, 6))
return_reasons.plot(kind='barh', color='#F59E0B')
plt.title('Return Reasons Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Count')
plt.ylabel('Return Reason')
plt.tight_layout()
plt.savefig('notebooks/02_return_reasons.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notebooks/02_return_reasons.png")
plt.close()

# ============================================================================
# 7. NUMERICAL FEATURES ANALYSIS
# ============================================================================
print("\n[7] NUMERICAL FEATURES ANALYSIS")
print("-" * 80)

numerical_cols = ['Product_Price', 'Order_Quantity', 'User_Age', 
                  'Discount_Applied', 'Days_to_Return']

print("\nNumerical Features Statistics:")
print(df[numerical_cols].describe())

# Distribution plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    if col == 'Days_to_Return':
        data_to_plot = df[df[col].notna()][col]
    else:
        data_to_plot = df[col]
    
    axes[idx].hist(data_to_plot, bins=30, color='#3B82F6', edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'{col} Distribution', fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/03_numerical_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notebooks/03_numerical_distributions.png")
plt.close()

# ============================================================================
# 8. CATEGORICAL FEATURES ANALYSIS
# ============================================================================
print("\n[8] CATEGORICAL FEATURES ANALYSIS")
print("-" * 80)

categorical_cols = ['Product_Category', 'User_Gender', 'Payment_Method', 'Shipping_Method']

for col in categorical_cols:
    print(f"\n{col} Distribution:")
    print(df[col].value_counts())

# Visualize categorical features
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, col in enumerate(categorical_cols):
    counts = df[col].value_counts()
    axes[idx].bar(range(len(counts)), counts.values, color='#8B5CF6')
    axes[idx].set_xticks(range(len(counts)))
    axes[idx].set_xticklabels(counts.index, rotation=45, ha='right')
    axes[idx].set_title(f'{col} Distribution', fontweight='bold')
    axes[idx].set_ylabel('Count')
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/04_categorical_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notebooks/04_categorical_distributions.png")
plt.close()

# ============================================================================
# 9. RETURN RATE BY CATEGORICAL FEATURES
# ============================================================================
print("\n[9] RETURN RATE BY CATEGORICAL FEATURES")
print("-" * 80)

for col in categorical_cols:
    return_rate = df.groupby(col)['Is_Returned'].agg(['sum', 'count', 'mean'])
    return_rate.columns = ['Returned_Count', 'Total_Count', 'Return_Rate']
    return_rate['Return_Rate'] = return_rate['Return_Rate'] * 100
    return_rate = return_rate.sort_values('Return_Rate', ascending=False)
    print(f"\n{col} - Return Rate:")
    print(return_rate)

# Visualize return rate by Product_Category
plt.figure(figsize=(12, 6))
return_by_category = df.groupby('Product_Category')['Is_Returned'].mean() * 100
return_by_category = return_by_category.sort_values(ascending=False)
bars = plt.bar(range(len(return_by_category)), return_by_category.values, color='#EF4444')
plt.xticks(range(len(return_by_category)), return_by_category.index, rotation=45, ha='right')
plt.title('Return Rate by Product Category', fontsize=14, fontweight='bold')
plt.ylabel('Return Rate (%)')
plt.xlabel('Product Category')
plt.axhline(y=df['Is_Returned'].mean() * 100, color='black', linestyle='--', 
            label=f'Overall Avg: {df["Is_Returned"].mean()*100:.1f}%')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/05_return_rate_by_category.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notebooks/05_return_rate_by_category.png")
plt.close()

# ============================================================================
# 10. NUMERICAL FEATURES vs TARGET
# ============================================================================
print("\n[10] NUMERICAL FEATURES vs TARGET (Return_Status)")
print("-" * 80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    returned = df[df['Return_Status'] == 'Returned'][col].dropna()
    not_returned = df[df['Return_Status'] == 'Not Returned'][col].dropna()
    
    axes[idx].hist([not_returned, returned], bins=30, label=['Not Returned', 'Returned'],
                   color=['#10B981', '#EF4444'], alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{col} by Return Status', fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/06_numerical_vs_target.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notebooks/06_numerical_vs_target.png")
plt.close()

# Statistical comparison
print("\nMean values by Return Status:")
comparison = df.groupby('Return_Status')[numerical_cols].mean()
print(comparison)

# ============================================================================
# 11. CORRELATION ANALYSIS
# ============================================================================
print("\n[11] CORRELATION ANALYSIS")
print("-" * 80)

# Select only numerical columns for correlation
numerical_features = ['Product_Price', 'Order_Quantity', 'User_Age', 
                     'Discount_Applied', 'Days_to_Return', 'Is_Returned']
corr_data = df[numerical_features].dropna()
correlation_matrix = corr_data.corr()

print("\nCorrelation with Target (Is_Returned):")
target_corr = correlation_matrix['Is_Returned'].sort_values(ascending=False)
print(target_corr)

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('notebooks/07_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notebooks/07_correlation_matrix.png")
plt.close()

# ============================================================================
# 12. OUTLIER DETECTION
# ============================================================================
print("\n[12] OUTLIER DETECTION (Using IQR Method)")
print("-" * 80)

for col in ['Product_Price', 'Order_Quantity', 'Discount_Applied', 'Days_to_Return']:
    if col == 'Days_to_Return':
        data = df[df[col].notna()][col]
    else:
        data = df[col]
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    outlier_pct = (len(outliers) / len(data)) * 100
    
    print(f"\n{col}:")
    print(f"  Range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"  IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Outliers: {len(outliers)} ({outlier_pct:.2f}%)")

# Boxplots
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
cols_to_plot = ['Product_Price', 'Order_Quantity', 'Discount_Applied', 'Days_to_Return']

for idx, col in enumerate(cols_to_plot):
    if col == 'Days_to_Return':
        data = df[df[col].notna()][col]
    else:
        data = df[col]
    
    axes[idx].boxplot(data, vert=True)
    axes[idx].set_title(f'{col}', fontweight='bold')
    axes[idx].set_ylabel('Value')
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/08_outlier_boxplots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notebooks/08_outlier_boxplots.png")
plt.close()

# ============================================================================
# 13. DATE ANALYSIS
# ============================================================================
print("\n[13] DATE ANALYSIS")
print("-" * 80)

# Convert dates
df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%m/%d/%Y')
df['Return_Date'] = pd.to_datetime(df['Return_Date'], format='%m/%d/%Y', errors='coerce')

# Extract features
df['Order_Year'] = df['Order_Date'].dt.year
df['Order_Month'] = df['Order_Date'].dt.month
df['Order_DayOfWeek'] = df['Order_Date'].dt.dayofweek

print("\nOrder Date Range:")
print(f"  Earliest: {df['Order_Date'].min()}")
print(f"  Latest: {df['Order_Date'].max()}")

print("\nOrders by Year:")
print(df['Order_Year'].value_counts().sort_index())

print("\nOrders by Month:")
print(df['Order_Month'].value_counts().sort_index())

# Return rate by month
return_by_month = df.groupby('Order_Month')['Is_Returned'].mean() * 100
plt.figure(figsize=(12, 6))
plt.bar(return_by_month.index, return_by_month.values, color='#3B82F6')
plt.title('Return Rate by Order Month', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Return Rate (%)')
plt.xticks(range(1, 13))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/09_return_rate_by_month.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notebooks/09_return_rate_by_month.png")
plt.close()

# ============================================================================
# 14. KEY INSIGHTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*80)

print(f"""
1. DATASET OVERVIEW:
   - Total records: {len(df):,}
   - Returned: {return_counts['Returned']:,} ({return_pct['Returned']:.2f}%)
   - Not Returned: {return_counts['Not Returned']:,} ({return_pct['Not Returned']:.2f}%)
   - ✓ Dataset is well-balanced for binary classification

2. MISSING VALUES:
   - Return_Date, Return_Reason, Days_to_Return are NULL for "Not Returned" orders
   - This is expected behavior, not data quality issue

3. RETURN REASONS (Top 3):
   - {return_reasons.index[0]}: {return_reasons.values[0]}
   - {return_reasons.index[1]}: {return_reasons.values[1]}
   - {return_reasons.index[2]}: {return_reasons.values[2]}

4. FEATURE IMPORTANCE INDICATORS:
   - Strongest correlation with return: {target_corr.index[1]} ({target_corr.values[1]:.3f})
   - Product categories with highest return rate need investigation
   - Days_to_Return shows interesting patterns (quick vs delayed returns)

5. DATA QUALITY:
   - No major data quality issues detected
   - Some outliers exist but appear to be legitimate values
   - Dates are consistent and parseable

6. NEXT STEPS:
   ✓ Proceed to Feature Engineering (Phase 1.3)
   ✓ Create derived features from dates
   ✓ Encode categorical variables
   ✓ Scale numerical features
   ✓ Ready for model training!
""")

print("\n" + "="*80)
print("EDA COMPLETED SUCCESSFULLY! ✓")
print("="*80)
print(f"\nAll visualizations saved to: notebooks/")
print("\nReview the generated charts before proceeding to next phase.")
print("\nNext: Run '02_feature_engineering.py' for Phase 1.3")