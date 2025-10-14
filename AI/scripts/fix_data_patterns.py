"""
Fix Synthetic Dataset - Add Realistic Return Patterns
Tạo correlations thực tế giữa features và return behavior
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

print("Loading original dataset...")
df = pd.read_csv('data/ecommerce_returns_synthetic_data.csv')
print(f"Original shape: {df.shape}")

# Parse dates - handle multiple formats
try:
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%m/%d/%Y')
except:
    # Already in datetime format or ISO format
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])

# ============================================================================
# DEFINE REALISTIC RETURN PATTERNS
# ============================================================================

def calculate_return_probability(row):
    """
    Tính xác suất return dựa trên business logic thực tế
    Base probability: 30%
    """
    prob = 0.30  # Base return rate
    
    # 1. PRODUCT CATEGORY EFFECT (±15%)
    category_effect = {
        'Clothing': 0.15,    # Size/fit issues - HIGH return
        'Electronics': -0.05, # Test before return - LOWER return  
        'Books': -0.10,      # Clear expectations - LOW return
        'Toys': 0.05,        # Kids change mind - HIGHER return
        'Home': 0.00         # Neutral
    }
    prob += category_effect.get(row['Product_Category'], 0)
    
    # 2. PRICE EFFECT (±20%)
    # Cheap items (<$50): less likely to return (hassle)
    # Expensive items (>$400): more scrutiny, higher return
    if row['Product_Price'] < 50:
        prob -= 0.10  # Too cheap to bother returning
    elif row['Product_Price'] > 400:
        prob += 0.20  # Expensive = careful consideration = higher return
    elif row['Product_Price'] > 300:
        prob += 0.10
    
    # 3. DISCOUNT EFFECT (±15%)
    # High discount = impulse buy = higher return
    discount_pct = row['Discount_Applied'] / max(row['Product_Price'], 1) * 100
    if discount_pct > 40:
        prob += 0.15  # Deep discount impulse buy
    elif discount_pct > 30:
        prob += 0.08
    elif discount_pct < 10:
        prob -= 0.05  # Full price = committed buyer
    
    # 4. QUANTITY EFFECT (±10%)
    # Bulk orders = business/planned = lower return
    if row['Order_Quantity'] >= 4:
        prob -= 0.10  # Bulk = planned purchase
    elif row['Order_Quantity'] == 1:
        prob += 0.05  # Single item = trial purchase
    
    # 5. AGE EFFECT (±10%)
    # Young users return more (online shopping habits)
    if row['User_Age'] < 25:
        prob += 0.10  # Gen Z returns more
    elif row['User_Age'] > 60:
        prob -= 0.05  # Seniors return less
    
    # 6. PAYMENT METHOD EFFECT (±5%)
    payment_effect = {
        'Credit Card': 0.05,   # Easy refund process
        'PayPal': 0.03,       # Protection = confidence to return
        'Debit Card': 0.00,   # Neutral
        'Gift Card': -0.05    # Already "spent" = keep it
    }
    prob += payment_effect.get(row['Payment_Method'], 0)
    
    # 7. SHIPPING METHOD EFFECT (±5%)
    shipping_effect = {
        'Next-Day': 0.05,     # Impulse/urgent = higher return
        'Express': 0.02,      
        'Standard': -0.03     # Planned purchase = lower return
    }
    prob += shipping_effect.get(row['Shipping_Method'], 0)
    
    # 8. GENDER EFFECT (small but realistic)
    if row['User_Gender'] == 'Female' and row['Product_Category'] == 'Clothing':
        prob += 0.08  # Fashion returns higher for female shoppers
    
    # 9. SEASONAL EFFECT (±5%)
    month = row['Order_Date'].month
    if month in [11, 12, 1]:  # Holiday season
        prob += 0.05  # Gift returns, impulse buys
    elif month in [6, 7, 8]:  # Summer
        prob -= 0.03  # Vacation purchases kept
    
    # Ensure probability is between 0 and 1
    return np.clip(prob, 0.05, 0.95)

# ============================================================================
# APPLY RETURN LOGIC
# ============================================================================

print("\nApplying realistic return patterns...")

# Calculate return probability for each order
df['return_probability'] = df.apply(calculate_return_probability, axis=1)

# Generate returns based on probability
df['will_return'] = np.random.random(len(df)) < df['return_probability']

# Update Return_Status
df['Return_Status'] = df['will_return'].map({True: 'Returned', False: 'Not Returned'})

# ============================================================================
# FIX RETURN-SPECIFIC FIELDS
# ============================================================================

print("Fixing return-specific fields...")

# Clear return fields for non-returned orders
df.loc[~df['will_return'], 'Return_Date'] = np.nan
df.loc[~df['will_return'], 'Return_Reason'] = np.nan
df.loc[~df['will_return'], 'Days_to_Return'] = np.nan

# Generate realistic return patterns for returned orders
for idx in df[df['will_return']].index:
    row = df.loc[idx]
    
    # Days to return based on reason (realistic distribution)
    reason_days_map = {
        'Defective': (1, 7),        # Quick return for defective
        'Wrong item': (1, 5),       # Quick discovery
        'Not as described': (2, 10), # After trying
        'Changed mind': (5, 30)     # Takes time to decide
    }
    
    # Assign return reason with realistic distribution
    if row['Product_Category'] == 'Electronics':
        reasons = ['Defective'] * 40 + ['Not as described'] * 30 + ['Changed mind'] * 20 + ['Wrong item'] * 10
    elif row['Product_Category'] == 'Clothing':
        reasons = ['Not as described'] * 35 + ['Changed mind'] * 30 + ['Wrong item'] * 20 + ['Defective'] * 15
    else:
        reasons = ['Changed mind'] * 35 + ['Not as described'] * 25 + ['Defective'] * 20 + ['Wrong item'] * 20
    
    return_reason = np.random.choice(reasons)
    df.at[idx, 'Return_Reason'] = return_reason
    
    # Generate days to return
    min_days, max_days = reason_days_map.get(return_reason, (1, 30))
    
    # High-value items returned faster
    if row['Product_Price'] > 400:
        max_days = min(max_days, 14)
    
    days_to_return = np.random.randint(min_days, max_days + 1)
    df.at[idx, 'Days_to_Return'] = days_to_return
    
    # Calculate return date
    return_date = row['Order_Date'] + timedelta(days=int(days_to_return))
    # Keep consistent format with original data
    df.at[idx, 'Return_Date'] = return_date.strftime('%m/%d/%Y').lstrip('0').replace('/0', '/')

# Drop helper columns
df = df.drop(['return_probability', 'will_return'], axis=1)

# Convert Order_Date back to original format  
# Check original format and maintain consistency
if df['Order_Date'].dtype == 'object':
    # Already string, keep as is
    pass
else:
    # Convert to m/d/yyyy format
    df['Order_Date'] = df['Order_Date'].dt.strftime('%m/%d/%Y').str.lstrip('0').str.replace('/0', '/')

# ============================================================================
# VERIFY PATTERNS
# ============================================================================

print("\n" + "="*60)
print("VERIFICATION OF NEW PATTERNS")
print("="*60)

# Overall return rate
return_rate = (df['Return_Status'] == 'Returned').mean()
print(f"\nOverall return rate: {return_rate:.1%}")

# Return rate by category
print("\nReturn rate by Product Category:")
for cat in df['Product_Category'].unique():
    rate = (df[df['Product_Category'] == cat]['Return_Status'] == 'Returned').mean()
    print(f"  {cat:15} {rate:.1%}")

# Return rate by price range
print("\nReturn rate by Price Range:")
price_ranges = [(0, 50), (50, 200), (200, 400), (400, 1000)]
for low, high in price_ranges:
    mask = (df['Product_Price'] >= low) & (df['Product_Price'] < high)
    rate = (df[mask]['Return_Status'] == 'Returned').mean()
    count = mask.sum()
    print(f"  ${low:3}-${high:4}: {rate:.1%} ({count} orders)")

# Return rate by age group
print("\nReturn rate by Age Group:")
age_ranges = [(18, 25), (25, 35), (35, 50), (50, 70)]
for low, high in age_ranges:
    mask = (df['User_Age'] >= low) & (df['User_Age'] < high)
    rate = (df[mask]['Return_Status'] == 'Returned').mean()
    print(f"  Age {low}-{high}: {rate:.1%}")

# High discount effect
print("\nReturn rate by Discount Level:")
df['Discount_Pct'] = (df['Discount_Applied'] / df['Product_Price'].clip(lower=1)) * 100
for threshold in [10, 30, 40]:
    mask = df['Discount_Pct'] >= threshold
    if mask.sum() > 0:
        rate = (df[mask]['Return_Status'] == 'Returned').mean()
        print(f"  Discount >={threshold}%: {rate:.1%} ({mask.sum()} orders)")

# ============================================================================
# SAVE FIXED DATASET
# ============================================================================

# Drop temporary column
df = df.drop('Discount_Pct', axis=1)

# Save to new file
output_file = 'data/ecommerce_returns_fixed.csv'
df.to_csv(output_file, index=False)
print(f"\n✅ Saved fixed dataset to: {output_file}")
print(f"Shape: {df.shape}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*60)
print("DATASET FIXING COMPLETED!")
print("="*60)

print("""
✅ PATTERNS ADDED:
1. Clothing has highest return rate (size/fit issues)
2. Books have lowest return rate (clear expectations)  
3. Expensive items (>$400) have higher scrutiny → more returns
4. High discounts (>40%) → impulse buys → more returns
5. Young users (<25) return more frequently
6. Bulk orders (4+ items) return less (planned purchases)
7. Gift card payments return less
8. Holiday season has higher returns

📊 EXPECTED MODEL PERFORMANCE:
- AUC: 0.65-0.75 (realistic, not perfect)
- Clear feature importance patterns
- Business-interpretable results

🎯 NEXT STEPS:
1. Rename file: mv data/ecommerce_returns_fixed.csv data/ecommerce_returns_synthetic_data.csv
2. Re-run: python 02_feature_engineering_v2.py
3. Re-run: python 03_train_xgboost.py
4. Expected: Reasonable AUC with interpretable patterns
""")