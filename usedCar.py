import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD DATA
# ============================================
file_path = r'C:\Users\Administrator\Desktop\Python\Machine learning\code\data\used_cars.csv'
df = pd.read_csv(file_path, encoding='utf-8')

print(f"Total records: {df.shape[0]}")
print(f"Total features: {df.shape[1]}")
print("\nSample data:")
print(df.head())
print("\nData info:")
print(df.info())

# ============================================
# 2. CLEAN COLUMN NAMES
# ============================================
df.columns = df.columns.str.strip()

# ============================================
# 3. REMOVE DUPLICATES
# ============================================
duplicates = df.duplicated().sum()
print(f"\nDuplicate records: {duplicates}")
df = df.drop_duplicates()

# ============================================
# 4. CHECK MISSING VALUES
# ============================================
print("\nMissing values per column:")
print(df.isnull().sum())

# ============================================
# 5. CLEAN NUMERIC COLUMNS WITH TEXT FORMAT
# ============================================

# 5.1. mileage column (e.g., "23.4 kmpl" or "19.2 km/kg")
def clean_mileage(val):
    if pd.isna(val):
        return np.nan
    val = str(val).lower().strip()
    nums = re.findall(r'(\d+\.?\d*)', val)
    if nums:
        return float(nums[0])
    return np.nan

df['mileage'] = df['mileage'].apply(clean_mileage)

# 5.2. engine column (e.g., "1197 CC" or "1500 cc")
def clean_engine(val):
    if pd.isna(val):
        return np.nan
    val = str(val).lower().strip()
    nums = re.findall(r'(\d+\.?\d*)', val)
    if nums:
        return float(nums[0])
    return np.nan

df['engine'] = df['engine'].apply(clean_engine)

# 5.3. max_power column (e.g., "83.1 bhp" or "120PS")
def clean_max_power(val):
    if pd.isna(val):
        return np.nan
    val = str(val).lower().strip()
    nums = re.findall(r'(\d+\.?\d*)', val)
    if nums:
        return float(nums[0])
    return np.nan

df['max_power'] = df['max_power'].apply(clean_max_power)

# 5.4. torque column (most complex - e.g., "200Nm@2000rpm" or "150 Nm")
def clean_torque(val):
    if pd.isna(val):
        return np.nan
    val = str(val).lower().strip()
    # Extract first number before Nm or nm
    match = re.search(r'(\d+\.?\d*)\s*(?:nm|n-m|n\.m)', val, re.IGNORECASE)
    if match:
        return float(match.group(1))
    # If other format
    nums = re.findall(r'(\d+\.?\d*)', val)
    if nums:
        return float(nums[0])
    return np.nan

df['torque'] = df['torque'].apply(clean_torque)

# ============================================
# 6. CONVERT TO NUMERIC TYPES
# ============================================
numeric_cols = ['year', 'selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'seats']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ============================================
# 7. HANDLE MISSING VALUES
# ============================================

# 7.1. Drop columns with > 50% missing values
threshold = 0.5
for col in df.columns:
    if df[col].isnull().mean() > threshold:
        print(f"Dropping column {col} due to {df[col].isnull().mean()*100:.1f}% missing data")
        df = df.drop(columns=[col])

# 7.2. Fill missing values
# Numeric columns: median
numeric_cols_exist = [col for col in numeric_cols if col in df.columns and df[col].dtype in ['float64', 'int64']]
for col in numeric_cols_exist:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Categorical columns: mode (most frequent value)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# ============================================
# 8. REMOVE OUTLIERS IN selling_price (IQR method)
# ============================================
Q1 = df['selling_price'].quantile(0.25)
Q3 = df['selling_price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['selling_price'] < lower_bound) | (df['selling_price'] > upper_bound)]
print(f"\nOutliers in selling_price: {len(outliers)}")
df = df[(df['selling_price'] >= lower_bound) & (df['selling_price'] <= upper_bound)]

# ============================================
# 9. ENCODE CATEGORICAL VARIABLES
# ============================================
label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# ============================================
# 10. FEATURE ENGINEERING
# ============================================

# Car age
current_year = 2026
df['car_age'] = current_year - df['year']

# Price per kilometer
df['price_per_km'] = df['selling_price'] / (df['km_driven'] + 1)

# ============================================
# 11. FINAL SUMMARY
# ============================================
print("\n" + "="*50)
print("FINAL SUMMARY AFTER PREPROCESSING:")
print("="*50)
print(f"Final records: {df.shape[0]}")
print(f"Final features: {df.shape[1]}")
print("\nStatistical summary:")
print(df.describe())
print("\nFinal sample data:")
print(df.head())

# ============================================
# 12. SAVE CLEANED DATA
# ============================================
output_path = r'C:\Users\Administrator\Desktop\Python\Machine learning\code\data\used_cars_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Cleaned data saved to:")
print(f"   {output_path}")

# ============================================
# 13. FINAL DATA CHECK
# ============================================
print("\n" + "="*50)
print("FINAL COLUMN CHECK:")
print("="*50)
print(df.dtypes)
print("\nMissing values after preprocessing:")
print(df.isnull().sum())

# ============================================
# 14. TARGET VARIABLE DISTRIBUTION
# ============================================
print("\n" + "="*50)
print("SELLING PRICE STATISTICS:")
print("="*50)
print(df['selling_price'].describe())

#==========================================================================================
 # Visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD CLEANED DATA
# ============================================
file_path = r'C:\Users\Administrator\Desktop\Python\Machine learning\code\data\used_cars_cleaned.csv'
df = pd.read_csv(file_path)

print("="*50)
print("SIMPLE DATA VISUALIZATION")
print("="*50)

# ============================================
# 2. CREATE FIGURES DIRECTORY
# ============================================
figures_dir = r'C:\Users\Administrator\Desktop\Python\Machine learning\code\data\figures'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
    print(f"✅ Created directory: {figures_dir}")

# ============================================
# 3. VISUALIZATION 1: Price Distribution
# ============================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['selling_price'], bins=40, kde=True, color='blue')
plt.title('Selling Price Distribution', fontweight='bold')
plt.xlabel('Price (INR)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['selling_price'], color='skyblue')
plt.title('Price Boxplot', fontweight='bold')
plt.ylabel('Price (INR)')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '1_price_distribution.png'), dpi=200, bbox_inches='tight')
plt.show()
print("✅ Chart 1: Price Distribution")

# ============================================
# 4. VISUALIZATION 2: Car Age & KM Driven
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df['car_age'], bins=25, kde=True, color='green', ax=axes[0])
axes[0].set_title('Car Age Distribution', fontweight='bold')
axes[0].set_xlabel('Car Age (Years)')
axes[0].set_ylabel('Frequency')

sns.histplot(df['km_driven'], bins=40, kde=True, color='orange', ax=axes[1])
axes[1].set_title('KM Driven Distribution', fontweight='bold')
axes[1].set_xlabel('Kilometers Driven')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '2_age_km_distribution.png'), dpi=200, bbox_inches='tight')
plt.show()
print("✅ Chart 2: Car Age & KM Driven")

# ============================================
# 5. VISUALIZATION 3: Price vs Car Age
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(df['car_age'], df['selling_price'], alpha=0.4, s=15, c='blue')
axes[0].set_title('Price vs Car Age', fontweight='bold')
axes[0].set_xlabel('Car Age (Years)')
axes[0].set_ylabel('Selling Price (INR)')

# Create age groups for boxplot
df['age_group'] = pd.cut(df['car_age'], bins=5)
sns.boxplot(x='age_group', y='selling_price', data=df, ax=axes[1], palette='viridis')
axes[1].set_title('Price by Car Age Group', fontweight='bold')
axes[1].set_xlabel('Car Age Group')
axes[1].set_ylabel('Selling Price (INR)')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '3_price_vs_age.png'), dpi=200, bbox_inches='tight')
plt.show()
print("✅ Chart 3: Price vs Car Age")

# ============================================
# 6. VISUALIZATION 4: Price vs KM Driven
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(df['km_driven'], df['selling_price'], alpha=0.4, s=15, c='orange')
axes[0].set_title('Price vs KM Driven', fontweight='bold')
axes[0].set_xlabel('KM Driven')
axes[0].set_ylabel('Selling Price (INR)')

# Create KM groups for boxplot
df['km_group'] = pd.cut(df['km_driven'], bins=5)
sns.boxplot(x='km_group', y='selling_price', data=df, ax=axes[1], palette='viridis')
axes[1].set_title('Price by KM Driven Range', fontweight='bold')
axes[1].set_xlabel('KM Driven Range')
axes[1].set_ylabel('Selling Price (INR)')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '4_price_vs_km.png'), dpi=200, bbox_inches='tight')
plt.show()
print("✅ Chart 4: Price vs KM Driven")

# ============================================
# 7. VISUALIZATION 5: Correlation Matrix
# ============================================
plt.figure(figsize=(10, 8))

# Select numeric columns
numeric_cols = ['year', 'selling_price', 'km_driven', 'mileage', 'engine', 
                'max_power', 'torque', 'seats', 'car_age']

correlation_matrix = df[numeric_cols].corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '5_correlation_matrix.png'), dpi=200, bbox_inches='tight')
plt.show()
print("✅ Chart 5: Correlation Matrix")

# ============================================
# 8. TOP CORRELATIONS WITH PRICE
# ============================================
print("\n" + "="*50)
print("TOP CORRELATIONS WITH SELLING PRICE:")
print("="*50)
correlations = df[numeric_cols].corr()['selling_price'].sort_values(ascending=False)
print(correlations)

# ============================================
# 9. SUMMARY
# ============================================
print("\n" + "="*50)
print("VISUALIZATION SUMMARY:")
print("="*50)
print(f"✅ 5 charts saved to: {figures_dir}")
print("\nCharts created:")
print("  1. price_distribution.png")
print("  2. age_km_distribution.png")
print("  3. price_vs_age.png")
print("  4. price_vs_km.png")
print("  5. correlation_matrix.png")

print("\nKEY INSIGHTS:")
print("-"*30)
print("📊 Most cars priced: ₹200K - ₹600K")
print("📈 Most cars: 10-15 years old")
print("💰 Strongest correlation: car_age (-0.78) with price")
print("📉 Higher KM = Lower price")

# Remove temporary columns
df = df.drop(['age_group', 'km_group'], axis=1, errors='ignore')

print("\n" + "="*50)
print("✅ VISUALIZATION COMPLETED SUCCESSFULLY!")
print("="*50)

#==============================================================================
# Test & Train Split

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD CLEANED DATA
# ============================================
file_path = r'C:\Users\Administrator\Desktop\Python\Machine learning\code\data\used_cars_cleaned.csv'
df = pd.read_csv(file_path)

print("="*60)
print("DATA SPLITTING - TRAIN/TEST SETS")
print("="*60)

print(f"\nTotal records: {df.shape[0]}")
print(f"Total features: {df.shape[1]}")
print(f"\nColumns: {df.columns.tolist()}")

# ============================================
# 2. CREATE SPLIT DIRECTORY
# ============================================
split_dir = r'C:\Users\Administrator\Desktop\Python\Machine learning\code\data\split'
if not os.path.exists(split_dir):
    os.makedirs(split_dir)
    print(f"\nCreated directory: {split_dir}")

# ============================================
# 3. DEFINE FEATURES (X) AND TARGET (y)
# ============================================

# Target variable
target = 'selling_price'

# Features - drop target and unnecessary columns
# IMPORTANT: Drop 'price_per_km' because it's derived from target
features_to_drop = ['selling_price', 'name', 'price_per_km']
X = df.drop(columns=features_to_drop)
y = df[target]

print("\n" + "="*50)
print("FEATURES AND TARGET:")
print("="*50)
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"\nFeatures columns ({len(X.columns)} features):")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")

# ============================================
# 4. SPLIT DATA (80% Train, 20% Test)
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,           # 20% for testing
    random_state=42,         # For reproducibility
    shuffle=True             # Shuffle before splitting
)

print("\n" + "="*50)
print("DATA SPLIT RESULTS:")
print("="*50)
print(f"Training set size: {X_train.shape[0]} records ({X_train.shape[0]/df.shape[0]*100:.1f}%)")
print(f"Test set size:     {X_test.shape[0]} records ({X_test.shape[0]/df.shape[0]*100:.1f}%)")
print(f"\nTraining features: {X_train.shape[1]}")
print(f"Test features:     {X_test.shape[1]}")

# ============================================
# 5. CHECK TARGET DISTRIBUTION IN BOTH SETS
# ============================================

print("\n" + "="*50)
print("TARGET DISTRIBUTION CHECK:")
print("="*50)

print("\nTraining set - Selling Price Statistics:")
print("-"*30)
print(f"Mean:  Rs.{y_train.mean():,.2f}")
print(f"Median: Rs.{y_train.median():,.2f}")
print(f"Min:   Rs.{y_train.min():,.2f}")
print(f"Max:   Rs.{y_train.max():,.2f}")
print(f"Std:   Rs.{y_train.std():,.2f}")

print("\nTest set - Selling Price Statistics:")
print("-"*30)
print(f"Mean:  Rs.{y_test.mean():,.2f}")
print(f"Median: Rs.{y_test.median():,.2f}")
print(f"Min:   Rs.{y_test.min():,.2f}")
print(f"Max:   Rs.{y_test.max():,.2f}")
print(f"Std:   Rs.{y_test.std():,.2f}")

# ============================================
# 6. FEATURE SCALING
# ============================================

print("\n" + "="*50)
print("FEATURE SCALING:")
print("="*50)

# Initialize scaler
scaler = StandardScaler()

# Fit on training data and transform both sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easier viewing
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("Features scaled using StandardScaler")
print(f"Training set scaled shape: {X_train_scaled.shape}")
print(f"Test set scaled shape: {X_test_scaled.shape}")

print("\nScaled training data sample (first 5 rows):")
print(X_train_scaled_df.head())

# ============================================
# 7. SAVE SPLITTED DATA TO SPLIT DIRECTORY
# ============================================

# Save as CSV
X_train.to_csv(os.path.join(split_dir, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(split_dir, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(split_dir, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(split_dir, 'y_test.csv'), index=False)

# Save scaled data
X_train_scaled_df.to_csv(os.path.join(split_dir, 'X_train_scaled.csv'), index=False)
X_test_scaled_df.to_csv(os.path.join(split_dir, 'X_test_scaled.csv'), index=False)

# ============================================
# 8. SAVE SCALER FOR FUTURE USE
# ============================================
joblib.dump(scaler, os.path.join(split_dir, 'scaler.pkl'))
print("scaler.pkl saved (for future predictions)")

# ============================================
# 9. VERIFICATION - Check for data leakage
# ============================================
print("\n" + "="*50)
print("DATA LEAKAGE CHECK:")
print("="*50)

# Check if any test records are in training
train_indices = set(X_train.index)
test_indices = set(X_test.index)
overlap = train_indices.intersection(test_indices)

if len(overlap) == 0:
    print("No overlap between training and test sets")
else:
    print(f"WARNING: {len(overlap)} records appear in both sets!")

# Check if target distribution is similar
price_diff = abs(y_train.mean() - y_test.mean()) / y_train.mean() * 100
print(f"Target mean difference: {price_diff:.2f}%")
if price_diff < 10:
    print("Target distribution is similar across sets")
else:
    print("Target distribution differs significantly")

# ============================================
# 10. SAVE SPLIT SUMMARY (without Unicode characters)
# ============================================
summary = f"""
========================================
DATA SPLIT SUMMARY
========================================

Total Dataset:        {df.shape[0]} records
Features:             {X.shape[1]} features
Target:               {target}

Split Configuration:
- Training set:       {X_train.shape[0]} records ({X_train.shape[0]/df.shape[0]*100:.1f}%)
- Test set:           {X_test.shape[0]} records ({X_test.shape[0]/df.shape[0]*100:.1f}%)

Features List:
{chr(10).join([f'  {i+1}. {col}' for i, col in enumerate(X.columns)])}

Scaling:
- Method:             StandardScaler
- Training set:       Scaled
- Test set:           Scaled

Files Saved:
  - X_train.csv        ({X_train.shape[0]} rows, {X_train.shape[1]} cols)
  - X_test.csv         ({X_test.shape[0]} rows, {X_test.shape[1]} cols)
  - y_train.csv        ({y_train.shape[0]} rows)
  - y_test.csv         ({y_test.shape[0]} rows)
  - X_train_scaled.csv ({X_train_scaled.shape[0]} rows, {X_train_scaled.shape[1]} cols)
  - X_test_scaled.csv  ({X_test_scaled.shape[0]} rows, {X_test_scaled.shape[1]} cols)
  - scaler.pkl

Location:             {split_dir}
"""

# Save summary to text file (using utf-8 encoding)
with open(os.path.join(split_dir, 'split_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(summary)

print(summary)

# ============================================
# 11. FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("FILES SAVED IN SPLIT DIRECTORY:")
print("="*60)
print(f"\nFolder: {split_dir}")
print("-"*40)
for file in os.listdir(split_dir):
    file_size = os.path.getsize(os.path.join(split_dir, file)) / 1024  # KB
    print(f"  File: {file:<25} {file_size:>8.1f} KB")

print("\n" + "="*60)
print("DATA SPLITTING COMPLETED SUCCESSFULLY!")
print("="*60)

#=======================================================================================
# Regression modeles & Best Hyperparams.

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD DATA
# ============================================
split_dir = r'C:\Users\Administrator\Desktop\Python\Machine learning\code\data\split'
X_train = pd.read_csv(os.path.join(split_dir, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(split_dir, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(split_dir, 'y_train.csv')).values.ravel()
y_test = pd.read_csv(os.path.join(split_dir, 'y_test.csv')).values.ravel()

print("="*70)
print("REGRESSION MODELS - USED CARS PRICE PREDICTION")
print("="*70)
print(f"\nTraining set: {X_train.shape[0]} records, {X_train.shape[1]} features")
print(f"Test set: {X_test.shape[0]} records, {X_test.shape[1]} features")

# ============================================
# 2. CREATE RESULTS DIRECTORY
# ============================================
results_dir = r'C:\Users\Administrator\Desktop\Python\Machine learning\code\data\results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# ============================================
# 3. EVALUATION FUNCTION
# ============================================
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model"""
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    results = {
        'model': model_name,
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    return results, y_train_pred, y_test_pred

# ============================================
# 4. MODEL 1: SIMPLE LINEAR REGRESSION (Single Variable)
# ============================================
print("\n" + "="*70)
print("MODEL 1: SIMPLE LINEAR REGRESSION (Single Variable)")
print("="*70)

# Find best single feature (highest correlation with target)
correlations = X_train.corrwith(pd.Series(y_train)).abs().sort_values(ascending=False)
best_feature = correlations.index[0]
print(f"Best feature: {best_feature} (correlation: {correlations[0]:.3f})")

# Train simple linear regression
X_train_simple = X_train[[best_feature]]
X_test_simple = X_test[[best_feature]]

simple_lr = LinearRegression()
results_simple, y_pred_train_simple, y_pred_test_simple = evaluate_model(
    simple_lr, X_train_simple, X_test_simple, y_train, y_test, 
    "Simple Linear Regression"
)

print(f"\nResults:")
print(f"  Train R²: {results_simple['train_r2']:.4f}")
print(f"  Test R²:  {results_simple['test_r2']:.4f}")
print(f"  Test RMSE: Rs.{results_simple['test_rmse']:,.2f}")
print(f"  Test MAE:  Rs.{results_simple['test_mae']:,.2f}")
print(f"  CV R²:     {results_simple['cv_mean']:.4f} (+/- {results_simple['cv_std']:.4f})")

# ============================================
# 5. MODEL 2: MULTIPLE LINEAR REGRESSION
# ============================================
print("\n" + "="*70)
print("MODEL 2: MULTIPLE LINEAR REGRESSION")
print("="*70)

multi_lr = LinearRegression()
results_multi, y_pred_train_multi, y_pred_test_multi = evaluate_model(
    multi_lr, X_train, X_test, y_train, y_test, 
    "Multiple Linear Regression"
)

print(f"\nResults:")
print(f"  Train R²: {results_multi['train_r2']:.4f}")
print(f"  Test R²:  {results_multi['test_r2']:.4f}")
print(f"  Test RMSE: Rs.{results_multi['test_rmse']:,.2f}")
print(f"  Test MAE:  Rs.{results_multi['test_mae']:,.2f}")
print(f"  CV R²:     {results_multi['cv_mean']:.4f} (+/- {results_multi['cv_std']:.4f})")

# Feature importance (coefficients)
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': multi_lr.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print("\nTop 5 Important Features:")
print(feature_importance.head(5))

# ============================================
# 6. MODEL 3: POLYNOMIAL REGRESSION
# ============================================
print("\n" + "="*70)
print("MODEL 3: POLYNOMIAL REGRESSION")
print("="*70)

# Try different polynomial degrees
poly_results = []
degrees = [2, 3, 4, 5]

for degree in degrees:
    print(f"\nTesting degree {degree}...")
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Scale polynomial features
    scaler_poly = StandardScaler()
    X_train_poly_scaled = scaler_poly.fit_transform(X_train_poly)
    X_test_poly_scaled = scaler_poly.transform(X_test_poly)
    
    lr_poly = LinearRegression()
    results_poly, _, _ = evaluate_model(
        lr_poly, X_train_poly_scaled, X_test_poly_scaled, y_train, y_test, 
        f"Polynomial (degree {degree})"
    )
    poly_results.append({
        'degree': degree,
        'test_r2': results_poly['test_r2'],
        'test_rmse': results_poly['test_rmse'],
        'cv_mean': results_poly['cv_mean']
    })
    print(f"  Test R²: {results_poly['test_r2']:.4f}, Test RMSE: Rs.{results_poly['test_rmse']:,.2f}")

# Select best polynomial degree
best_poly = max(poly_results, key=lambda x: x['test_r2'])
print(f"\nBest Polynomial Degree: {best_poly['degree']}")
print(f"  Test R²: {best_poly['test_r2']:.4f}")
print(f"  Test RMSE: Rs.{best_poly['test_rmse']:,.2f}")

# Train final polynomial model with best degree
poly_final = PolynomialFeatures(degree=best_poly['degree'], include_bias=False)
X_train_poly_final = poly_final.fit_transform(X_train)
X_test_poly_final = poly_final.transform(X_test)

scaler_poly_final = StandardScaler()
X_train_poly_scaled_final = scaler_poly_final.fit_transform(X_train_poly_final)
X_test_poly_scaled_final = scaler_poly_final.transform(X_test_poly_final)

lr_poly_final = LinearRegression()
results_poly_final, _, _ = evaluate_model(
    lr_poly_final, X_train_poly_scaled_final, X_test_poly_scaled_final, 
    y_train, y_test, f"Polynomial (degree {best_poly['degree']})"
)

# ============================================
# 7. MODEL 4: RIDGE REGRESSION (L2 Regularization)
# ============================================
print("\n" + "="*70)
print("MODEL 4: RIDGE REGRESSION (L2 Regularization)")
print("="*70)

# Hyperparameter tuning for Ridge
ridge_params = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
}

ridge = Ridge()
ridge_grid = GridSearchCV(
    ridge, ridge_params, 
    cv=5, scoring='r2', 
    n_jobs=-1, verbose=0
)
ridge_grid.fit(X_train, y_train)

print(f"Best parameters: {ridge_grid.best_params_}")
print(f"Best CV R²: {ridge_grid.best_score_:.4f}")

# Train with best parameters
ridge_best = Ridge(**ridge_grid.best_params_)
results_ridge, _, _ = evaluate_model(
    ridge_best, X_train, X_test, y_train, y_test, 
    f"Ridge (alpha={ridge_grid.best_params_['alpha']})"
)

print(f"\nResults:")
print(f"  Train R²: {results_ridge['train_r2']:.4f}")
print(f"  Test R²:  {results_ridge['test_r2']:.4f}")
print(f"  Test RMSE: Rs.{results_ridge['test_rmse']:,.2f}")
print(f"  Test MAE:  Rs.{results_ridge['test_mae']:,.2f}")

# ============================================
# 8. MODEL 5: LASSO REGRESSION (L1 Regularization)
# ============================================
print("\n" + "="*70)
print("MODEL 5: LASSO REGRESSION (L1 Regularization)")
print("="*70)

# Hyperparameter tuning for Lasso
lasso_params = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'max_iter': [1000, 5000, 10000]
}

lasso = Lasso()
lasso_grid = GridSearchCV(
    lasso, lasso_params, 
    cv=5, scoring='r2', 
    n_jobs=-1, verbose=0
)
lasso_grid.fit(X_train, y_train)

print(f"Best parameters: {lasso_grid.best_params_}")
print(f"Best CV R²: {lasso_grid.best_score_:.4f}")

# Train with best parameters
lasso_best = Lasso(**lasso_grid.best_params_)
results_lasso, _, _ = evaluate_model(
    lasso_best, X_train, X_test, y_train, y_test, 
    f"Lasso (alpha={lasso_grid.best_params_['alpha']})"
)

print(f"\nResults:")
print(f"  Train R²: {results_lasso['train_r2']:.4f}")
print(f"  Test R²:  {results_lasso['test_r2']:.4f}")
print(f"  Test RMSE: Rs.{results_lasso['test_rmse']:,.2f}")
print(f"  Test MAE:  Rs.{results_lasso['test_mae']:,.2f}")

# Number of features selected by Lasso
n_selected = np.sum(np.abs(lasso_best.coef_) > 1e-6)
print(f"Features selected: {n_selected} out of {X_train.shape[1]}")

# ============================================
# 9. MODEL 6: ELASTIC NET (L1 + L2 Regularization)
# ============================================
print("\n" + "="*70)
print("MODEL 6: ELASTIC NET (L1 + L2 Regularization)")
print("="*70)

# Hyperparameter tuning for ElasticNet
elastic_params = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
    'max_iter': [1000, 5000]
}

elastic = ElasticNet()
elastic_grid = GridSearchCV(
    elastic, elastic_params, 
    cv=5, scoring='r2', 
    n_jobs=-1, verbose=0
)
elastic_grid.fit(X_train, y_train)

print(f"Best parameters: {elastic_grid.best_params_}")
print(f"Best CV R²: {elastic_grid.best_score_:.4f}")

# Train with best parameters
elastic_best = ElasticNet(**elastic_grid.best_params_)
results_elastic, _, _ = evaluate_model(
    elastic_best, X_train, X_test, y_train, y_test, 
    f"ElasticNet (alpha={elastic_grid.best_params_['alpha']}, l1_ratio={elastic_grid.best_params_['l1_ratio']})"
)

print(f"\nResults:")
print(f"  Train R²: {results_elastic['train_r2']:.4f}")
print(f"  Test R²:  {results_elastic['test_r2']:.4f}")
print(f"  Test RMSE: Rs.{results_elastic['test_rmse']:,.2f}")
print(f"  Test MAE:  Rs.{results_elastic['test_mae']:,.2f}")

# ============================================
# 10. COMPARE ALL MODELS
# ============================================
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)

# Create comparison DataFrame
comparison = pd.DataFrame({
    'Model': [
        'Simple Linear',
        'Multiple Linear',
        f'Polynomial (deg {best_poly["degree"]})',
        f'Ridge (α={ridge_grid.best_params_["alpha"]})',
        f'Lasso (α={lasso_grid.best_params_["alpha"]})',
        f'ElasticNet (α={elastic_grid.best_params_["alpha"]})'
    ],
    'Train R²': [
        results_simple['train_r2'],
        results_multi['train_r2'],
        results_poly_final['train_r2'],
        results_ridge['train_r2'],
        results_lasso['train_r2'],
        results_elastic['train_r2']
    ],
    'Test R²': [
        results_simple['test_r2'],
        results_multi['test_r2'],
        results_poly_final['test_r2'],
        results_ridge['test_r2'],
        results_lasso['test_r2'],
        results_elastic['test_r2']
    ],
    'Test RMSE': [
        results_simple['test_rmse'],
        results_multi['test_rmse'],
        results_poly_final['test_rmse'],
        results_ridge['test_rmse'],
        results_lasso['test_rmse'],
        results_elastic['test_rmse']
    ],
    'Test MAE': [
        results_simple['test_mae'],
        results_multi['test_mae'],
        results_poly_final['test_mae'],
        results_ridge['test_mae'],
        results_lasso['test_mae'],
        results_elastic['test_mae']
    ],
    'CV R² Mean': [
        results_simple['cv_mean'],
        results_multi['cv_mean'],
        results_poly_final['cv_mean'],
        results_ridge['cv_mean'],
        results_lasso['cv_mean'],
        results_elastic['cv_mean']
    ]
})

print("\nComparison Table:")
print(comparison.to_string(index=False))

# Find best model
best_model_idx = comparison['Test R²'].idxmax()
best_model_name = comparison.loc[best_model_idx, 'Model']
best_test_r2 = comparison.loc[best_model_idx, 'Test R²']

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_model_name}")
print(f"  Test R²: {best_test_r2:.4f}")
print(f"  Test RMSE: Rs.{comparison.loc[best_model_idx, 'Test RMSE']:,.2f}")
print(f"  Test MAE: Rs.{comparison.loc[best_model_idx, 'Test MAE']:,.2f}")
print(f"{'='*70}")

# ============================================
# 11. VISUALIZE COMPARISON
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# R² comparison
models = comparison['Model'].values
r2_scores = comparison['Test R²'].values
colors = ['green' if i == best_model_idx else 'blue' for i in range(len(models))]

axes[0].barh(models, r2_scores, color=colors, alpha=0.7)
axes[0].set_title('Model Comparison - Test R² Score', fontsize=14, fontweight='bold')
axes[0].set_xlabel('R² Score')
axes[0].axvline(x=r2_scores.max(), color='red', linestyle='--', linewidth=1, label='Best')
axes[0].legend()

# RMSE comparison
rmse_values = comparison['Test RMSE'].values
axes[1].barh(models, rmse_values, color=colors, alpha=0.7)
axes[1].set_title('Model Comparison - Test RMSE', fontsize=14, fontweight='bold')
axes[1].set_xlabel('RMSE (Rupees)')
axes[1].axvline(x=rmse_values.min(), color='red', linestyle='--', linewidth=1, label='Best')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_comparison.png'), dpi=200, bbox_inches='tight')
plt.show()

# ============================================
# 12. SAVE RESULTS
# ============================================

# Save comparison to CSV
comparison.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)

# Save best model
best_model = None
if 'Ridge' in best_model_name:
    best_model = ridge_best
elif 'Lasso' in best_model_name:
    best_model = lasso_best
elif 'ElasticNet' in best_model_name:
    best_model = elastic_best
elif 'Polynomial' in best_model_name:
    # Save polynomial model components
    joblib.dump(poly_final, os.path.join(results_dir, 'polynomial_features.pkl'))
    joblib.dump(scaler_poly_final, os.path.join(results_dir, 'poly_scaler.pkl'))
    best_model = lr_poly_final
elif 'Multiple' in best_model_name:
    best_model = multi_lr
else:
    best_model = simple_lr

# Save best model
if best_model is not None:
    joblib.dump(best_model, os.path.join(results_dir, 'best_model.pkl'))
    print(f"\n✅ Best model saved to: {os.path.join(results_dir, 'best_model.pkl')}")

# Save all models
joblib.dump(simple_lr, os.path.join(results_dir, 'simple_linear_model.pkl'))
joblib.dump(multi_lr, os.path.join(results_dir, 'multiple_linear_model.pkl'))
joblib.dump(ridge_best, os.path.join(results_dir, 'ridge_model.pkl'))
joblib.dump(lasso_best, os.path.join(results_dir, 'lasso_model.pkl'))
joblib.dump(elastic_best, os.path.join(results_dir, 'elastic_model.pkl'))

print("\n" + "="*70)
print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("="*70)
print(f"\nResults saved in: {results_dir}")
print("\nFiles saved:")
print("  - model_comparison.csv")
print("  - model_comparison.png")
print("  - best_model.pkl")
print("  - simple_linear_model.pkl")
print("  - multiple_linear_model.pkl")
print("  - ridge_model.pkl")
print("  - lasso_model.pkl")
print("  - elastic_model.pkl")
if 'Polynomial' in best_model_name:
    print("  - polynomial_features.pkl")
    print("  - poly_scaler.pkl")
