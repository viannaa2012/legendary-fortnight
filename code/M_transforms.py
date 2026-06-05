# M_transforms.py
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def count_noisy_records(data, numeric_cols):
    """Count different types of noisy records"""
    empty_records = data.isna().all(axis=1).sum()
    records_with_empty = data.isna().any(axis=1).sum()
    
    non_numeric_rows = set()
    for col in numeric_cols:
        if col in data.columns:
            for idx, value in data[col].items():
                if pd.notna(value):
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        non_numeric_rows.add(idx)
    
    non_numeric_count = len(non_numeric_rows)
    total_noisy = len(set(data[data.isna().any(axis=1)].index).union(non_numeric_rows))
    
    return {
        'empty_records': empty_records,
        'records_with_empty': records_with_empty,
        'non_numeric_count': non_numeric_count,
        'total_noisy': total_noisy,
        'non_numeric_rows': non_numeric_rows
    }

def show_noisy_details(data, numeric_cols, stats):
    """Display detailed information about noisy records"""
    print("\n📋 Noisy Records Details:")
    
    if stats['empty_records'] > 0:
        print(f"\n  --- Completely Empty Records ({stats['empty_records']}) ---")
        for idx in data[data.isna().all(axis=1)].index:
            print(f"    Row {idx}")
    
    single_empty = data[data.isna().any(axis=1) & ~data.isna().all(axis=1)]
    if len(single_empty) > 0:
        print(f"\n  --- Records with Empty Cells ({len(single_empty)}) ---")
        for idx in single_empty.index:
            empty_cols = data.columns[data.loc[idx].isna()].tolist()
            client = data.loc[idx, 'client_id'] if 'client_id' in data.columns else 'Unknown'
            loan = data.loc[idx, 'loan_id'] if 'loan_id' in data.columns else 'Unknown'
            print(f"    Row {idx} (Client: {client}, Loan: {loan}): Empty in {empty_cols}")
    
    if stats['non_numeric_count'] > 0:
        print(f"\n  --- Records with Non-numeric Data ({stats['non_numeric_count']}) ---")
        for col in numeric_cols:
            for idx in stats['non_numeric_rows']:
                val = data.loc[idx, col] if col in data.columns else None
                if pd.notna(val):
                    try:
                        float(val)
                    except:
                        client = data.loc[idx, 'client_id'] if 'client_id' in data.columns else 'Unknown'
                        print(f"    Row {idx} (Client: {client}): '{val}' in {col}")

def remove_completely_empty(data):
    """Remove rows that are completely empty"""
    empty_records = data.isna().all(axis=1).sum()
    if empty_records > 0:
        print(f"\n🗑 Removing {empty_records} completely empty record(s)...")
        data = data.dropna(how='all')
        print(f"✓ Remaining records: {len(data)}")
    return data

def convert_to_numeric(data, numeric_cols):
    """Convert non-numeric values to NaN"""
    print("\n📊 Step 1: Converting non-numeric values...")
    for col in numeric_cols:
        if col in data.columns:
            before = data[col].copy()
            data[col] = pd.to_numeric(data[col], errors='coerce')
            converted = (pd.to_numeric(before, errors='coerce').isna() & before.notna()).sum()
            if converted > 0:
                print(f"  ✓ Converted {converted} value(s) in '{col}'")
    return data

def handle_empty_cells(data, numeric_cols):
    """Handle empty cells based on user choice"""
    print("\n🔧 Step 2: Handling empty cells...")
    remaining_empty = data.isna().any(axis=1).sum()
    
    if remaining_empty == 0:
        print("  ✓ No empty cells found!")
        return data
    
    print(f"  Found {remaining_empty} record(s) with empty cells")
    
    empty_counts = data.isna().sum()
    for col, count in empty_counts[empty_counts > 0].items():
        print(f"    - {col}: {count} empty")
    
    print("\n  Options:")
    print("    1. Fill with Mean (for numeric) / 'Unknown' (for text)")
    print("    2. Fill with Median (for numeric) / 'Unknown' (for text)")
    print("    3. Fill with Mode")
    print("    4. Remove rows with empty cells")
    print("    5. Keep as is")
    
    choice = input("\n  Choose (1-5): ").strip()
    
    if choice == '1':
        print("\n  --- Filling with Mean ---")
        for col in numeric_cols:
            if col in data.columns and data[col].isna().any():
                mean_val = data[col].mean()
                if pd.notna(mean_val):
                    data[col].fillna(mean_val, inplace=True)
                    print(f"    ✓ '{col}' filled with mean: {mean_val:.2f}")
        
        non_numeric = [c for c in data.columns if c not in numeric_cols]
        for col in non_numeric:
            if col in data.columns and data[col].isna().any():
                data[col].fillna('Unknown', inplace=True)
                print(f"    ✓ '{col}' filled with 'Unknown'")
            
    elif choice == '2':
        print("\n  --- Filling with Median ---")
        for col in numeric_cols:
            if col in data.columns and data[col].isna().any():
                median_val = data[col].median()
                if pd.notna(median_val):
                    data[col].fillna(median_val, inplace=True)
                    print(f"    ✓ '{col}' filled with median: {median_val:.2f}")
        
        non_numeric = [c for c in data.columns if c not in numeric_cols]
        for col in non_numeric:
            if col in data.columns and data[col].isna().any():
                data[col].fillna('Unknown', inplace=True)
                print(f"    ✓ '{col}' filled with 'Unknown'")
            
    elif choice == '3':
        print("\n  --- Filling with Mode ---")
        for col in numeric_cols:
            if col in data.columns and data[col].isna().any():
                mode_val = data[col].mode()
                if len(mode_val) > 0:
                    data[col].fillna(mode_val[0], inplace=True)
                    print(f"    ✓ '{col}' filled with mode: {mode_val[0]:.2f}")
                else:
                    median_val = data[col].median()
                    if pd.notna(median_val):
                        data[col].fillna(median_val, inplace=True)
                        print(f"    ✓ No mode, used median: {median_val:.2f}")
        
        non_numeric = [c for c in data.columns if c not in numeric_cols]
        for col in non_numeric:
            if col in data.columns and data[col].isna().any():
                mode_val = data[col].mode()
                if len(mode_val) > 0:
                    data[col].fillna(mode_val[0], inplace=True)
                    print(f"    ✓ '{col}' filled with mode: {mode_val[0]}")
                else:
                    data[col].fillna('Unknown', inplace=True)
                    print(f"    ✓ '{col}' filled with 'Unknown'")
                
    elif choice == '4':
        before = len(data)
        data = data.dropna()
        print(f"\n  ✓ Removed {before - len(data)} record(s)")
    else:
        print("\n  ✓ Keeping empty cells as is")
    
    return data

def convert_to_integers(data, numeric_cols):
    """Convert numeric columns to integer types"""
    print("\n🔢 Step 3: Converting to integers...")
    for col in numeric_cols:
        if col in data.columns:
            if data[col].isna().any():
                data[col] = data[col].round().astype('Int64')
            else:
                data[col] = data[col].round().astype(int)
            print(f"  ✓ Converted '{col}' to {data[col].dtype}")
    return data

def display_summary(original_count, cleaned_data, numeric_cols):
    """Display final summary statistics"""
    print("\n" + "="*50)
    print("📊 Final Summary:")
    print(f"  Original: {original_count} records")
    print(f"  Final: {len(cleaned_data)} records")
    print(f"  Removed: {original_count - len(cleaned_data)} records")
    
    final_empty = cleaned_data.isna().any(axis=1).sum()
    if final_empty > 0:
        print(f"  ⚠ Warning: {final_empty} record(s) still have empty cells")
    else:
        print("  ✓ All empty cells handled")
    
    if 'loan_amount' in cleaned_data.columns:
        print(f"\n  Loan Statistics:")
        print(f"    Total loan amount: {cleaned_data['loan_amount'].sum():,.2f}")
        print(f"    Average loan: {cleaned_data['loan_amount'].mean():,.2f}")
        print(f"    Min loan: {cleaned_data['loan_amount'].min():,.2f}")
        print(f"    Max loan: {cleaned_data['loan_amount'].max():,.2f}")
    
    if 'repaid' in cleaned_data.columns:
        if cleaned_data['repaid'].dtype in ['int64', 'float64', 'Int64']:
            repaid_count = cleaned_data['repaid'].sum()
            print(f"\n  Repayment Status:")
            print(f"    Repaid loans: {repaid_count}")
            print(f"    Pending loans: {len(cleaned_data) - repaid_count}")
    
    print("\n📋 Sample of cleaned data:")
    print(cleaned_data.head())

# ============ OUTLIER DETECTION FUNCTIONS ============

def detect_outliers_iqr(data, numeric_cols, multiplier=1.5):
    """Detect outliers using IQR method"""
    outlier_info = {}
    all_outlier_rows = set()
    
    print("\n" + "="*60)
    print("📊 OUTLIER DETECTION USING IQR METHOD")
    print("="*60)
    
    existing_numeric_cols = [col for col in numeric_cols if col in data.columns]
    
    if not existing_numeric_cols:
        print("No numeric columns found for outlier detection!")
        return {
            'outlier_info': {},
            'total_outlier_rows': 0,
            'all_outlier_rows': set()
        }
    
    for col in existing_numeric_cols:
        col_data = data[col].dropna()
        
        if len(col_data) == 0:
            print(f"\n📍 Column: {col} - No valid data")
            continue
            
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
        outlier_indices = set(data[outliers_mask].index)
        outlier_count = len(outlier_indices)
        
        outlier_indices_list = list(outlier_indices)
        
        outlier_info[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': outlier_count,
            'outlier_indices': outlier_indices,
            'outlier_values': data.loc[outlier_indices_list, col].to_dict() if outlier_count > 0 else {}
        }
        
        all_outlier_rows.update(outlier_indices)
        
        print(f"\n📍 Column: {col}")
        print(f"   Q1 (25th percentile): {Q1:,.2f}")
        print(f"   Q3 (75th percentile): {Q3:,.2f}")
        print(f"   IQR: {IQR:,.2f}")
        print(f"   Normal range: [{lower_bound:,.2f}, {upper_bound:,.2f}]")
        print(f"   ⚠ Outliers found: {outlier_count}")
        
        if outlier_count > 0 and outlier_count <= 10:
            values_list = list(outlier_info[col]['outlier_values'].values())
            print(f"   Outlier values: {values_list}")
        elif outlier_count > 10:
            values_list = list(outlier_info[col]['outlier_values'].values())[:10]
            print(f"   First 10 outlier values: {values_list}")
    
    print("\n" + "="*60)
    print("📊 OUTLIER SUMMARY")
    print("="*60)
    print(f"Total unique rows with outliers: {len(all_outlier_rows)}")
    
    if len(all_outlier_rows) > 0:
        print("\nOutlier distribution by column:")
        for col, info in outlier_info.items():
            if info['outlier_count'] > 0:
                total_values = len(data[col].dropna())
                percentage = (info['outlier_count'] / total_values * 100) if total_values > 0 else 0
                print(f"   • {col}: {info['outlier_count']} outliers ({percentage:.1f}% of data)")
    
    return {
        'outlier_info': outlier_info,
        'total_outlier_rows': len(all_outlier_rows),
        'all_outlier_rows': all_outlier_rows
    }

def show_outlier_details(data, outlier_info):
    """Display detailed information about outliers"""
    if not outlier_info:
        print("\nNo outlier information to display.")
        return
        
    print("\n" + "="*60)
    print("📋 DETAILED OUTLIER REPORT")
    print("="*60)
    
    has_outliers = False
    for col, info in outlier_info.items():
        if info['outlier_count'] > 0:
            has_outliers = True
            print(f"\n📍 Column: {col}")
            print(f"   Total outliers: {info['outlier_count']}")
            print(f"   Normal range: [{info['lower_bound']:,.2f}, {info['upper_bound']:,.2f}]")
            
            print(f"\n   Outlier records:")
            sorted_outliers = sorted(info['outlier_values'].items(), key=lambda x: x[1], reverse=True)
            
            for idx, value in sorted_outliers[:15]:
                context_parts = []
                
                if 'client_id' in data.columns and idx in data.index:
                    context_parts.append(f"Client: {data.loc[idx, 'client_id']}")
                
                if 'loan_id' in data.columns and idx in data.index:
                    context_parts.append(f"Loan ID: {data.loc[idx, 'loan_id']}")
                
                context = " | ".join(context_parts) if context_parts else ""
                print(f"     Row {idx}: {value:,.2f} {context}")
                
            if info['outlier_count'] > 15:
                print(f"     ... and {info['outlier_count'] - 15} more")
    
    if not has_outliers:
        print("\n✅ No outliers found in any column!")

def handle_outliers(data, numeric_cols, outlier_result):
    """Handle detected outliers"""
    if outlier_result['total_outlier_rows'] == 0:
        print("\n✅ No outliers detected!")
        return data
    
    print("\n" + "="*60)
    print("🔧 OUTLIER HANDLING OPTIONS")
    print("="*60)
    print(f"Total rows with outliers: {outlier_result['total_outlier_rows']}")
    
    print("\nOptions:")
    print("  1. Remove all rows containing outliers")
    print("  2. Cap outliers (replace with bounds)")
    print("  3. Keep outliers (no action)")
    print("  4. View outlier details before deciding")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    if choice == '1':
        before = len(data)
        rows_to_remove = list(outlier_result['all_outlier_rows'])
        data = data.drop(index=rows_to_remove)
        print(f"\n✅ Removed {before - len(data)} row(s) containing outliers")
        print(f"   Remaining records: {len(data)}")
        
    elif choice == '2':
        print("\n📊 Capping outliers...")
        for col, info in outlier_result['outlier_info'].items():
            if info['outlier_count'] > 0 and col in data.columns:
                lower_bound = info['lower_bound']
                upper_bound = info['upper_bound']
                
                before_cap = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                
                print(f"   ✓ Capped {before_cap} value(s) in '{col}'")
                print(f"     Range: [{lower_bound:,.2f}, {upper_bound:,.2f}]")
                
    elif choice == '4':
        show_outlier_details(data, outlier_result['outlier_info'])
        return handle_outliers(data, numeric_cols, outlier_result)
    else:
        print("\n✅ Keeping outliers as is")
    
    return data

def get_outlier_summary_table(outlier_result, data):
    """Generate a summary table of outliers"""
    if outlier_result['total_outlier_rows'] == 0:
        return pd.DataFrame()
    
    summary_rows = []
    for col, info in outlier_result['outlier_info'].items():
        if info['outlier_count'] > 0 and col in data.columns:
            total_values = len(data[col].dropna())
            summary_rows.append({
                'Column': col,
                'Outlier Count': info['outlier_count'],
                'Percentage': f"{(info['outlier_count']/total_values*100):.1f}%",
                'Lower Bound': f"{info['lower_bound']:,.2f}",
                'Upper Bound': f"{info['upper_bound']:,.2f}",
                'Min Value': f"{data[col].min():,.2f}",
                'Max Value': f"{data[col].max():,.2f}"
            })
    
    return pd.DataFrame(summary_rows)

def convert_date_columns(data, date_cols):
    """Convert date columns to datetime format"""
    print("\n📅 Converting date columns...")
    for col in date_cols:
        if col in data.columns:
            try:
                data[col] = pd.to_datetime(data[col], errors='coerce')
                print(f"  ✓ Converted '{col}' to datetime")
            except Exception as e:
                print(f"  ⚠ Could not convert '{col}' to datetime: {e}")
    return data

def calculate_loan_duration(data):
    """Calculate loan duration in days"""
    if 'loan_start' in data.columns and 'loan_end' in data.columns:
        print("\n⏱ Calculating loan duration...")
        data['loan_duration_days'] = (data['loan_end'] - data['loan_start']).dt.days
        print(f"  ✓ Added 'loan_duration_days' column")
        if data['loan_duration_days'].notna().any():
            print(f"    Average duration: {data['loan_duration_days'].mean():.0f} days")
    return data

# ============ FEATURE TRANSFORMATION FUNCTIONS ============

def check_distribution_statistics(data, numeric_cols):
    """Check distribution statistics for numeric columns"""
    print("\n" + "="*60)
    print("📊 DISTRIBUTION STATISTICS CHECK")
    print("="*60)
    
    distribution_info = {}
    
    for col in numeric_cols:
        if col not in data.columns:
            continue
            
        col_data = data[col].dropna()
        if len(col_data) == 0:
            continue
        
        skewness = col_data.skew()
        kurtosis = col_data.kurtosis()
        
        distribution_info[col] = {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'min': col_data.min(),
            'max': col_data.max(),
            'mean': col_data.mean(),
            'median': col_data.median()
        }
        
        print(f"\n📍 Column: {col}")
        print(f"   Skewness: {skewness:.3f}")
        print(f"   Kurtosis: {kurtosis:.3f}")
        print(f"   Mean: {col_data.mean():,.2f}")
        print(f"   Median: {col_data.median():,.2f}")
        
        if skewness > 1:
            print(f"   ⚠ Highly right skewed - transformation recommended!")
        elif skewness > 0.5:
            print(f"   ⚠ Moderately right skewed - transformation optional")
        elif skewness < -1:
            print(f"   ⚠ Highly left skewed - transformation recommended!")
        elif skewness < -0.5:
            print(f"   ⚠ Moderately left skewed - transformation optional")
        else:
            print(f"   ✅ Approximately normal - no transformation needed")
    
    return distribution_info

def apply_log_transformation(data, col):
    """Apply log transformation"""
    if col not in data.columns:
        return data
    
    min_val = data[col].min()
    if pd.isna(min_val):
        return data
        
    if min_val <= 0:
        constant = abs(min_val) + 1
        data[f'{col}_log'] = np.log(data[col] + constant)
    else:
        data[f'{col}_log'] = np.log(data[col])
    
    print(f"  ✓ Applied log transformation to '{col}' -> '{col}_log'")
    return data

def apply_sqrt_transformation(data, col):
    """Apply square root transformation"""
    if col not in data.columns:
        return data
    
    min_val = data[col].min()
    if pd.isna(min_val):
        return data
        
    if min_val < 0:
        constant = abs(min_val) + 1
        data[f'{col}_sqrt'] = np.sqrt(data[col] + constant)
    else:
        data[f'{col}_sqrt'] = np.sqrt(data[col])
    
    print(f"  ✓ Applied square root transformation to '{col}' -> '{col}_sqrt'")
    return data

def suggest_transformations(distribution_info):
    """Suggest transformations based on skewness"""
    print("\n" + "="*60)
    print("💡 SUGGESTED TRANSFORMATIONS")
    print("="*60)
    
    suggestions = {}
    
    for col, info in distribution_info.items():
        skewness = info['skewness']
        
        if abs(skewness) > 1:
            if skewness > 0:
                print(f"\n📍 {col}: Highly right skewed (skewness = {skewness:.3f})")
                print("   Suggestions: Log or Square Root transformation")
                suggestions[col] = ['log', 'sqrt']
            else:
                print(f"\n📍 {col}: Highly left skewed (skewness = {skewness:.3f})")
                print("   Suggestions: Square transformation")
                suggestions[col] = ['square']
        elif abs(skewness) > 0.5:
            print(f"\n📍 {col}: Moderately skewed (skewness = {skewness:.3f})")
            print("   Suggestions: Log transformation (optional)")
            suggestions[col] = ['log']
        else:
            print(f"\n📍 {col}: Approximately normal - no transformation needed")
            suggestions[col] = []
    
    return suggestions

def auto_transform_features(data, numeric_cols):
    """Automatically apply transformations to skewed features"""
    print("\n" + "="*60)
    print("🔄 AUTO TRANSFORMATION")
    print("="*60)
    
    dist_info = check_distribution_statistics(data, numeric_cols)
    suggestions = suggest_transformations(dist_info)
    
    transformed_cols = []
    
    for col, suggested_trans in suggestions.items():
        if not suggested_trans:
            continue
            
        print(f"\n📊 Processing {col}")
        
        if 'log' in suggested_trans:
            data = apply_log_transformation(data, col)
            transformed_cols.append(f'{col}_log')
            
        if 'sqrt' in suggested_trans:
            data = apply_sqrt_transformation(data, col)
            transformed_cols.append(f'{col}_sqrt')
    
    return data, transformed_cols, dist_info

def standardize_features(data, cols, method='zscore'):
    """Standardize features"""
    print(f"\n📊 Standardizing features...")
    
    for col in cols:
        if col not in data.columns:
            continue
            
        valid_idx = data[col].dropna().index
        if len(valid_idx) == 0:
            continue
            
        if method == 'zscore':
            mean_val = data.loc[valid_idx, col].mean()
            std_val = data.loc[valid_idx, col].std()
            if std_val > 0:
                data[f'{col}_standardized'] = (data[col] - mean_val) / std_val
                print(f"  ✓ Standardized '{col}' (Z-Score)")
        elif method == 'minmax':
            min_val = data.loc[valid_idx, col].min()
            max_val = data.loc[valid_idx, col].max()
            if max_val > min_val:
                data[f'{col}_normalized'] = (data[col] - min_val) / (max_val - min_val)
                print(f"  ✓ Normalized '{col}' (Min-Max)")
    
    return data

def prepare_for_ml(data, target_col='repaid'):
    """Prepare data for ML models"""
    print("\n" + "="*60)
    print("🤖 PREPARING DATA FOR MACHINE LEARNING")
    print("="*60)
    
    exclude_cols = ['client_id', 'loan_id', target_col]
    numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                    if col not in exclude_cols]
    
    print(f"\n📊 Numeric columns for transformation: {numeric_cols}")
    
    data, transformed_cols, dist_info = auto_transform_features(data, numeric_cols)
    
    if transformed_cols:
        print("\n" + "="*60)
        print("📏 STANDARDIZATION")
        print("="*60)
        
        standardize_choice = input("\nDo you want to standardize features? (y/n): ").strip().lower()
        if standardize_choice == 'y':
            print("\nStandardization methods:")
            print("  1. Z-Score")
            print("  2. Min-Max")
            
            method_choice = input("Choose method (1-2): ").strip()
            method = 'zscore' if method_choice == '1' else 'minmax'
            
            all_features = transformed_cols + numeric_cols
            data = standardize_features(data, all_features, method)
    
    return data, transformed_cols, dist_info

# ============ CATEGORICAL ENCODING FUNCTIONS ============

def detect_categorical_columns(data, max_unique_values=20):
    """Detect categorical columns in the dataset"""
    categorical_cols = []
    numerical_cols = []
    
    for col in data.columns:
        if col in ['repaid', 'client_id', 'loan_id']:
            continue
            
        if data[col].dtype == 'object' or data[col].dtype.name == 'category':
            categorical_cols.append(col)
        elif data[col].nunique() <= max_unique_values:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    
    return categorical_cols, numerical_cols

def show_categorical_columns_info(data, categorical_cols):
    """Display information about categorical columns"""
    print("\n" + "="*60)
    print("📊 CATEGORICAL COLUMNS INFORMATION")
    print("="*60)
    
    if not categorical_cols:
        print("No categorical columns found!")
        return
    
    for col in categorical_cols:
        unique_count = data[col].nunique()
        unique_values = data[col].dropna().unique()[:10]
        
        print(f"\n📍 Column: {col}")
        print(f"   Unique values: {unique_count}")
        print(f"   Sample values: {list(unique_values)}")
        print(f"   Missing values: {data[col].isna().sum()}")
        
        print(f"   Top 5 value counts:")
        value_counts = data[col].value_counts().head(5)
        for val, count in value_counts.items():
            percentage = (count / len(data)) * 100
            print(f"      {val}: {count} ({percentage:.1f}%)")

def apply_label_encoding(data, col):
    """Apply Label Encoding to a categorical column"""
    if col not in data.columns:
        print(f"  ❌ Column '{col}' not found!")
        return data
    
    le = LabelEncoder()
    
    if data[col].isna().any():
        print(f"  ⚠ Column '{col}' has missing values. Filling with 'Unknown'")
        data[col] = data[col].fillna('Unknown')
    
    data[f'{col}_encoded'] = le.fit_transform(data[col].astype(str))
    
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  ✓ Applied Label Encoding to '{col}' -> '{col}_encoded'")
    
    return data

def apply_onehot_encoding(data, col, drop_first=False):
    """Apply One-Hot Encoding to a categorical column"""
    if col not in data.columns:
        print(f"  ❌ Column '{col}' not found!")
        return data
    
    if data[col].isna().any():
        print(f"  ⚠ Column '{col}' has missing values. Filling with 'Unknown'")
        data[col] = data[col].fillna('Unknown')
    
    dummies = pd.get_dummies(data[col], prefix=col, drop_first=drop_first, dtype=int)
    data = pd.concat([data, dummies], axis=1)
    
    print(f"  ✓ Applied One-Hot Encoding to '{col}'")
    print(f"    Created {len(dummies.columns)} new columns")
    
    return data

def apply_frequency_encoding(data, col):
    """Apply Frequency Encoding"""
    if col not in data.columns:
        print(f"  ❌ Column '{col}' not found!")
        return data
    
    freq_map = data[col].value_counts(normalize=True).to_dict()
    data[f'{col}_freq'] = data[col].map(freq_map)
    
    print(f"  ✓ Applied Frequency Encoding to '{col}' -> '{col}_freq'")
    
    return data

def apply_target_encoding(data, col, target_col):
    """Apply Target Encoding"""
    if col not in data.columns:
        print(f"  ❌ Column '{col}' not found!")
        return data
    
    if target_col not in data.columns:
        print(f"  ❌ Target column '{target_col}' not found!")
        return data
    
    target_mean = data.groupby(col)[target_col].mean().to_dict()
    data[f'{col}_target_encoded'] = data[col].map(target_mean)
    
    print(f"  ✓ Applied Target Encoding to '{col}' -> '{col}_target_encoded'")
    
    return data

def automatic_encode_categorical(data, categorical_cols, target_col='repaid'):
    """Automatically encode categorical columns"""
    print("\n" + "="*60)
    print("🔄 AUTO ENCODING CATEGORICAL VARIABLES")
    print("="*60)
    
    encoded_data = data.copy()
    encoding_summary = {}
    
    for col in categorical_cols:
        unique_count = data[col].nunique()
        
        print(f"\n📍 Encoding '{col}' (unique values: {unique_count})")
        
        if unique_count == 2:
            encoded_data = apply_label_encoding(encoded_data, col)
            encoding_summary[col] = 'Label Encoding'
        elif unique_count <= 5:
            encoded_data = apply_onehot_encoding(encoded_data, col, drop_first=False)
            encoding_summary[col] = 'One-Hot Encoding'
        elif unique_count <= 10:
            encoded_data = apply_target_encoding(encoded_data, col, target_col)
            encoding_summary[col] = 'Target Encoding'
        else:
            encoded_data = apply_frequency_encoding(encoded_data, col)
            encoding_summary[col] = 'Frequency Encoding'
    
    return encoded_data, encoding_summary

def manual_encode_categorical(data, categorical_cols, target_col='repaid'):
    """Manual encoding with user choice"""
    print("\n" + "="*60)
    print("🔤 MANUAL CATEGORICAL ENCODING")
    print("="*60)
    
    encoded_data = data.copy()
    encoding_summary = {}
    
    for col in categorical_cols:
        unique_count = data[col].nunique()
        
        print(f"\n{'='*50}")
        print(f"Column: {col}")
        print(f"Unique values: {unique_count}")
        print(f"{'='*50}")
        
        print("\nEncoding methods:")
        print("  1. Label Encoding")
        print("  2. One-Hot Encoding")
        print("  3. Frequency Encoding")
        print("  4. Target Encoding")
        print("  5. Skip this column")
        
        choice = input(f"\nChoose method for '{col}' (1-5): ").strip()
        
        if choice == '1':
            encoded_data = apply_label_encoding(encoded_data, col)
            encoding_summary[col] = 'Label Encoding'
        elif choice == '2':
            encoded_data = apply_onehot_encoding(encoded_data, col, drop_first=False)
            encoding_summary[col] = 'One-Hot Encoding'
        elif choice == '3':
            encoded_data = apply_frequency_encoding(encoded_data, col)
            encoding_summary[col] = 'Frequency Encoding'
        elif choice == '4':
            encoded_data = apply_target_encoding(encoded_data, col, target_col)
            encoding_summary[col] = 'Target Encoding'
        else:
            print(f"  ⏭ Skipping '{col}'")
            encoding_summary[col] = 'Skipped'
    
    return encoded_data, encoding_summary

def show_encoding_summary(encoding_summary):
    """Display summary of applied encodings"""
    print("\n" + "="*60)
    print("📊 ENCODING SUMMARY")
    print("="*60)
    
    for col, method in encoding_summary.items():
        print(f"  • {col}: {method}")

# اضافه کردن به انتهای فایل M_transforms.py

# ============ FEATURE ENGINEERING FUNCTIONS ============

def create_polynomial_features(data, cols, degree=2, include_interaction=True):
    """
    Create polynomial features (square, cube, etc.) and interaction terms
    """
    print("\n" + "="*60)
    print("🔷 CREATING POLYNOMIAL FEATURES")
    print("="*60)
    
    poly_data = data.copy()
    new_features = []
    
    for col in cols:
        if col not in data.columns:
            continue
            
        # Square term
        data[f'{col}_square'] = data[col] ** 2
        new_features.append(f'{col}_square')
        print(f"  ✓ Created {col}_square")
        
        # Cube term (if degree >= 3)
        if degree >= 3:
            data[f'{col}_cube'] = data[col] ** 3
            new_features.append(f'{col}_cube')
            print(f"  ✓ Created {col}_cube")
    
    # Interaction terms between features
    if include_interaction and len(cols) > 1:
        print(f"\n  Creating interaction terms:")
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                col1, col2 = cols[i], cols[j]
                if col1 in data.columns and col2 in data.columns:
                    interaction_name = f'{col1}_x_{col2}'
                    data[interaction_name] = data[col1] * data[col2]
                    new_features.append(interaction_name)
                    print(f"    ✓ {interaction_name}")
    
    return data, new_features

def create_ratio_features(data, col_pairs):
    """
    Create ratio features from pairs of columns
    col_pairs: list of tuples like [('col1', 'col2'), ...]
    """
    print("\n" + "="*60)
    print("🔷 CREATING RATIO FEATURES")
    print("="*60)
    
    new_features = []
    
    for col1, col2 in col_pairs:
        if col1 in data.columns and col2 in data.columns:
            # Avoid division by zero
            data[f'{col1}_over_{col2}'] = data[col1] / (data[col2] + 1e-6)
            new_features.append(f'{col1}_over_{col2}')
            print(f"  ✓ Created {col1}_over_{col2}")
            
            # Reciprocal ratio
            data[f'{col2}_over_{col1}'] = data[col2] / (data[col1] + 1e-6)
            new_features.append(f'{col2}_over_{col1}')
            print(f"  ✓ Created {col2}_over_{col1}")
    
    return data, new_features

def create_date_based_features(data, date_col):
    """
    Create features from date columns (year, month, day, quarter, etc.)
    """
    print("\n" + "="*60)
    print("🔷 CREATING DATE-BASED FEATURES")
    print("="*60)
    
    if date_col not in data.columns:
        print(f"  ❌ Column '{date_col}' not found!")
        return data, []
    
    new_features = []
    
    # Convert to datetime if not already
    if data[date_col].dtype != 'datetime64[ns]':
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    
    # Extract date components
    data[f'{date_col}_year'] = data[date_col].dt.year
    new_features.append(f'{date_col}_year')
    print(f"  ✓ Created {date_col}_year")
    
    data[f'{date_col}_month'] = data[date_col].dt.month
    new_features.append(f'{date_col}_month')
    print(f"  ✓ Created {date_col}_month")
    
    data[f'{date_col}_day'] = data[date_col].dt.day
    new_features.append(f'{date_col}_day')
    print(f"  ✓ Created {date_col}_day")
    
    data[f'{date_col}_quarter'] = data[date_col].dt.quarter
    new_features.append(f'{date_col}_quarter')
    print(f"  ✓ Created {date_col}_quarter")
    
    data[f'{date_col}_dayofweek'] = data[date_col].dt.dayofweek
    new_features.append(f'{date_col}_dayofweek')
    print(f"  ✓ Created {date_col}_dayofweek")
    
    data[f'{date_col}_weekend'] = (data[date_col].dt.dayofweek >= 5).astype(int)
    new_features.append(f'{date_col}_weekend')
    print(f"  ✓ Created {date_col}_weekend")
    
    # Cyclical encoding for month (sine/cosine)
    data[f'{date_col}_month_sin'] = np.sin(2 * np.pi * data[f'{date_col}_month'] / 12)
    data[f'{date_col}_month_cos'] = np.cos(2 * np.pi * data[f'{date_col}_month'] / 12)
    new_features.extend([f'{date_col}_month_sin', f'{date_col}_month_cos'])
    print(f"  ✓ Created {date_col}_month_sin and {date_col}_month_cos")
    
    return data, new_features

def create_binning_features(data, col, num_bins=5, method='quantile'):
    """
    Create binned/categorical features from continuous variables
    method: 'quantile', 'uniform', 'kmeans'
    """
    print("\n" + "="*60)
    print("🔷 CREATING BINNED FEATURES")
    print("="*60)
    
    if col not in data.columns:
        print(f"  ❌ Column '{col}' not found!")
        return data, []
    
    new_features = []
    
    if method == 'quantile':
        # Quantile-based bins (equal frequency)
        data[f'{col}_bin'] = pd.qcut(data[col], q=num_bins, labels=False, duplicates='drop')
        print(f"  ✓ Created {col}_bin (quantile-based, {num_bins} bins)")
        new_features.append(f'{col}_bin')
        
    elif method == 'uniform':
        # Uniform bins (equal width)
        data[f'{col}_bin'] = pd.cut(data[col], bins=num_bins, labels=False)
        print(f"  ✓ Created {col}_bin (uniform bins, {num_bins} bins)")
        new_features.append(f'{col}_bin')
    
    return data, new_features

def create_aggregate_features(data, group_col, agg_cols, agg_functions=['mean', 'std', 'count']):
    """
    Create aggregated features based on grouping
    """
    print("\n" + "="*60)
    print("🔷 CREATING AGGREGATE FEATURES")
    print("="*60)
    
    if group_col not in data.columns:
        print(f"  ❌ Group column '{group_col}' not found!")
        return data, []
    
    new_features = []
    
    for agg_col in agg_cols:
        if agg_col not in data.columns:
            continue
            
        for func in agg_functions:
            if func == 'mean':
                agg_value = data.groupby(group_col)[agg_col].transform('mean')
                feature_name = f'{group_col}_{agg_col}_mean'
            elif func == 'std':
                agg_value = data.groupby(group_col)[agg_col].transform('std')
                feature_name = f'{group_col}_{agg_col}_std'
            elif func == 'count':
                agg_value = data.groupby(group_col)[agg_col].transform('count')
                feature_name = f'{group_col}_{agg_col}_count'
            elif func == 'min':
                agg_value = data.groupby(group_col)[agg_col].transform('min')
                feature_name = f'{group_col}_{agg_col}_min'
            elif func == 'max':
                agg_value = data.groupby(group_col)[agg_col].transform('max')
                feature_name = f'{group_col}_{agg_col}_max'
            else:
                continue
            
            data[feature_name] = agg_value
            new_features.append(feature_name)
            print(f"  ✓ Created {feature_name}")
    
    return data, new_features

def create_interaction_with_target(data, cols, target_col):
    """
    Create interaction features between features and target (for feature engineering)
    """
    print("\n" + "="*60)
    print("🔷 CREATING TARGET INTERACTION FEATURES")
    print("="*60)
    
    if target_col not in data.columns:
        print(f"  ❌ Target column '{target_col}' not found!")
        return data, []
    
    new_features = []
    
    for col in cols:
        if col in data.columns and col != target_col:
            # Interaction with target mean
            target_mean_by_feature = data.groupby(col)[target_col].mean()
            feature_name = f'{col}_target_mean'
            data[feature_name] = data[col].map(target_mean_by_feature)
            new_features.append(feature_name)
            print(f"  ✓ Created {feature_name}")
    
    return data, new_features

def create_cumulative_features(data, group_col, sort_col, agg_col):
    """
    Create cumulative features (running totals) over time
    """
    print("\n" + "="*60)
    print("🔷 CREATING CUMULATIVE FEATURES")
    print("="*60)
    
    if group_col not in data.columns or sort_col not in data.columns or agg_col not in data.columns:
        print(f"  ❌ Required columns not found!")
        return data, []
    
    new_features = []
    
    # Sort by group and date
    data_sorted = data.sort_values([group_col, sort_col])
    
    # Cumulative sum
    data[f'{agg_col}_cumsum_{group_col}'] = data_sorted.groupby(group_col)[agg_col].cumsum()
    new_features.append(f'{agg_col}_cumsum_{group_col}')
    print(f"  ✓ Created {agg_col}_cumsum_{group_col}")
    
    # Cumulative count
    data[f'{agg_col}_cumcount_{group_col}'] = data_sorted.groupby(group_col).cumcount() + 1
    new_features.append(f'{agg_col}_cumcount_{group_col}')
    print(f"  ✓ Created {agg_col}_cumcount_{group_col}")
    
    return data, new_features

def create_statistical_features(data, cols, window_size=3):
    """
    Create rolling statistical features (moving average, etc.)
    """
    print("\n" + "="*60)
    print("🔷 CREATING ROLLING STATISTICAL FEATURES")
    print("="*60)
    
    new_features = []
    
    for col in cols:
        if col not in data.columns:
            continue
            
        # Moving average
        data[f'{col}_ma{window_size}'] = data[col].rolling(window=window_size, min_periods=1).mean()
        new_features.append(f'{col}_ma{window_size}')
        print(f"  ✓ Created {col}_ma{window_size}")
        
        # Moving standard deviation
        data[f'{col}_std{window_size}'] = data[col].rolling(window=window_size, min_periods=1).std()
        new_features.append(f'{col}_std{window_size}')
        print(f"  ✓ Created {col}_std{window_size}")
        
        # Difference (momentum)
        data[f'{col}_diff1'] = data[col].diff(1)
        new_features.append(f'{col}_diff1')
        print(f"  ✓ Created {col}_diff1")
    
    return data, new_features

def apply_all_feature_engineering(data, target_col='repaid'):
    """
    Apply all feature engineering techniques
    """
    print("\n" + "="*60)
    print("🔧 ADVANCED FEATURE ENGINEERING")
    print("="*60)
    
    feature_data = data.copy()
    all_new_features = []
    
    # Define numeric columns (excluding target and IDs)
    exclude_cols = ['client_id', 'loan_id', target_col]
    numeric_cols = [col for col in feature_data.select_dtypes(include=[np.number]).columns 
                    if col not in exclude_cols]
    
    print(f"\n📊 Numeric columns for engineering: {numeric_cols}")
    
    # 1. Polynomial Features (degree 2)
    feature_data, poly_features = create_polynomial_features(
        feature_data, 
        [col for col in numeric_cols if col in ['loan_amount', 'rate', 'loan_duration_days']],
        degree=2,
        include_interaction=True
    )
    all_new_features.extend(poly_features)
    
    # 2. Ratio Features
    if 'loan_amount' in feature_data.columns and 'loan_duration_days' in feature_data.columns:
        feature_data, ratio_features = create_ratio_features(
            feature_data,
            [('loan_amount', 'loan_duration_days'), ('loan_amount', 'rate')]
        )
        all_new_features.extend(ratio_features)
    
    # 3. Date-based Features (if available)
    for date_col in ['loan_start', 'loan_end']:
        if date_col in feature_data.columns:
            feature_data, date_features = create_date_based_features(feature_data, date_col)
            all_new_features.extend(date_features)
    
    # 4. Binned Features
    for col in ['loan_amount', 'rate', 'loan_duration_days']:
        if col in feature_data.columns:
            feature_data, bin_features = create_binning_features(feature_data, col, num_bins=5, method='quantile')
            all_new_features.extend(bin_features)
    
    # 5. Interaction with Target
    feature_data, target_interaction = create_interaction_with_target(
        feature_data,
        ['loan_type_encoded' if 'loan_type_encoded' in feature_data.columns else 'loan_type'],
        target_col
    )
    all_new_features.extend(target_interaction)
    
    print(f"\n✨ Total new features created: {len(all_new_features)}")
    print(f"📊 Total features now: {feature_data.shape[1]}")
    
    return feature_data, all_new_features

def show_feature_engineering_summary(data, original_cols, new_features):
    """
    Display summary of feature engineering
    """
    print("\n" + "="*60)
    print("📊 FEATURE ENGINEERING SUMMARY")
    print("="*60)
    
    print(f"\n📈 Feature count:")
    print(f"   Original features: {len(original_cols)}")
    print(f"   New features created: {len(new_features)}")
    print(f"   Total features: {data.shape[1]}")
    
    print(f"\n🆕 New features:")
    for i, feature in enumerate(new_features, 1):
        print(f"   {i}. {feature}")
    
    # Show correlation with target
    if 'repaid' in data.columns:
        print(f"\n📊 Correlation with target (repaid):")
        correlations = {}
        for feature in new_features:
            if feature in data.columns:
                corr = data[feature].corr(data['repaid'])
                if not pd.isna(corr):
                    correlations[feature] = corr
        
        # Sort by absolute correlation
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        for feature, corr in sorted_corrs:
            strength = "Strong" if abs(corr) > 0.3 else "Moderate" if abs(corr) > 0.1 else "Weak"
            direction = "Positive" if corr > 0 else "Negative"
            print(f"   • {feature}: {corr:.3f} ({direction}, {strength})")

# اضافه کردن به انتهای فایل M_transforms.py

# ============ TRAIN TEST SPLIT FUNCTIONS ============

def split_train_test_data(data, target_col='repaid', test_size=0.2, val_size=0.1, random_state=42, stratify=True):
    """
    Split data into train, validation, and test sets
    
    Parameters:
    - data: DataFrame
    - target_col: name of target column
    - test_size: proportion for test set (default 0.2)
    - val_size: proportion for validation set (default 0.1)
    - random_state: random seed for reproducibility
    - stratify: whether to stratify by target column
    
    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split
    
    print("\n" + "="*60)
    print("📊 SPLITTING DATA INTO TRAIN, VALIDATION, AND TEST SETS")
    print("="*60)
    
    # Separate features and target
    if target_col not in data.columns:
        print(f"❌ Target column '{target_col}' not found!")
        return None, None, None, None, None, None
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    print(f"\n📊 Dataset info:")
    print(f"   Total samples: {len(data)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target: {target_col}")
    print(f"   Class distribution: {dict(y.value_counts())}")
    
    # First split: separate test set
    stratify_param = y if stratify else None
    remaining_size = 1 - test_size
    val_relative_size = val_size / remaining_size if remaining_size > 0 else 0
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=stratify_param
    )
    
    # Second split: separate validation from training
    stratify_param_temp = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_relative_size, 
        random_state=random_state, 
        stratify=stratify_param_temp
    )
    
    # Print split results
    print(f"\n📊 Split results:")
    print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(data)*100:.1f}%)")
    print(f"   Validation set: {len(X_val)} samples ({len(X_val)/len(data)*100:.1f}%)")
    print(f"   Test set: {len(X_test)} samples ({len(X_test)/len(data)*100:.1f}%)")
    
    if stratify:
        print(f"\n📊 Stratified distribution:")
        print(f"   Training - Repaid: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        print(f"   Validation - Repaid: {y_val.sum()} ({y_val.mean()*100:.1f}%)")
        print(f"   Test - Repaid: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def split_train_test_simple(data, target_col='repaid', test_size=0.2, random_state=42, stratify=True):
    """
    Simple split into train and test sets only (no validation)
    """
    from sklearn.model_selection import train_test_split
    
    print("\n" + "="*60)
    print("📊 SPLITTING DATA INTO TRAIN AND TEST SETS")
    print("="*60)
    
    if target_col not in data.columns:
        print(f"❌ Target column '{target_col}' not found!")
        return None, None, None, None
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=stratify_param
    )
    
    print(f"\n📊 Split results:")
    print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(data)*100:.1f}%)")
    print(f"   Test set: {len(X_test)} samples ({len(X_test)/len(data)*100:.1f}%)")
    
    if stratify:
        print(f"\n📊 Stratified distribution:")
        print(f"   Training - Repaid: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        print(f"   Test - Repaid: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

def save_split_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir='data/split'):
    """
    Save split datasets to CSV files
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine features and target for each set
    train_data = X_train.copy()
    train_data['repaid'] = y_train.values
    
    test_data = X_test.copy()
    test_data['repaid'] = y_test.values
    
    # Save files
    train_data.to_csv(f"{output_dir}/train.csv", index=False)
    test_data.to_csv(f"{output_dir}/test.csv", index=False)
    
    print(f"\n✅ Saved split data to '{output_dir}/'")
    print(f"   Train: {output_dir}/train.csv ({len(train_data)} samples)")
    print(f"   Test: {output_dir}/test.csv ({len(test_data)} samples)")
    
    if X_val is not None and y_val is not None:
        val_data = X_val.copy()
        val_data['repaid'] = y_val.values
        val_data.to_csv(f"{output_dir}/validation.csv", index=False)
        print(f"   Validation: {output_dir}/validation.csv ({len(val_data)} samples)")
    
    return True

def show_split_summary(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Display summary of split datasets
    """
    print("\n" + "="*60)
    print("📊 SPLIT DATASETS SUMMARY")
    print("="*60)
    
    print(f"\n📈 Dataset sizes:")
    print(f"   Train: {X_train.shape} (features: {X_train.shape[1]})")
    if X_val is not None:
        print(f"   Validation: {X_val.shape} (features: {X_val.shape[1]})")
    print(f"   Test: {X_test.shape} (features: {X_test.shape[1]})")
    
    print(f"\n🎯 Target distribution:")
    print(f"   Train - Repaid: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    if X_val is not None:
        print(f"   Validation - Repaid: {y_val.sum()} ({y_val.mean()*100:.1f}%)")
    print(f"   Test - Repaid: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
    
    # Feature names
    print(f"\n📋 Features in dataset:")
    for i, col in enumerate(X_train.columns[:10], 1):
        print(f"   {i}. {col}")
    if len(X_train.columns) > 10:
        print(f"   ... and {len(X_train.columns) - 10} more")