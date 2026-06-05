# M_main.py
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extractions import read_csv_file
from load import load
from M_transforms import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    print("\n" + "="*60)
    print("🏦 LOAN DATA PROCESSING SYSTEM")
    print("="*60)
    
    # Load data
    file_path = 'data/loans.csv'
    print(f"\n📂 Loading file: {file_path}")
    
    try:
        data = read_csv_file(file_path)
        if data is None or len(data) == 0:
            print(f"Error: Could not load data from {file_path}!")
            return
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    print(f"\n✅ Data loaded successfully!")
    print(f"   Total records: {len(data)}")
    print(f"   Columns: {list(data.columns)}")
    
    # Define columns
    numeric_cols = ['loan_amount', 'rate', 'repaid']
    date_cols = ['loan_start', 'loan_end']
    
    available_numeric = [col for col in numeric_cols if col in data.columns]
    available_dates = [col for col in date_cols if col in data.columns]
    
    print(f"\n📊 Processing columns:")
    print(f"   Numeric columns: {available_numeric}")
    print(f"   Date columns: {available_dates}")
    
    # ============ STEP 1: NOISY RECORDS DETECTION ============
    print("\n" + "="*60)
    print("STEP 1: NOISY RECORDS DETECTION")
    print("="*60)
    
    stats = count_noisy_records(data, available_numeric)
    
    print(f"\n✓ Completely empty records: {stats['empty_records']}")
    print(f"✓ Records with empty cells: {stats['records_with_empty']}")
    print(f"✓ Records with non-numeric data: {stats['non_numeric_count']}")
    print(f"✓ Total noisy records: {stats['total_noisy']}")
    
    if stats['total_noisy'] > 0:
        show_noisy_details(data, available_numeric, stats)
    
    data = remove_completely_empty(data)
    
    if stats['empty_records'] > 0:
        stats = count_noisy_records(data, available_numeric)
    
    # ============ STEP 2: OUTLIER DETECTION ============
    print("\n" + "="*60)
    print("STEP 2: OUTLIER DETECTION")
    print("="*60)
    
    detect_outlier_choice = input("\nDo you want to detect outliers? (y/n): ").strip().lower()
    
    outlier_result = None
    if detect_outlier_choice == 'y':
        print("\nOutlier sensitivity:")
        print("  1. Standard (multiplier=1.5) - Recommended")
        print("  2. Extreme (multiplier=3.0) - Less sensitive")
        print("  3. Custom")
        
        mult_choice = input("Choose (1-3): ").strip()
        if mult_choice == '1':
            multiplier = 1.5
        elif mult_choice == '2':
            multiplier = 3.0
        elif mult_choice == '3':
            try:
                multiplier = float(input("Enter multiplier (e.g., 2.0): ").strip())
            except:
                multiplier = 1.5
        else:
            multiplier = 1.5
        
        outlier_result = detect_outliers_iqr(data, available_numeric, multiplier)
        
        if outlier_result and outlier_result['total_outlier_rows'] > 0:
            summary_table = get_outlier_summary_table(outlier_result, data)
            if len(summary_table) > 0:
                print("\n📊 Outlier Summary Table:")
                print(summary_table.to_string(index=False))
            
            data = handle_outliers(data, available_numeric, outlier_result)
        elif outlier_result:
            print("\n✅ No outliers detected in the dataset!")
    else:
        print("\n✅ Skipping outlier detection")
    
    # ============ STEP 3: DATA CLEANING ============
    has_noise = stats['total_noisy'] > 0
    has_outliers = outlier_result and outlier_result['total_outlier_rows'] > 0
    
    if has_noise or has_outliers:
        print("\n" + "="*60)
        print("STEP 3: DATA CLEANING")
        print("="*60)
        
        data_cleaned = data.copy()
        
        if available_dates:
            data_cleaned = convert_date_columns(data_cleaned, available_dates)
        
        if 'loan_start' in data_cleaned.columns and 'loan_end' in data_cleaned.columns:
            data_cleaned = calculate_loan_duration(data_cleaned)
        
        if available_numeric:
            data_cleaned = convert_to_numeric(data_cleaned, available_numeric)
        
        if available_numeric:
            data_cleaned = handle_empty_cells(data_cleaned, available_numeric)
        
        if len(data_cleaned) > 0 and available_numeric:
            data_cleaned = convert_to_integers(data_cleaned, available_numeric)
        
        os.makedirs('data', exist_ok=True)
        output_file = "./data/target1.csv"
        load(data_cleaned, output_file)
        
        print(f"\n✅ Saved cleaned data to: {output_file}")
        
        final_data = data_cleaned
        
    else:
        print("\n✅ Data is noise-free!")
        
        if available_dates:
            data = convert_date_columns(data, available_dates)
        
        if 'loan_start' in data.columns and 'loan_end' in data.columns:
            data = calculate_loan_duration(data)
        
        if available_numeric:
            data = convert_to_integers(data, available_numeric)
        
        os.makedirs('data', exist_ok=True)
        output_file = "./data/target1.csv"
        load(data, output_file)
        print(f"\n✅ Saved {len(data)} records to {output_file}")
        
        final_data = data
    
    # ============ STEP 4: FEATURE TRANSFORMATION FOR ML ============
    print("\n" + "="*60)
    print("STEP 4: FEATURE TRANSFORMATION FOR MACHINE LEARNING")
    print("="*60)
    
    transform_choice = input("\nDo you want to apply feature transformations for ML? (y/n): ").strip().lower()
    
    transformed_data = final_data.copy()
    transformed_cols = []
    dist_info = {}
    
    if transform_choice == 'y':
        transformed_data, transformed_cols, dist_info = prepare_for_ml(transformed_data, target_col='repaid')
        
        transformed_file = "./data/target1_transformed.csv"
        load(transformed_data, transformed_file)
        print(f"\n✅ Saved transformed data to: {transformed_file}")
        
        print("\n" + "="*60)
        print("📊 TRANSFORMATION SUMMARY")
        print("="*60)
        
        print(f"\n✨ New columns created:")
        for col in transformed_cols:
            if col in transformed_data.columns:
                print(f"   • {col}")
        
        for orig_col in dist_info.keys():
            for trans_type in ['_log', '_sqrt']:
                trans_col = f"{orig_col}{trans_type}"
                if trans_col in transformed_data.columns:
                    print(f"\n📈 {orig_col} → {trans_col}:")
                    print(f"   Original skewness: {dist_info[orig_col]['skewness']:.3f}")
                    new_skewness = transformed_data[trans_col].dropna().skew()
                    print(f"   Transformed skewness: {new_skewness:.3f}")
                    improvement = abs(dist_info[orig_col]['skewness']) - abs(new_skewness)
                    print(f"   Improvement: {improvement:.3f} ({'✓' if improvement > 0 else '✗'})")
        
        if 'loan_duration_days' in transformed_data.columns:
            print(f"\n📅 Loan Duration Statistics:")
            print(f"   Average: {transformed_data['loan_duration_days'].mean():.0f} days")
            print(f"   Median: {transformed_data['loan_duration_days'].median():.0f} days")
            print(f"   Min: {transformed_data['loan_duration_days'].min():.0f} days")
            print(f"   Max: {transformed_data['loan_duration_days'].max():.0f} days")
    
    else:
        print("\n✅ Skipping feature transformation")
    
    # ============ STEP 5: CATEGORICAL ENCODING ============
    print("\n" + "="*60)
    print("STEP 5: CATEGORICAL VARIABLE ENCODING")
    print("="*60)
    
    encode_choice = input("\nDo you want to encode categorical variables? (y/n): ").strip().lower()
    
    encoded_data_for_ml = None
    categorical_cols = []
    
    if encode_choice == 'y':
        data_to_encode = transformed_data if transform_choice == 'y' else final_data
        
        categorical_cols, _ = detect_categorical_columns(data_to_encode)
        
        if categorical_cols:
            show_categorical_columns_info(data_to_encode, categorical_cols)
            
            print("\nEncoding options:")
            print("  1. Automatic encoding (recommended)")
            print("  2. Manual encoding (choose for each column)")
            
            encoding_mode = input("\nChoose mode (1-2): ").strip()
            
            if encoding_mode == '2':
                encoded_data, encoding_summary = manual_encode_categorical(
                    data_to_encode, categorical_cols, target_col='repaid'
                )
            else:
                encoded_data, encoding_summary = automatic_encode_categorical(
                    data_to_encode, categorical_cols, target_col='repaid'
                )
            
            show_encoding_summary(encoding_summary)
            
            encoded_file = "./data/target1_encoded.csv"
            load(encoded_data, encoded_file)
            print(f"\n✅ Saved encoded data to: {encoded_file}")
            
            encoded_data_for_ml = encoded_data
            
        else:
            print("\n✅ No categorical columns found!")
            encoded_data_for_ml = data_to_encode
    
    else:
        print("\n✅ Skipping categorical encoding")
        encoded_data_for_ml = transformed_data if transform_choice == 'y' else final_data
    
    # ============ STEP 6: SMART FEATURE ENGINEERING ============
    print("\n" + "="*60)
    print("STEP 6: SMART FEATURE ENGINEERING")
    print("="*60)
    
    smart_fe_choice = input("\nDo you want to create smart features? (y/n): ").strip().lower()
    
    smart_data = None
    new_smart_features = []
    
    if smart_fe_choice == 'y':
        data_for_smart = encoded_data_for_ml.copy()
        
        print("\n" + "="*60)
        print("🔧 CREATING SMART FEATURES")
        print("="*60)
        
        # 1. Loan amount per day (repayment rate indicator)
        if 'loan_amount' in data_for_smart.columns and 'loan_duration_days' in data_for_smart.columns:
            data_for_smart['amount_per_day'] = data_for_smart['loan_amount'] / (data_for_smart['loan_duration_days'] + 1)
            new_smart_features.append('amount_per_day')
            print("  ✓ Created amount_per_day (loan amount per day)")
        
        # 2. Interest burden (loan_amount * rate)
        if 'loan_amount' in data_for_smart.columns and 'rate' in data_for_smart.columns:
            data_for_smart['interest_burden'] = data_for_smart['loan_amount'] * data_for_smart['rate'] / 100
            new_smart_features.append('interest_burden')
            print("  ✓ Created interest_burden (loan_amount × rate)")
        
        # 3. Square of loan amount (for non-linear relationships)
        if 'loan_amount' in data_for_smart.columns:
            data_for_smart['loan_amount_sq'] = data_for_smart['loan_amount'] ** 2
            new_smart_features.append('loan_amount_sq')
            print("  ✓ Created loan_amount_sq (quadratic term)")
        
        # 4. Rate category (low/medium/high)
        if 'rate' in data_for_smart.columns:
            data_for_smart['rate_category'] = pd.cut(data_for_smart['rate'], 
                                                      bins=[-1, 2, 5, 100], 
                                                      labels=['low', 'medium', 'high'])
            new_smart_features.append('rate_category')
            print("  ✓ Created rate_category (low/medium/high)")
        
        # 5. Interaction: rate × duration
        if 'rate' in data_for_smart.columns and 'loan_duration_days' in data_for_smart.columns:
            data_for_smart['rate_duration_interaction'] = data_for_smart['rate'] * data_for_smart['loan_duration_days']
            new_smart_features.append('rate_duration_interaction')
            print("  ✓ Created rate_duration_interaction")
        
        # 6. Log of loan amount (if not already exists)
        if 'loan_amount' in data_for_smart.columns and 'loan_amount_log' not in data_for_smart.columns:
            data_for_smart['loan_amount_log'] = np.log(data_for_smart['loan_amount'] + 1)
            new_smart_features.append('loan_amount_log')
            print("  ✓ Created loan_amount_log")
        
        # 7. One-hot encode rate_category
        if 'rate_category' in data_for_smart.columns:
            dummies = pd.get_dummies(data_for_smart['rate_category'], prefix='rate_cat', dtype=int)
            data_for_smart = pd.concat([data_for_smart, dummies], axis=1)
            new_smart_features.extend(dummies.columns.tolist())
            print(f"  ✓ One-hot encoded rate_category -> {len(dummies.columns)} features")
        
        print(f"\n✨ Created {len(new_smart_features)} smart features!")
        
        # Save smart featured data
        smart_file = "./data/target1_smart.csv"
        load(data_for_smart, smart_file)
        print(f"\n✅ Saved smart featured data to: {smart_file}")
        
        smart_data = data_for_smart
        
    else:
        print("\n✅ Skipping smart feature engineering")
        smart_data = encoded_data_for_ml
    
    # ============ STEP 7: TRAIN TEST SPLIT ============
    print("\n" + "="*60)
    print("STEP 7: TRAIN TEST SPLIT")
    print("="*60)
    
    split_choice = input("\nDo you want to split data into train/test sets? (y/n): ").strip().lower()
    
    X_train, X_test, y_train, y_test = None, None, None, None
    
    if split_choice == 'y':
        # Choose which dataset to split
        print("\nWhich dataset do you want to split?")
        print("  1. Smart featured data (target1_smart.csv) - Recommended")
        print("  2. Encoded data (target1_encoded.csv)")
        print("  3. Transformed data (target1_transformed.csv)")
        print("  4. Cleaned data (target1.csv)")
        
        dataset_choice = input("Choose dataset (1-4): ").strip()
        
        if dataset_choice == '1' and smart_data is not None:
            data_to_split = smart_data
            data_name = "Smart Featured Data"
        elif dataset_choice == '2' and encoded_data_for_ml is not None:
            data_to_split = encoded_data_for_ml
            data_name = "Encoded Data"
        elif dataset_choice == '3' and transform_choice == 'y':
            data_to_split = transformed_data
            data_name = "Transformed Data"
        elif dataset_choice == '4':
            data_to_split = final_data
            data_name = "Cleaned Data"
        else:
            print("⚠ Selected dataset not available! Using smart data if available.")
            if smart_data is not None:
                data_to_split = smart_data
                data_name = "Smart Featured Data"
            else:
                data_to_split = final_data
                data_name = "Cleaned Data"
        
        print(f"\n📊 Splitting: {data_name}")
        print(f"   Shape: {data_to_split.shape}")
        
        # Split parameters
        test_size = float(input("\nTest set size (0.1-0.3, default 0.2): ").strip() or "0.2")
        random_state = int(input("Random seed (default 42): ").strip() or "42")
        
        # Separate features and target
        X = data_to_split.drop(columns=['repaid'])
        y = data_to_split['repaid']
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        # Save splits
        os.makedirs('data/split', exist_ok=True)
        
        train_df = X_train.copy()
        train_df['repaid'] = y_train
        train_df.to_csv('data/split/train.csv', index=False)
        
        test_df = X_test.copy()
        test_df['repaid'] = y_test
        test_df.to_csv('data/split/test.csv', index=False)
        
        print(f"\n✅ Data split completed!")
        print(f"   Train: {len(X_train)} samples -> data/split/train.csv")
        print(f"   Test: {len(X_test)} samples -> data/split/test.csv")
        print(f"\n📊 Target distribution:")
        print(f"   Train - Repaid: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        print(f"   Test - Repaid: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
        
    else:
        print("\n✅ Skipping train/test split")
    
    # ============ FINAL REPORT ============
    print("\n" + "="*60)
    print("🎯 FINAL REPORT")
    print("="*60)
    
    print(f"\n📁 Files saved:")
    print(f"   1. Cleaned data: ./data/target1.csv")
    if transform_choice == 'y':
        print(f"   2. Transformed data: ./data/target1_transformed.csv")
    if encode_choice == 'y' and categorical_cols:
        print(f"   3. Encoded data: ./data/target1_encoded.csv")
    if smart_fe_choice == 'y':
        print(f"   4. Smart featured data: ./data/target1_smart.csv")
    if split_choice == 'y':
        print(f"   5. Train set: ./data/split/train.csv")
        print(f"   6. Test set: ./data/split/test.csv")
    
    print(f"\n📊 Data shapes:")
    print(f"   Cleaned data: {final_data.shape}")
    if transform_choice == 'y':
        print(f"   Transformed data: {transformed_data.shape}")
    if encode_choice == 'y' and categorical_cols:
        print(f"   Encoded data: {encoded_data_for_ml.shape}")
    if smart_fe_choice == 'y':
        print(f"   Smart featured data: {smart_data.shape}")
    
    print(f"\n🎯 Target variable: repaid")
    if 'repaid' in final_data.columns:
        print(f"   Class distribution:")
        print(f"      Repaid (1): {(final_data['repaid'] == 1).sum()} ({final_data['repaid'].mean()*100:.1f}%)")
        print(f"      Not repaid (0): {(final_data['repaid'] == 0).sum()} ({(1-final_data['repaid'].mean())*100:.1f}%)")
    
    # Final insights
    print("\n📈 FINAL INSIGHTS:")
    if 'loan_duration_days' in final_data.columns:
        print(f"   • Average loan duration: {final_data['loan_duration_days'].mean():.0f} days")
    if 'loan_amount' in final_data.columns:
        print(f"   • Average loan amount: ${final_data['loan_amount'].mean():,.2f}")
    if 'rate' in final_data.columns:
        print(f"   • Average interest rate: {final_data['rate'].mean():.2f}%")
    
    # Show split summary if done
    if split_choice == 'y' and X_train is not None:
        print(f"\n📊 Train/Test Split Summary:")
        print(f"   Train shape: {X_train.shape}")
        print(f"   Test shape: {X_test.shape}")
        print(f"   Features: {list(X_train.columns[:5])}...")
    
    print("\n" + "="*60)
    print("✅ PROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return final_data, transformed_data if transform_choice == 'y' else None, encoded_data_for_ml if encode_choice == 'y' else None, smart_data if smart_fe_choice == 'y' else None, (X_train, X_test, y_train, y_test) if split_choice == 'y' else None

if __name__ == "__main__":
    result = main()