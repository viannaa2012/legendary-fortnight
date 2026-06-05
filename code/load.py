# load.py
def load(data, file_path):
    """Save dataframe to CSV file"""
    data.to_csv(file_path, index=False)
    print(f"Data successfully saved to {file_path}")