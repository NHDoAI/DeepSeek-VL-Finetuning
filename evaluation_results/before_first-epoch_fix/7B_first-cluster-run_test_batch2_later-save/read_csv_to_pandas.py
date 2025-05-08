import pandas as pd
import os

def read_categorical_labels(file_path=None):
    """
    Read categorical labels from a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str, optional): Path to the CSV file. If None, uses the default path.
        
    Returns:
        pandas.DataFrame: DataFrame containing the categorical labels data
    """
    if file_path is None:
        # Default path to the categorical labels CSV file
        file_path = os.path.join(
            "training_code", 
            "local_training", 
            "version_evaluation", 
            "1.3b_batch2_firstepoch", 
            "categorical_labels", 
            "test-set_categorical_labels.csv"
        )
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at: {file_path}")
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Print basic information about the DataFrame
    print(f"Successfully loaded data from {file_path}")
    print(f"Shape: {df.shape}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    
    print("\nSample data (first 5 rows):")
    print(df.head())
    
    return df

if __name__ == "__main__":
    # Example usage
    df = read_categorical_labels()
    
    # You can add additional analysis here
    print("\nValue counts for 'lane':")
    print(df['lane'].value_counts())
    
    print("\nValue counts for 'obstacle':")
    print(df['obstacle'].value_counts())
    
    print("\nValue counts for 'final_decision':")
    print(df['final_decision'].value_counts())
