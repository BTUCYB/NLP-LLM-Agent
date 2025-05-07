import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # 1. Change input file path
    input_file = "data/device/device_split.csv"
    output_train = "data/device/device_train.csv"
    output_test = "data/device/device_test.csv"
    
    # Load dataset
    df = pd.read_csv(input_file)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Split into 80% training and 20% testing
    split_index = int(0.8 * len(df))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    # Create directories if needed
    Path("data").mkdir(exist_ok=True)
    
    # 2. Save with different filenames
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)
    
    # 3. Update verification messages
    print(f"Original dataset: {len(df)} device activities")
    print(f"Training set (80%): {len(train_df)} activities")
    print(f"Testing set (20%): {len(test_df)} activities")
    print(f"Verification: {len(train_df)+len(test_df)} total activities")
    print("Device data split completed successfully!")

if __name__ == "__main__":
    main()