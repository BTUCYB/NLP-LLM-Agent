import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # Load dataset
    df = pd.read_csv("data/email_2010_part1.csv")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Split into 80% training and 20% testing
    split_index = int(0.8 * len(df))
    train_df = df.iloc[:split_index]  # First 80% for training
    test_df = df.iloc[split_index:]   # Remaining 20% for testing
    
    # Create directories if needed
    Path("data").mkdir(exist_ok=True)
    
    # Save splits
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    
    # Verify split
    print(f"Original dataset: {len(df)} emails")
    print(f"Training set (80%): {len(train_df)} emails")
    print(f"Testing set (20%): {len(test_df)} emails")
    print(f"Verification: {len(train_df)+len(test_df)} total emails")
    print("Split completed successfully!")

if __name__ == "__main__":
    main()