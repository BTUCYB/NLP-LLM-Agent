import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # Load dataset from device folder
    df = pd.read_csv("data/device/device_split.csv")
    
    # Stratified split based on activity_binary
    train_df = df.groupby('activity_binary', group_keys=False).apply(
        lambda x: x.sample(frac=0.8, random_state=42))
    test_df = df.drop(train_df.index)
    
    # Create device subdirectories
    Path("data/device/train").mkdir(parents=True, exist_ok=True)
    Path("data/device/test").mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_df.to_csv("data/device/train/train.csv", index=False)
    test_df.to_csv("data/device/test/test.csv", index=False)
    
    print(f"Original samples: {len(df)}")
    print(f"Training samples: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"Testing samples: {len(test_df)} ({len(test_df)/len(df):.1%})")
    print("Stratified split completed!")

if __name__ == "__main__":
    main()