# split_logon.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42
REQUIRED_COLUMNS = ['id', 'date', 'user', 'pc', 'activity', 'activity_binary', 'hour', 'dayofweek']

def main():
    try:
        # Initialize directories
        Path("data/logon").mkdir(parents=True, exist_ok=True)
        
        # Load and validate data
        file_path = "data/logon/logon_split.csv"  # Update path to your logon CSV
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        df = pd.read_csv(file_path)
        
        if not set(REQUIRED_COLUMNS).issubset(df.columns):
            missing = set(REQUIRED_COLUMNS) - set(df.columns)
            raise ValueError(f"Missing columns: {missing}")
            
        logger.info(f"Loaded dataset with {len(df)} logon activities")
        
        # Split data (80% train, 20% test)
        np.random.seed(RANDOM_SEED)
        df_shuffled = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        split_index = int(0.8 * len(df_shuffled))
        train_df = df_shuffled.iloc[:split_index].copy()
        test_df = df_shuffled.iloc[split_index:].copy()
        
        # Validate and save splits
        assert len(train_df) + len(test_df) == len(df_shuffled), "Split size mismatch"
        train_df.to_csv("data/logon/train.csv", index=False)
        test_df.to_csv("data/logon/test.csv", index=False)
        logger.info(f"Split complete: {len(train_df)} train (80%) | {len(test_df)} test (20%)")

    except Exception as e:
        logger.error(f"Splitting failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()