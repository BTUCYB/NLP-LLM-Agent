import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import logging
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import time
import re
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
# Replace v20190423 with the latest version if needed
from tencentcloud.nlp.v20190408 import nlp_client, models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
TENCENT_SECRET_ID = os.getenv("TENCENT_SECRET_ID")
TENCENT_SECRET_KEY = os.getenv("TENCENT_SECRET_KEY")

# Tencent Cloud Client Configuration
def create_nlp_client():
    cred = credential.Credential(TENCENT_SECRET_ID, TENCENT_SECRET_KEY)
    http_profile = HttpProfile()
    http_profile.endpoint = "nlp.tencentcloudapi.com"
    client_profile = ClientProfile()
    client_profile.httpProfile = http_profile
    return nlp_client.NlpClient(cred, "ap-guangzhou", client_profile)

# Constants
BATCH_SIZE = 5
MAX_WORKERS = 3
RETRY_LIMIT = 3
TIMEOUT = 30
RANDOM_SEED = 42
REQUIRED_COLUMNS = ['id', 'date', 'user', 'pc', 'activity', 'hour', 'dayofweek', 'activity_binary']

# --- Utility Functions ---
def parse_date(date_str):
    """Parse dates with flexible format handling"""
    formats = [
        '%d/%m/%Y %H:%M:%S',
        '%d/%m/%Y %H:%M',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y %H:%M',
        '%Y-%m-%d %H:%M:%S'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    logger.warning(f"Failed to parse date: {date_str}")
    return datetime.now()

# --- Feature Extraction ---
def generate_device_text(row):
    """Generate natural language description for device activities"""
    try:
        time_str = parse_date(row['date']).strftime('%B %d, %Y at %I:%M %p')
        description = (
            f"Device activity from {row['user']} on {row['pc']} at {time_str}. "
            f"Activity: {row['activity']} (Binary: {row['activity_binary']}). "
            f"Occurred on {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][row['dayofweek']-1]} "
            f"at hour {row['hour']}."
        )
        return description
    except Exception as e:
        logger.error(f"Error generating text for row {row.name}: {str(e)}")
        return "Activity description unavailable"

# --- Threat Detection with Tencent API ---
def process_batch(batch_texts, progress_bar):
    """Process a batch of texts with Tencent NLP API"""
    predictions = []
    client = create_nlp_client()
    
    for text in batch_texts:
        for attempt in range(RETRY_LIMIT):
            try:
                req = models.TextClassificationRequest()
                req.Text = text
                req.Flag = 2  # Security classification
                
                resp = client.TextClassification(req)
                
                # Process Tencent API response
                if resp.Classes:
                    top_class = max(resp.Classes, key=lambda x: x.Confidence)
                    prediction = "threat" if "threat" in top_class.ClassName.lower() else "normal"
                else:
                    prediction = "uncertain"
                
                predictions.append(prediction)
                break
            except Exception as e:
                if attempt == RETRY_LIMIT - 1:
                    logger.warning(f"Failed after {RETRY_LIMIT} attempts: {str(e)}")
                    predictions.append("api_error")
                time.sleep(2 ** attempt)
    
    progress_bar.update(len(batch_texts))
    return predictions

def get_predictions(texts):
    """Get threat predictions using batch processing"""
    predictions = []
    with tqdm(total=len(texts), desc="Analyzing activities") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i+BATCH_SIZE]
                futures.append(executor.submit(process_batch, batch, progress_bar))
                time.sleep(1)  # Tencent API rate limiting

            for future in concurrent.futures.as_completed(futures):
                try:
                    predictions.extend(future.result())
                except Exception as e:
                    predictions.extend(["error"] * BATCH_SIZE)
    return predictions  


def get_predictions(texts):
    """Get threat predictions using batch processing"""
    predictions = []
    with tqdm(total=len(texts), desc="Analyzing activities") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i+BATCH_SIZE]
                futures.append(executor.submit(process_batch, batch, progress_bar))
                time.sleep(0.5)

            for future in concurrent.futures.as_completed(futures):
                try:
                    predictions.extend(future.result())
                except Exception as e:
                    predictions.extend(["error"] * BATCH_SIZE)
    return predictions

# --- Risk Analysis ---
def calculate_device_risk(row):
    """Calculate comprehensive risk score for device activities"""
    risk_score = 0
    
    if row['predicted_label'] == 'threat':
        risk_score += 10
    elif row['predicted_label'] == 'uncertain':
        risk_score += 5
        
    # Time-based risk
    if 20 <= row['hour'] < 6:  # Late night activities
        risk_score += 4
        
    # Frequency risk
    if row['activity_binary'] == 1:  # Suspicious connections
        risk_score += 2
        
    # Day of week risk (weekends)
    if row['dayofweek'] >= 5:  # Friday(5), Saturday(6), Sunday(7)
        risk_score += 3
        
    # Risk classification
    if risk_score >= 15:
        return 'CRITICAL'
    elif risk_score >= 10:
        return 'HIGH'
    elif risk_score >= 6:
        return 'MEDIUM'
    return 'LOW'

# --- Insight Generation ---
def generate_insights(df, output_path):
    """Generate text file with key metrics"""
    insights = [
        "Device Threat Insights:",
        f"- Total activities analyzed: {len(df)}",
        f"- Critical risks: {len(df[df['risk_level'] == 'CRITICAL'])}",
        f"- High risks: {len(df[df['risk_level'] == 'HIGH'])}",
        f"- Late night activities (8PM-6AM): {len(df[df['hour'].between(20,23) | df['hour'].between(0,5)])}",
        f"- Weekend activities: {len(df[df['dayofweek'] >= 5])}",
        f"- Average risk score: {round(df['risk_level'].map({'LOW':0, 'MEDIUM':1, 'HIGH':2, 'CRITICAL':3}).mean(), 2)}"
    ]
    
    with open(output_path, 'w') as f:
        f.write("\n".join(insights))


def main():
    try:
        # Initialize directories
        Path("data").mkdir(parents=True, exist_ok=True)
        Path("results").mkdir(parents=True, exist_ok=True)
        
        # Load and validate data
        file_path = "data/device/device_split.csv"
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        df = pd.read_csv(file_path)
        
        if not set(REQUIRED_COLUMNS).issubset(df.columns):
            missing = set(REQUIRED_COLUMNS) - set(df.columns)
            raise ValueError(f"Missing columns: {missing}")
            
        logger.info(f"Loaded dataset with {len(df)} device activities")
        
        # Split data (80% train, 20% test)
        np.random.seed(RANDOM_SEED)
        df_shuffled = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        split_index = int(0.8 * len(df_shuffled))
        train_df = df_shuffled.iloc[:split_index].copy()
        test_df = df_shuffled.iloc[split_index:].copy()
        
        # Validate and save splits
        assert len(train_df) + len(test_df) == len(df_shuffled), "Split size mismatch"
        train_df.to_csv("data/device/train.csv", index=False)
        test_df.to_csv("data/device/test.csv", index=False)
        logger.info(f"Split complete: {len(train_df)} train (80%) | {len(test_df)} test (20%)")
        
        # Process training set
        logger.info("Processing training data...")
        train_df["text"] = train_df.apply(generate_device_text, axis=1)
        train_df["predicted_label"] = get_predictions(train_df["text"].tolist())
        train_df["risk_level"] = train_df.apply(calculate_device_risk, axis=1)
        train_df.to_csv("results/device/train_result.csv", index=False)
        generate_insights(train_df, "results/device/train_result.txt")
        
        # Process test set
        logger.info("Processing test data...")
        test_df["text"] = test_df.apply(generate_device_text, axis=1)
        test_df["predicted_label"] = get_predictions(test_df["text"].tolist())
        test_df["risk_level"] = test_df.apply(calculate_device_risk, axis=1)
        test_df.to_csv("results/test_result.csv", index=False)
        generate_insights(test_df, "results/test_result.txt")
        
        # Generate combined report
        combined_df = pd.concat([train_df, test_df])
        combined_df.to_csv("results/device/devices.csv", index=False)
        logger.info("Generated all result files")
        
        # Final report
        print("\n=== Execution Summary ===")
        print(f"Training samples: {len(train_df)} activities (80%)")
        print(f"Testing samples: {len(test_df)} activities (20%)")
        print(f"Total threats detected: {len(combined_df[combined_df['risk_level'].isin(['HIGH', 'CRITICAL'])])}")
        print(f"Successful API calls: {len(combined_df[combined_df['predicted_label'].isin(['threat', 'normal'])])}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()