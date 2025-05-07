import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import logging
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import time
import ast
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Constants
BATCH_SIZE = 8
MAX_WORKERS = 4
RETRY_LIMIT = 3
TIMEOUT = 30
RANDOM_SEED = 42

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

def preprocess_text(text):
    """Clean and normalize text content"""
    try:
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:500]
    except Exception as e:
        logger.error(f"Text preprocessing failed: {str(e)}")
        return ""

def parse_email_recipients(recipient_str):
    """Safely parse stringified lists of recipients"""
    try:
        return ast.literal_eval(recipient_str)
    except:
        return []

# --- Feature Extraction ---
def generate_email_text(row):
    """Generate natural language description with threat indicators"""
    try:
        to_recipients = parse_email_recipients(row['to'])
        bcc_recipients = parse_email_recipients(row['bcc'])
        
        clean_content = preprocess_text(row['content_clean'])
        time_str = parse_date(row['date']).strftime('%B %d, %Y at %I:%M %p')
        external_recipients = len([r for r in to_recipients if 'dtaa.com' not in r])
        
        description = (
            f"Email from {row['from']} sent on {time_str} by {row['user']} (PC: {row['pc']}). "
            f"Recipients: {len(to_recipients)} direct, {len(bcc_recipients)} hidden. "
            f"Attachments: {row['attachments']} ({row['size']} bytes). "
            f"Content features: {clean_content}. "
            f"External recipients: {external_recipients}. "
            f"Security indicators: [BCC: {len(bcc_recipients)}, Attachments: {row['attachments']}]"
        )
        
        return description
    except Exception as e:
        logger.error(f"Error generating text for row {row.name}: {str(e)}")
        return "Email description unavailable"

# --- Threat Detection ---
def process_batch(batch_texts, progress_bar):
    """Process a batch of email texts with retry logic"""
    predictions = []
    for text in batch_texts:
        for attempt in range(RETRY_LIMIT):
            try:
                response = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=HEADERS,
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Analyze emails for insider threats. Classify as 'threat' or 'normal'. "
                                    "Consider: data exfiltration patterns, unusual after-hours activity, "
                                    "sensitive content sharing, hidden recipients, suspicious attachments, "
                                    "and unauthorized external communications. Focus on behavioral anomalies."
                                )
                            },
                            {
                                "role": "user",
                                "content": f"{text}\n\nClassification (threat/normal only):"
                            }
                        ],
                        "temperature": 0.1,
                        "max_tokens": 1,
                        "stop": ["\n"]
                    },
                    timeout=TIMEOUT
                )
                response.raise_for_status()
                prediction = response.json()["choices"][0]["message"]["content"].strip().lower()
                valid_pred = prediction if prediction in ['threat', 'normal'] else 'uncertain'
                predictions.append(valid_pred)
                break
            except requests.exceptions.RequestException as e:
                if attempt == RETRY_LIMIT - 1:
                    predictions.append("api_error")
                time.sleep(2 ** attempt)
            except Exception as e:
                predictions.append("error")
                break
    progress_bar.update(len(batch_texts))
    return predictions

def get_predictions(texts):
    """Get threat predictions using batch processing"""
    predictions = []
    with tqdm(total=len(texts), desc="Analyzing emails") as progress_bar:
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
def calculate_email_risk(row):
    """Calculate comprehensive risk score for emails"""
    risk_score = 0
    
    if row['predicted_label'] == 'threat':
        risk_score += 8
    elif row['predicted_label'] == 'uncertain':
        risk_score += 4
        
    recipients = parse_email_recipients(row['to'])
    bcc_recipients = parse_email_recipients(row['bcc'])
    
    risk_score += min(len(bcc_recipients) * 2, 6)
    risk_score += min(row['attachments'] * 1.5, 4)
    if row['size'] > 50000:
        risk_score += 3
    if any('external.com' in email for email in recipients):
        risk_score += 3
        
    email_time = parse_date(row['date']).hour
    if 20 <= email_time < 6:
        risk_score += 2
        
    if risk_score >= 10:
        return 'CRITICAL'
    elif risk_score >= 7:
        return 'HIGH'
    elif risk_score >= 4:
        return 'MEDIUM'
    return 'LOW'

# --- Insight Generation ---
def generate_insights(df, output_path):
    """Generate text file with key metrics"""
    insights = [
        "Insider Threat Insights:",
        f"- Total emails analyzed: {len(df)}",
        f"- Critical risks: {len(df[df['risk_level'] == 'CRITICAL'])}",
        f"- High risks: {len(df[df['risk_level'] == 'HIGH'])}",
        f"- Suspicious attachments (>3): {len(df[df['attachments'] > 3])}",
        f"- After-hours communications: {len(df[df['date'].apply(lambda x: (parse_date(x).hour >= 20) | (parse_date(x).hour < 6))])}",
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
        file_path = "data/email_2010_part1.csv"
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        df = pd.read_csv(file_path)
        required_columns = ['id', 'date', 'user', 'pc', 'to', 'cc', 'bcc', 'from', 'attachments', 'content_clean']
        
        if not set(required_columns).issubset(df.columns):
            missing = set(required_columns) - set(df.columns)
            raise ValueError(f"Missing columns: {missing}")
            
        logger.info(f"Loaded dataset with {len(df)} emails")
        
        # --- Corrected Split Logic ---
        np.random.seed(RANDOM_SEED)
        # Shuffle and split
        df_shuffled = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        split_index = int(0.8 * len(df_shuffled))
        train_df = df_shuffled.iloc[:split_index]  # 80% training
        test_df = df_shuffled.iloc[split_index:]   # 20% testing

        # Validate split
        assert len(train_df) + len(test_df) == len(df_shuffled), "Split size mismatch"
        assert len(pd.merge(train_df, test_df, how='inner')) == 0, "Data leakage detected"
        
        # Save splits
        train_df.to_csv("data/train.csv", index=False, encoding='utf-8-sig')
        test_df.to_csv("data/test.csv", index=False, encoding='utf-8-sig')
        logger.info(f"Split complete: {len(train_df)} train (80%) | {len(test_df)} test (20%)")
        
        # Process training set
        logger.info("Processing training data...")
        train_df["text"] = train_df.apply(generate_email_text, axis=1)
        train_df["predicted_label"] = get_predictions(train_df["text"].tolist())
        train_df["risk_level"] = train_df.apply(calculate_email_risk, axis=1)
        train_df.to_csv("results/train_result.csv", index=False)
        generate_insights(train_df, "results/train_result.txt")
        
        # Process test set
        logger.info("Processing test data...")
        test_df["text"] = test_df.apply(generate_email_text, axis=1)
        test_df["predicted_label"] = get_predictions(test_df["text"].tolist())
        test_df["risk_level"] = test_df.apply(calculate_email_risk, axis=1)
        test_df.to_csv("results/test_result.csv", index=False)
        generate_insights(test_df, "results/test_result.txt")
        
        # Generate combined report
        combined_df = pd.concat([train_df, test_df])
        combined_df.to_csv("results/Steven.csv", index=False)
        logger.info("Generated all result files")
        
        # Final report
        print("\n=== Execution Summary ===")
        print(f"Training samples: {len(train_df)} emails (80%)")  # Changed from 20%
        print(f"Testing samples: {len(test_df)} emails (20%)")    # Changed from 80%
        print(f"Total threats detected: {len(combined_df[combined_df['risk_level'].isin(['HIGH', 'CRITICAL'])])}")
        print(f"Successful API calls: {len(combined_df[combined_df['predicted_label'].isin(['threat', 'normal'])])}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()