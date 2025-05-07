import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import logging
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import time
import ast
import re  # Added for text preprocessing

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
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
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
        # Parse recipients and attachments
        to_recipients = parse_email_recipients(row['to'])
        cc_recipients = parse_email_recipients(row['cc'])
        bcc_recipients = parse_email_recipients(row['bcc'])
        
        # Preprocess content
        clean_content = preprocess_text(row['content_clean'][:500])  # Truncate for API
        
        # Feature engineering
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
def get_predictions(texts):
    """Get threat predictions using batch processing"""
    predictions = []
    with tqdm(total=len(texts), desc="Analyzing emails") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i+BATCH_SIZE]
                futures.append(
                    executor.submit(process_batch, batch, progress_bar)
                )
                time.sleep(0.5)

            for future in concurrent.futures.as_completed(futures):
                try:
                    predictions.extend(future.result())
                except Exception as e:
                    logger.error(f"Batch failed: {str(e)}")
                    predictions.extend(["error"] * BATCH_SIZE)
    return predictions

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
                    logger.warning(f"Failed batch after {RETRY_LIMIT} attempts: {str(e)}")
                    predictions.append("api_error")
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                predictions.append("error")
                break
    progress_bar.update(len(batch_texts))
    return predictions

# --- Risk Analysis ---
def calculate_email_risk(row):
    """Calculate comprehensive risk score for emails"""
    risk_score = 0
    
    # Base score from model prediction
    if row['predicted_label'] == 'threat':
        risk_score += 8
    elif row['predicted_label'] == 'uncertain':
        risk_score += 4
        
    # Feature-based scoring
    recipients = parse_email_recipients(row['to'])
    bcc_recipients = parse_email_recipients(row['bcc'])
    
    # Threat indicators
    risk_score += min(len(bcc_recipients) * 2, 6)  # Max 6 points
    risk_score += min(row['attachments'] * 1.5, 4)  # Max 4 points
    if row['size'] > 50000:
        risk_score += 3
    if any('external.com' in email for email in recipients):
        risk_score += 3
        
    # Temporal analysis
    email_time = parse_date(row['date']).hour
    if 20 <= email_time < 6:  # Late night emails
        risk_score += 2
        
    # Normalization
    if risk_score >= 10:
        return 'CRITICAL'
    elif risk_score >= 7:
        return 'HIGH'
    elif risk_score >= 4:
        return 'MEDIUM'
    return 'LOW'

# --- Main Workflow ---
def main():
    try:
        Path("data").mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)
        
        # Load data with validation
        df = pd.read_csv("data/email_2010_part2.csv")
        required_columns = ['id', 'date', 'user', 'pc', 'to', 'cc', 'bcc', 'from', 'attachments', 'content_clean']
        
        if not set(required_columns).issubset(df.columns):
            missing = set(required_columns) - set(df.columns)
            raise ValueError(f"Missing columns: {missing}")
            
        logger.info(f"Processing {len(df)} emails")
        
        # Feature extraction
        df["text"] = df.apply(generate_email_text, axis=1)
        
        # Threat detection
        logger.info("Starting NLP threat analysis...")
        df["predicted_label"] = get_predictions(df["text"].tolist())
        
        # Risk assessment
        df["risk_level"] = df.apply(calculate_email_risk, axis=1)
        
        # Save results
        output_path = "results/email_threat_analysis.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Analysis saved to {output_path}")
        
        # Generate insights
        print("\nInsider Threat Insights:")
        print(f"- Critical risks: {len(df[df['risk_level'] == 'CRITICAL'])}")
        print(f"- Suspicious attachments: {len(df[df['attachments'] > 3])}")
        print(f"- After-hours communications: {len(df[df['date'].apply(lambda x: 20 <= parse_date(x).hour < 6)])}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()