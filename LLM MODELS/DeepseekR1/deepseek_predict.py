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
BATCH_SIZE = 10  # Adjust based on API limits
MAX_WORKERS = 5  # Concurrent threads
RETRY_LIMIT = 3
TIMEOUT = 25

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

def generate_activity_text(row):
    """Generate natural language description from log data"""
    try:
        time_str = parse_date(row['date']).strftime('%B %d, %Y at %I:%M %p')
        return (
            f"User {row['user']} {row['activity'].lower()} "
            f"{row['pc']} on {time_str}. "
            f"Session ID: {row['id']}"
        )
    except Exception as e:
        logger.error(f"Error generating text for row {row.name}: {str(e)}")
        return "Activity description unavailable"

def process_batch(batch_texts, progress_bar):
    """Process a batch of texts with retry logic"""
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
                                    "You are a cybersecurity analyst. Classify connection events as exactly 'threat' or 'normal'. "
                                    "Consider these threat indicators: unusual connection times, multiple concurrent connections, "
                                    "suspicious user behavior, or unexpected disconnections."
                                )
                            },
                            {
                                "role": "user",
                                "content": f"Activity: {text}\n\nClassification (threat/normal only):"
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
                predictions.append(prediction if prediction in ['threat', 'normal'] else 'uncertain')
                break
            except requests.exceptions.RequestException as e:
                if attempt == RETRY_LIMIT - 1:
                    logger.warning(f"Failed batch after {RETRY_LIMIT} attempts: {str(e)}")
                    predictions.append("api_error")
                time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                predictions.append("error")
                break
    progress_bar.update(len(batch_texts))
    return predictions

def get_predictions(texts):
    """Get threat predictions using batch processing"""
    predictions = []
    with tqdm(total=len(texts), desc="Analyzing logs") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i+BATCH_SIZE]
                futures.append(  # Fixed parenthesis
                    executor.submit(process_batch, batch, progress_bar)
                )
                time.sleep(0.5)  # Rate limiting

            for future in concurrent.futures.as_completed(futures):
                try:
                    predictions.extend(future.result())
                except Exception as e:
                    logger.error(f"Batch failed: {str(e)}")
                    predictions.extend(["error"] * BATCH_SIZE)
    return predictions

def calculate_risk(row):
    """Calculate risk level based on multiple factors"""
    if row['predicted_label'] == 'threat':
        return 'CRITICAL'
    elif row['predicted_label'] == 'uncertain':
        return 'HIGH'
    elif 'disconnect' in row['activity'].lower():
        return 'MEDIUM'
    return 'LOW'

def main():
    try:
        # Ensure directories exist
        Path("data").mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)
        
        # Load data with validation
        df = pd.read_csv("data/device.csv")
        required_columns = ['id', 'date', 'user', 'pc', 'activity']
        
        if not set(required_columns).issubset(df.columns):
            missing = set(required_columns) - set(df.columns)
            raise ValueError(f"CSV missing required columns: {missing}")
            
        logger.info(f"Loaded {len(df)} records from device.csv")
        
        # Generate activity descriptions
        df["text"] = df.apply(generate_activity_text, axis=1)
        
        # Batch processing for predictions
        logger.info("Starting threat detection analysis...")
        df["predicted_label"] = get_predictions(df["text"].tolist())
        
        # Calculate risk levels
        df["risk_level"] = df.apply(calculate_risk, axis=1)
        
        # Save results
        output_path = "results/threat_analysis_report.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Analysis complete. Results saved to {output_path}")
        
        # Generate report
        print("\nThreat Analysis Summary:")
        print(df[["text", "predicted_label", "risk_level"]].head())
        print(f"\nCritical risks: {len(df[df['risk_level'] == 'CRITICAL'])}")
        print(f"High risks: {len(df[df['risk_level'] == 'HIGH'])}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Saving partial results...")
        df.to_csv("results/partial_analysis.csv", index=False)
        logger.info("Partial results saved to results/partial_analysis.csv")
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        raise

if __name__ == "__main__":
    main()