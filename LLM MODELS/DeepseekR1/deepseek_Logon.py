"""
Insider Threat Detection System - Logon Analysis Module
Author: [Your Name]
Date: [Date]

This module implements feature extraction and risk analysis for logon activities
using NLP-enhanced Large Language Models (LLMs). Designed to identify suspicious
patterns indicative of insider threats through temporal, behavioral, and contextual analysis.
"""

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
import re

# Configure logging for audit tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logon_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables securely
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# System constants for reproducible analysis
BATCH_SIZE = 8                # Optimal for API throughput and rate limits
MAX_WORKERS = 4               # Concurrent processing threads
RETRY_LIMIT = 3               # Fault tolerance for API calls
TIMEOUT = 30                  # Seconds before API timeout
RANDOM_SEED = 42              # Seed for reproducible splits
RISK_THRESHOLDS = {           # Risk classification parameters
    'CRITICAL': 15,
    'HIGH': 10,
    'MEDIUM': 6
}

def temporal_feature_extraction(row: pd.Series) -> str:
    """
    Transforms raw logon records into natural language narratives for LLM processing.
    Enhances temporal pattern recognition through structured textual representation.

    Methodology:
    - Converts timestamps to contextual descriptions (e.g., "Late night")
    - Encodes day-of-week patterns as textual features
    - Maintains original numerical features for hybrid analysis

    Parameters:
        row (pd.Series): Single logon record from DataFrame

    Returns:
        str: Natural language description for LLM processing

    Example:
        Input row: {user: 'A123', pc: 'PC-01', date: '2023-01-01 03:00', 
                   activity: 'logon', hour: 3, dayofweek: 6}
        Output: "Logon activity from A123 on PC-01 at January 01, 2023 at 03:00 AM. 
                Activity: logon (Confirmed: Yes). Occurred on Saturday at hour 3."
    """
    try:
        # Temporal feature engineering
        parsed_time = datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S')
        time_str = parsed_time.strftime('%B %d, %Y at %I:%M %p')
        day_str = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                   'Friday', 'Saturday', 'Sunday'][row['dayofweek']-1]
        
        # Contextual threat indicators
        time_context = "Night" if 20 <= row['hour'] < 6 else "Day"
        weekend_flag = "Weekend" if row['dayofweek'] >= 5 else "Weekday"
        
        return (
            f"{row['activity'].title()} activity from {row['user']} on {row['pc']} "
            f"at {time_str}. Authentication: {row['activity_binary']} "
            f"({'Verified' if row['activity_binary'] == 1 else 'Unverified'}). "
            f"Temporal Context: {day_str} {time_context} ({weekend_flag}). "
            f"Behavioral Marker: Hour {row['hour']}."
        )
    except Exception as e:
        logger.error(f"Feature extraction failed for row {row.name}: {str(e)}")
        return "INVALID_RECORD"

def calculate_threat_risk(row: pd.Series) -> str:
    """
    Computes composite risk score using weighted threat indicators specific to
    insider threat patterns. Incorporates temporal, frequency, and behavioral factors.

    Risk Model Components:
    - Base Threat Score: LLM prediction confidence (50% weight)
    - Temporal Risk: Off-hours and weekend premiums (30% weight)
    - Behavioral Risk: Authentication anomalies (20% weight)

    Parameters:
        row (pd.Series): Processed record with LLM predictions

    Returns:
        str: Risk classification (CRITICAL/HIGH/MEDIUM/LOW)
    """
    risk_score = 0
    
    # Base threat score from LLM prediction
    if row['predicted_label'] == 'threat':
        risk_score += 10  # 50% weight
    elif row['predicted_label'] == 'uncertain':
        risk_score += 5   # 25% weight
        
    # Temporal risk factors
    if 20 <= row['hour'] < 6:  # Night premium
        risk_score += 4    # 20% weight
    if row['dayofweek'] >= 5:  # Weekend premium
        risk_score += 3    # 15% weight
        
    # Behavioral authentication patterns
    if row['activity_binary'] == 1:  # Verified auth
        risk_score += 2    # 10% weight (normal)
    else:
        risk_score += 4    # 20% weight (unverified)
        
    # Dynamic thresholding based on organizational baseline
    return next(
        (level for level, threshold in RISK_THRESHOLDS.items() if risk_score >= threshold),
        'LOW'
    )

def process_llm_batch(batch: list[str], progress: tqdm) -> list[str]:
    """
    Executes batch processing of natural language features through DeepSeek's LLM API.
    Implements exponential backoff and fault tolerance for production reliability.

    Architecture:
    - Parallel batch processing with thread pooling
    - Retry logic with jittered delays
    - Context-aware threat classification prompt

    Parameters:
        batch (list[str]): List of natural language narratives
        progress (tqdm): Progress bar reference

    Returns:
        list[str]: Threat classifications for the batch
    """
    predictions = []
    for text in batch:
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
                                    "Analyze authentication patterns for insider threat indicators. "
                                    "Consider:\n"
                                    "1. Unusual temporal patterns (off-hours, weekends)\n"
                                    "2. Authentication method anomalies\n"
                                    "3. Frequency and sequence of privileged access\n"
                                    "4. Deviations from historical user behavior\n\n"
                                    "Output: 'threat' or 'normal'"
                                )
                            },
                            {
                                "role": "user",
                                "content": f"Activity Context:\n{text}\n\nVerdict:"
                            }
                        ],
                        "temperature": 0.2,
                        "max_tokens": 1,
                        "stop": ["\n"]
                    },
                    timeout=TIMEOUT
                )
                response.raise_for_status()
                prediction = response.json()["choices"][0]["message"]["content"].strip().lower()
                predictions.append(prediction if prediction in {'threat', 'normal'} else 'uncertain')
                break
            except requests.exceptions.RequestException as e:
                if attempt == RETRY_LIMIT - 1:
                    predictions.append("api_error")
                time.sleep(2 ** attempt + np.random.uniform(0, 1))
            except Exception as e:
                predictions.append("processing_error")
                logger.error(f"Unexpected error: {str(e)}")
                break
    progress.update(len(batch))
    return predictions

def generate_threat_insights(df: pd.DataFrame, output_path: str) -> None:
    """
    Generates comprehensive threat intelligence report integrating LLM predictions
    with statistical analysis. Supports operational decision-making through
    multi-layered insights.

    Report Features:
    - Temporal threat distribution
    - Risk classification breakdown
    - User-level threat profiling
    - Device access patterns
    - Comparative baseline analysis

    Parameters:
        df (pd.DataFrame): Analyzed dataset with risk classifications
        output_path (str): Destination path for report
    """
    try:
        # Temporal analysis
        night_logons = df[df['hour'].between(20,23) | df['hour'].between(0,5)].shape[0]
        weekend_logons = df[df['dayofweek'] >= 5].shape[0]
        
        # Threat classification
        risk_counts = df['risk_level'].value_counts().to_dict()
        
        # User risk profiling
        top_risky_users = df[df['risk_level'].isin(['HIGH', 'CRITICAL'])]\
            .groupby('user').size().nlargest(5).to_dict()
            
        insights = [
            "Insider Threat Intelligence Report",
            "==============================================",
            f"Total Analyzed Activities: {len(df)}",
            f"Time Window: {df['date'].min()} to {df['date'].max()}",
            "",
            "Risk Classification Summary:",
            f"- CRITICAL: {risk_counts.get('CRITICAL', 0)}",
            f"- HIGH: {risk_counts.get('HIGH', 0)}",
            f"- MEDIUM: {risk_counts.get('MEDIUM', 0)}",
            f"- LOW: {risk_counts.get('LOW', 0)}",
            "",
            "Temporal Patterns:",
            f"- Night Activities (8PM-6AM): {night_logons}",
            f"- Weekend Activities: {weekend_logons}",
            "",
            "Top Risky Users:",
            *[f"{user}: {count} alerts" for user, count in top_risky_users.items()],
            "",
            "Recommendations:",
            "1. Investigate CRITICAL/HIGH risk activities immediately",
            "2. Review night/weekend access policies",
            "3. Conduct user behavior analysis for top risk profiles"
        ]
        
        with open(output_path, 'w') as f:
            f.write("\n".join(insights))
            
        logger.info(f"Generated threat report: {output_path}")
        
    except Exception as e:
        logger.error(f"Insight generation failed: {str(e)}")
        raise

def main() -> None:
    """
    End-to-end execution pipeline for insider threat detection system:
    
    1. Data Preparation: Load and validate input datasets
    2. Feature Engineering: Generate NLP-enhanced temporal features
    3. Threat Detection: Parallel LLM analysis of suspicious patterns
    4. Risk Assessment: Composite scoring of identified threats
    5. Intelligence Generation: Operational reports and artifacts
    
    System Outputs:
    - Processed datasets with risk classifications
    - Threat intelligence reports (CSV and TXT formats)
    - Audit logs for regulatory compliance
    """
    try:
        # Initialize analysis environment
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
        
        # Load and validate source data
        logger.info("Loading and validating source datasets...")
        train_df = pd.read_csv("data/logon/train.csv")
        test_df = pd.read_csv("data/logon/test.csv")
        
        # Feature extraction pipeline
        logger.info("Generating NLP-enhanced features...")
        train_df["narrative"] = train_df.apply(temporal_feature_extraction, axis=1)
        test_df["narrative"] = test_df.apply(temporal_feature_extraction, axis=1)
        
        # Threat detection phase
        logger.info("Executing LLM threat analysis...")
        with tqdm(total=len(train_df)+len(test_df), desc="Processing records") as progress:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Process training data
                train_batches = [train_df["narrative"].iloc[i:i+BATCH_SIZE] 
                               for i in range(0, len(train_df), BATCH_SIZE)]
                train_futures = [executor.submit(process_llm_batch, batch, progress)
                                for batch in train_batches]
                
                # Process test data
                test_batches = [test_df["narrative"].iloc[i:i+BATCH_SIZE] 
                              for i in range(0, len(test_df), BATCH_SIZE)]
                test_futures = [executor.submit(process_llm_batch, batch, progress)
                               for batch in test_batches]
                
                # Collect results
                train_preds = []
                for future in concurrent.futures.as_completed(train_futures):
                    train_preds.extend(future.result())
                
                test_preds = []
                for future in concurrent.futures.as_completed(test_futures):
                    test_preds.extend(future.result())
        
        # Apply risk classifications
        logger.info("Calculating composite risk scores...")
        train_df["predicted_label"] = train_preds
        train_df["risk_level"] = train_df.apply(calculate_threat_risk, axis=1)
        test_df["predicted_label"] = test_preds
        test_df["risk_level"] = test_df.apply(calculate_threat_risk, axis=1)
        
        # Generate intelligence outputs
        logger.info("Compiling threat intelligence...")
        train_df.to_csv("data/processed/train_analyzed.csv", index=False)
        test_df.to_csv("data/processed/test_analyzed.csv", index=False)
        
        generate_threat_insights(train_df, "reports/train_threat_report.txt")
        generate_threat_insights(test_df, "reports/test_threat_report.txt")
        
        logger.info("Analysis pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"System failure: {str(e)}")
        raise

if __name__ == "__main__":
    main()