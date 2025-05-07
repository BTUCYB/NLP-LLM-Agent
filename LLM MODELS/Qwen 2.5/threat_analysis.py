import os
import dashscope
import pandas as pd
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
dashscope.api_key = os.getenv("QWEN_API_KEY")

# Configuration
INPUT_DATA = "data/raw_data.csv"
OUTPUT_REPORT = "results/threat_analysis_report.csv"
SUMMARY_FILE = "results/summary_report.txt"

def analyze_activities():
    """Main analysis function"""
    df = pd.read_csv(INPUT_DATA)
    results = []
    
    for _, row in df.iterrows():
        # Generate analysis prompt
        prompt = f"""Analyze this activity for insider threat indicators:
        User: {row['user']}
        Device: {row['pc']}
        Activity: {row['activity']}
        Timestamp: {row['date']}

        Consider these risk factors:
        1. Unusual time access
        2. Multiple device connections
        3. Session duration anomalies
        4. Privileged account usage
        5. Data access patterns

        Provide analysis in JSON format with: risk_score (1-5), indicators, confidence (0-1)
        """
        
        # Get Qwen analysis
        try:
            response = dashscope.TextGeneration.call(
                model='qwen2.5-omni-7b-chat',
                prompt=prompt,
                temperature=0.2,
                max_tokens=500
            )
            analysis = parse_response(response.output['text'])
        except Exception as e:
            analysis = {"error": str(e)}
        
        results.append({
            "id": row['id'],
            "user": row['user'],
            "pc": row['pc'],
            "activity": row['activity'],
            "timestamp": row['date'],
            **analysis
        })
    
    return pd.DataFrame(results)

def parse_response(response_text):
    """Parse model response to structured data"""
    try:
        json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
        return json.loads(json_str)
    except:
        return {"risk_score": None, "indicators": [], "confidence": None}

def generate_summary(report_df):
    """Generate textual summary report"""
    summary = [
        f"Insider Threat Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 50,
        f"Total Activities Analyzed: {len(report_df)}",
        f"Average Risk Score: {report_df['risk_score'].mean():.2f}",
        f"High Risk Activities (Score >3): {len(report_df[report_df['risk_score'] > 3])}",
        "\nTop Threat Indicators:",
        *[f"- {ind}" for ind in report_df['indicators'].explode().value_counts().index[:3]]
    ]
    
    return '\n'.join(summary)

if __name__ == "__main__":
    # Run analysis
    report_df = analyze_activities()
    
    # Save full results
    report_df.to_csv(OUTPUT_REPORT, index=False)
    
    # Generate summary
    summary = generate_summary(report_df)
    with open(SUMMARY_FILE, 'w') as f:
        f.write(summary)
    
    print(f"Analysis complete. Results saved to {OUTPUT_REPORT}")
    print("Summary Report:\n" + summary)