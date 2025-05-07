analysis_template = """Analyze this insider threat event:
- User: {user}
- Device: {pc}
- Activity: {activity}
- Timestamp: {timestamp}

Consider these risk factors:
1. Unusual time access
2. Multiple device connections
3. Privileged account usage
4. Session duration anomalies
5. Data exfiltration patterns

Provide analysis in JSON format:
{{
    "risk_score": 1-5,
    "indicators": [],
    "confidence": 0-1,
    "recommendation": ""
}}"""