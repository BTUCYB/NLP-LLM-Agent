import requests

def ask_deepseek(prompt, model="deepseek-chat"):
    api_key = "sk-89b961d2cb394eb5a0ef2173d95e91e1" 
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"Request failed: {str(e)}")
        return None

# 使用示例
answer = ask_deepseek("9.9和9.11谁更大？")
print(answer)