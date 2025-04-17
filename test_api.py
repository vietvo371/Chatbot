import requests
import json

def test_chat():
    url = "http://localhost:5000/chat"
    headers = {
        "Content-Type": "application/json"
    }
    
    # Test messages
    messages = [
        "Xin chào, bạn là ai?",
        "Hãy giải thích về trí tuệ nhân tạo",
        "Cảm ơn bạn"
    ]
    
    for message in messages:
        print("\n=== Sending message:", message, "===")
        
        # Send POST request
        response = requests.post(
            url,
            headers=headers,
            json={"message": message}
        )
        
        # Print response
        print("Status Code:", response.status_code)
        print("Response:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_chat() 