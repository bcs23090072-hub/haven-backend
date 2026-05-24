import requests

url = 'http://127.0.0.1:5000/chat'

print("🤖 AI Chatbot Started! (Type 'quit' to exit)")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    # Send message to Flask Server
    try:
        response = requests.post(url, json={"message": user_input})
        if response.status_code == 200:
            bot_reply = response.json().get('response')
            print(f"Bot: {bot_reply}")
        else:
            print("Error: Server returned an error.")
    except:
        print("❌ Connection failed. Please ensure app.py is running!")