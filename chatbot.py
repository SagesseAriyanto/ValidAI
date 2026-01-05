import time

def get_resp(user_input):
    # Placeholder function for chatbot response
    response = "This is a placeholder response from ValidAI."

    for word in response.split():
        yield word + " "
        time.sleep(0.05)  # Simulate streaming delay
