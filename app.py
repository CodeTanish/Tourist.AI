import random
import json
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import stripe  # For payment gateway
from flask_mail import Mail, Message  # For sending tickets via email

# Flask App and Configurations
app = Flask(__name__)
CORS(app)

# Stripe Configuration
stripe.api_key = 'your_stripe_secret_key'  # Replace with your Stripe secret key

# Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'your_email_password'  # Replace with your email password
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents and model
with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data['hidden_size']
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Track the conversation state (date and passengers)
conversation_state = {}

bot_name = "Chatbot"

# API route for chatbot interaction
@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_input = request.json['message']  # Get the user message from POST request

    user_id = request.json.get('user_id', 'default')  # Add user_id if you need unique users
    if user_id not in conversation_state:
        conversation_state[user_id] = {'date': None, 'adults': None, 'children': None}

    state = conversation_state[user_id]  # Get current conversation state

    # Tokenize and predict
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                
                # If bot is asking for a date
                if "choose the date" in response.lower():
                    return jsonify({
                        'response': response, 
                        'showDateSelection': True  # Tell frontend to show date selection options
                    })
                
                # If bot is asking for adults and children
                if "number of adult and children" in response.lower():
                    state['date'] = user_input  # Save selected date in conversation state
                    return jsonify({
                        'response': response, 
                        'showPassengerInfo': True  # Tell frontend to show passenger info form
                    })
                
                # If user already provided passenger info, finalize and trigger payment
                if state['date'] and state['adults'] is not None and state['children'] is not None:
                    response += " Ready to proceed with payment."
                
                return jsonify({'response': response})
    else:
        return jsonify({'response': "I do not understand..."})

# API route to handle date selection
@app.route('/select-date', methods=['POST'])
def select_date():
    data = request.json
    user_id = data.get('user_id', 'default')
    date = data.get('date')  # "Today" or "Tomorrow"

    # Update conversation state with the selected date
    conversation_state[user_id]['date'] = date

    return jsonify({'response': f"Thank you! You selected {date}. Please enter the number of adults and children."})

# API route to handle passenger information submission
@app.route('/submit-passenger-info', methods=['POST'])
def submit_passenger_info():
    data = request.json
    user_id = data.get('user_id', 'default')
    adults = data.get('adults')
    children = data.get('children')

    # Update conversation state with the passenger information
    conversation_state[user_id]['adults'] = adults
    conversation_state[user_id]['children'] = children

    return jsonify({'response': f"Thank you! You've selected {adults} adults and {children} children. Proceeding to payment."})

# API route for creating payment
@app.route('/create-payment', methods=['POST'])
def create_payment():
    try:
        # Get the payment amount and other details from the request
        data = request.json
        amount = data['amount']  # Amount in cents, for example 500 for $5

        # Create Stripe Payment Intent
        intent = stripe.PaymentIntent.create(
            amount=amount,
            currency='usd',
            payment_method_types=['card'],
        )

        return jsonify({
            'clientSecret': intent['client_secret']
        })
    except Exception as e:
        return jsonify(error=str(e)), 403

# API route for handling successful payment and sending ticket
@app.route('/payment-success', methods=['POST'])
def payment_success():
    try:
        data = request.json
        user_email = data['email']

        # Send ticket to the user via email
        send_ticket_via_email(user_email)

        return jsonify({'message': 'Payment successful and ticket sent!'})
    except Exception as e:
        return jsonify(error=str(e)), 500

# Helper function to send tickets via email
def send_ticket_via_email(user_email):
    try:
        msg = Message(
            'Your Ticket',
            sender='your_email@gmail.com',
            recipients=[user_email]
        )
        msg.body = 'Thank you for your purchase! Here is your ticket.'
        mail.send(msg)
        print("Ticket sent to:", user_email)
    except Exception as e:
        print("Failed to send email:", e)

# Run the Flask app
if __name__ == '__main__':
    app.run()
