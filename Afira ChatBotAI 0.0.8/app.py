from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import json
import random
from datetime import datetime
from collections import Counter
import uuid
import sys
import os

api_folder = r"D:\\AfiraChatBotAI\\API_OpenWeather"
sys.path.append(api_folder)
from extractcity import extract_city
from getweather import get_weather

heart_dis_pred_folder = r"D:\\AfiraChatBotAI\\Afira ChatBotAI 0.0.8\\Predictions\\Heart_Disease_Prediction"
sys.path.append(heart_dis_pred_folder)
from heart_predictor import HeartDiseasePredictor

asthma_dis_pred_folder = r"D:\AfiraChatBotAI\\Afira ChatBotAI 0.0.8\\Predictions\\Asthma_Prediction"
sys.path.append(asthma_dis_pred_folder)
from asthma_predictor import AsthmaPredictor

app = Flask(__name__)
CORS(app)

model = None
label_encoder = None
vocab = None
word2idx = None
idf = None
intents_data = None

heart_pred = HeartDiseasePredictor(model_dir=heart_dis_pred_folder)
asthma_pred = AsthmaPredictor(model_dir=asthma_dis_pred_folder)

user_sessions = {}

def tokenize(text):
    return text.lower().split()


def compute_tf(document):
    tokens = tokenize(document)
    tf_counter = Counter(tokens)
    total_count = len(tokens)
    tf_vector = np.zeros(len(vocab))
    
    for token, count in tf_counter.items():
        if token in word2idx:
            tf_vector[word2idx[token]] = count / total_count
    return tf_vector


def text_to_tfidf(text):
    tf_vector = compute_tf(text)
    tfidf_vector = tf_vector * idf
    return tfidf_vector.reshape(1, -1)


def load_models():
    global model, label_encoder, vocab, word2idx, idf, intents_data
    
    try:
        print("Loading models from notebook...")
        
        model = pickle.load(open('nlp_model_lr.pkl', 'rb'))
        print("Model loaded")
        
        label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        print("Label encoder loaded")
        
        vocab = pickle.load(open('vocab.pkl', 'rb'))
        print(f"Vocabulary loaded ({len(vocab)} words)")
        
        word2idx = pickle.load(open('word2idx.pkl', 'rb'))
        print("Word2idx loaded")
        
        idf = pickle.load(open('idf.pkl', 'rb'))
        print("IDF loaded")
        
        with open('chatbotdata.json', 'r', encoding='utf-8') as f:
            intents_data = json.load(f)
        print(f"Intents data loaded ({len(intents_data['intents'])} intents)")
        
        print("\nAll models loaded successfully!")
        return True
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Make sure all .pkl files are in the same directory as app.py")
        return False
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def handle_prediction_intent(user_message, user_id):
    if heart_pred.check_keywords(user_message):
        result = heart_pred.start_conversation(user_id)
        if result.get('session_data'):
            user_sessions[user_id] = result['session_data']
        return result
    
    if asthma_pred.check_keywords(user_message):
        result = asthma_pred.start_conversation(user_id)
        if result.get('session_data'):
            user_sessions[user_id] = result['session_data']
        return result
    
    user_sessions[user_id] = {
        'context': 'awaiting_prediction_type',
        'collecting_data': False
    }
    
    for intent in intents_data['intents']:
        if intent['name'] == 'predictions':
            response_text = random.choice(intent['responses'])
            break
    
    return {
        'intent': 'predictions',
        'response': response_text,
        'user_id': user_id
    }


def handle_ongoing_conversation(user_message, user_id):
    
    session = user_sessions[user_id]
    context = session.get('context')
    
    if context == 'awaiting_prediction_type':
        if heart_pred.check_keywords(user_message):
            del user_sessions[user_id]
            result = heart_pred.start_conversation(user_id)
            if result.get('session_data'):
                user_sessions[user_id] = result['session_data']
            return result
        elif asthma_pred.check_keywords(user_message):
            del user_sessions[user_id]
            result = asthma_pred.start_conversation(user_id)
            if result.get('session_data'):
                user_sessions[user_id] = result['session_data']
            return result
        else:
            del user_sessions[user_id]
            return {
                'intent': 'predictions',
                'response': "I currently support heart disease and asthma risk prediction. Which one would you like to try?",
                'user_id': user_id
            }
    
    if context == 'heart_disease_prediction' and session.get('collecting_data'):
        result = heart_pred.handle_conversation_step(user_message, session, user_id)
        if result.get('session_data'):
            user_sessions[user_id] = result['session_data']
        else:
            if user_id in user_sessions:
                del user_sessions[user_id]
        return result
    
    if context == 'asthma_prediction' and session.get('collecting_data'):
        result = asthma_pred.handle_conversation_step(user_message, session, user_id)
        if result.get('session_data'):
            user_sessions[user_id] = result['session_data']
        else:
            if user_id in user_sessions:
                del user_sessions[user_id]
        return result
    
    return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        user_id = data.get('user_id', str(uuid.uuid4()))
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        if user_id in user_sessions:
            result = handle_ongoing_conversation(user_message, user_id)
            if result:
                return jsonify(result)
        
        tfidf_vector = text_to_tfidf(user_message)
        prediction = model.predict(tfidf_vector)[0]
        intent_name = label_encoder.inverse_transform([prediction])[0]
        probabilities = model.predict_proba(tfidf_vector)[0]
        confidence = float(np.max(probabilities))
        
        print(f"User: '{user_message}' -> Intent: {intent_name} ({confidence * 100:.1f}%)")
        
        ### intent "predictions" ###
        if intent_name == "predictions":
            return jsonify(handle_prediction_intent(user_message, user_id))
        
        ### intent"ask_time" ###
        if intent_name == "ask_time":
            now = datetime.now().strftime("%H:%M:%S")
            for intent in intents_data['intents']:
                if intent['name'] == "ask_time":
                    template = random.choice(intent['responses'])
                    break
            
            response_text = template.replace("{time}", now)
            
            return jsonify({
                'intent': 'ask_time',
                'confidence': confidence,
                'response': response_text,
                'user_id': user_id
            })
        
        ### intent "ask_weather" ###
        if intent_name == "ask_weather":
            city = extract_city(user_message)
            
            for intent in intents_data['intents']:
                if intent['name'] == "ask_weather":
                    intro_msg = random.choice(intent['responses'])
                    break
            
            if city:
                weather = get_weather(city)
                
                if weather:
                    response_text = (
                        f"{intro_msg}\n\n"
                        f"📍 Weather in **{weather['city']}**:\n"
                        f"🌡️ Temperature: {weather['temp']}°C (feels like {weather['feels']}°C)\n"
                        f"🌤️ Condition: {weather['desc']}\n"
                        f"💧 Humidity: {weather['humidity']}%\n"
                        f"💨 Wind: {weather['wind']} m/s"
                    )
                else:
                    response_text = (
                        f"{intro_msg}\n\n"
                        f"Sorry, I couldn't find weather info for '{city}'."
                    )
            else:
                response_text = (
                    f"{intro_msg}\n\n"
                    "Tell me a city! For example:\n"
                    "→ *weather in London*\n"
                    "→ *forecast for Paris*\n"
                    "→ *is it raining in Rome?*"
                )
            
            return jsonify({
                "intent": intent_name,
                "confidence": confidence,
                "response": response_text,
                "user_id": user_id
            })
        
        response_text = "I'm not sure how to respond to that."
        for intent in intents_data['intents']:
            if intent['name'] == intent_name:
                response_text = random.choice(intent['responses'])
                break
        
        return jsonify({
            'intent': intent_name,
            'confidence': confidence,
            'response': response_text,
            'user_id': user_id
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'vocab_size': len(vocab) if vocab is not None else 0,
        'heart_model_loaded': heart_pred.is_model_loaded(),
        'asthma_model_loaded': asthma_pred.is_model_loaded()
    })


@app.route('/reset_session', methods=['POST'])
def reset_session():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if user_id in user_sessions:
            del user_sessions[user_id]
            return jsonify({'status': 'success', 'message': 'Session reset successfully'})
        
        return jsonify({'status': 'success', 'message': 'No active session found'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Afira AI Flask Server...\n")
    
    if load_models():
        print("\nServer running on http://localhost:5000")
        print("Frontend should connect to this URL\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\nFailed to start server - models not loaded")
        print("Check that all .pkl files exist in the same directory")