from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import json
import random
from collections import Counter

app = Flask(__name__)
CORS(app)

model = None
label_encoder = None
vocab = None
word2idx = None
idf = None
intents_data = None

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
    
@app.route('/predict', method=['POST'])
def predict():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty messages'}), 400
        
        tfidf_vector = text_to_tfidf(user_message)
        
        prediction = model.predict(tfidf_vector)[0]
        intent_name = label_encoder.inverse_transform([prediction])[0]
        
        probabilities = model.predict_proba(tfidf_vector)[0]
        confidence = float(np.max(probabilities))
        
        response_text = "I'm not sure how to respond to that."
        for intent in intents_data['intents']:
            if intent['name'] == intent_name:
                response_text = random.choice(intent['responses'])
                break
            
        print(f"User: '{user_message}' -> Intent: {intent_name} ({confidence * 100:.1f}%)")
        
        return jsonify({
            'intent': intent_name,
            'confidence': confidence,
            'response': response_text
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
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