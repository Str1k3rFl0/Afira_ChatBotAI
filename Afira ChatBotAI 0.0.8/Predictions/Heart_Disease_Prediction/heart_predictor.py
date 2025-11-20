import pickle
import numpy as np
import os

class HeartDiseasePredictor:
    def __init__(self, model_dir=None):
        self.theta = None
        self.scaler = None
        self.model_dir = model_dir if model_dir else os.getcwd()
        self.fields = [
            {'name': 'male', 'question': 'Are you male? (yes/no)', 'type': 'binary'},
            {'name': 'age', 'question': 'What is your age?', 'type': 'numeric'},
            {'name': 'education', 'question': 'Education level (1-4, where 1=some high school, 4=college)', 'type': 'numeric'},
            {'name': 'currentSmoker', 'question': 'Are you currently a smoker? (yes/no)', 'type': 'binary'},
            {'name': 'cigsPerDay', 'question': 'How many cigarettes per day?', 'type': 'numeric'},
            {'name': 'BPMeds', 'question': 'Are you on blood pressure medication? (yes/no)', 'type': 'binary'},
            {'name': 'prevalentStroke', 'question': 'Have you had a stroke? (yes/no)', 'type': 'binary'},
            {'name': 'prevalentHyp', 'question': 'Do you have hypertension? (yes/no)', 'type': 'binary'},
            {'name': 'diabetes', 'question': 'Do you have diabetes? (yes/no)', 'type': 'binary'},
            {'name': 'totChol', 'question': 'What is your total cholesterol level (mg/dL)?', 'type': 'numeric'},
            {'name': 'sysBP', 'question': 'What is your systolic blood pressure?', 'type': 'numeric'},
            {'name': 'diaBP', 'question': 'What is your diastolic blood pressure?', 'type': 'numeric'},
            {'name': 'BMI', 'question': 'What is your BMI (Body Mass Index)?', 'type': 'numeric'},
            {'name': 'heartRate', 'question': 'What is your heart rate (bpm)?', 'type': 'numeric'},
            {'name': 'glucose', 'question': 'What is your glucose level (mg/dL)?', 'type': 'numeric'}
        ]
        self.load_models()
        
    def load_models(self):
        try:
            theta_path = os.path.join(self.model_dir, 'heart_disease_theta.pkl')
            scaler_path = os.path.join(self.model_dir, 'heart_disease_scaler.pkl')
            
            print(f"Looking for theta at: {theta_path}")
            print(f"Looking for scaler at: {scaler_path}")
            
            if not os.path.exists(theta_path):
                print(f"ERROR: File not found: {theta_path}")
                return False
            if not os.path.exists(scaler_path):
                print(f"ERROR: File not found: {scaler_path}")
                return False
            
            with open(theta_path, 'rb') as f:
                self.theta = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            print(f"Heart disease prediction models loaded successfully!")
            print(f"Theta shape: {self.theta.shape}")
            print(f"Scaler type: {type(self.scaler).__name__}")
            return True
            
        except FileNotFoundError as e:
            print(f"Warning: Heart Disease models not found!")
            print(f"Model directory: {self.model_dir}")
            print(f"Error: {e}")
            print("Run heartdisprediction.ipynb first or check file paths.")
            return False
        except Exception as e:
            print(f"Error loading heart disease models: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def parse_input(self, user_input, field_type):
        user_input = user_input.lower().strip()
        
        if field_type == 'binary':
            if any(word in user_input for word in ['yes', 'y', 'true', '1']):
                return 1
            elif any(word in user_input for word in ['no', 'n', 'false', '0']):
                return 0
            else:
                return None
        
        elif field_type == 'numeric':
            try:
                numbers = [float(s) for s in user_input.split() if s.replace('.', '').replace('-', '').isdigit()]
                if numbers:
                    return numbers[0]
                return None
            except:
                return None 
        
        return None
    
    def make_prediction(self, user_data):
        if self.theta is None or self.scaler is None:
            return None, "Heart disease prediction model not loaded"
        
        try:
            features = [
                user_data['male'],
                user_data['age'],
                user_data['education'],
                user_data['currentSmoker'],
                user_data['cigsPerDay'],
                user_data['BPMeds'],
                user_data['prevalentStroke'],
                user_data['prevalentHyp'],
                user_data['diabetes'],
                user_data['totChol'],
                user_data['sysBP'],
                user_data['diaBP'],
                user_data['BMI'],
                user_data['heartRate'],
                user_data['glucose']
            ]
            
            numerical_indices = [1, 4, 9, 10, 11, 12, 13, 14]
            features_array = np.array(features).reshape(1, -1)
            
            features_to_scale = features_array[:, numerical_indices]
            features_scaled = self.scaler.transform(features_to_scale)
            features_array[:, numerical_indices] = features_scaled
            
            features_with_bias = np.hstack([np.ones((1, 1)), features_array])
            
            probability = self.sigmoid(features_with_bias.dot(self.theta))[0][0]
            
            return probability, None
            
        except Exception as e:
            return None, f"Error making prediction: {str(e)}"
    
    def format_prediction_response(self, probability):
        risk_percentage = probability * 100
        
        if risk_percentage < 10:
            risk_level = "Low"
            emoji = "✅"
            advice = "Great! Maintain a healthy lifestyle with regular exercise and balanced diet."
        elif risk_percentage < 30:
            risk_level = "Moderate"
            emoji = "⚠️"
            advice = "Consider lifestyle improvements and regular health check-ups."
        else:
            risk_level = "High"
            emoji = "🚨"
            advice = "Please consult with a healthcare professional for a proper medical evaluation as soon as possible."
        
        response_text = (
            f"**Heart Disease Risk Assessment**\n\n"
            f"{emoji} **Risk Level: {risk_level}**\n"
            f"**10-Year CHD Probability: {risk_percentage:.1f}%**\n\n"
            f"{advice}\n\n"
            f"*Note: This is a statistical prediction based on the Framingham Heart Study. "
            f"It is not a medical diagnosis. Always consult healthcare professionals for medical advice.*"
        )
        
        return response_text
    
    def get_total_fields(self):
        return len(self.fields)
    
    def get_field(self, index):
        if 0 <= index < len(self.fields):
            return self.fields[index]
        return None
    
    def is_model_loaded(self):
        return self.theta is not None and self.scaler is not None
    
    def start_conversation(self, user_id):
        if not self.is_model_loaded():
            return {
                'intent': 'error',
                'response': 'Heart disease prediction is currently unavailable. Please contact support.',
                'user_id': user_id
            }
        
        first_field = self.get_field(0)
        total_fields = self.get_total_fields()
        
        response_text = (
            "**Heart Disease Risk Assessment**\n\n"
            "I'll help you assess your 10-year risk of coronary heart disease (CHD). "
            "I need to collect some health information.\n\n"
            f"**Question 1 of {total_fields}**\n\n"
            f"{first_field['question']}"
        )
        
        return {
            'intent': 'heart_disease_prediction',
            'response': response_text,
            'user_id': user_id,
            'collecting_data': True,
            'progress': f"1/{total_fields}",
            'session_data': {
                'collecting_data': True,
                'current_field': 0,
                'user_data': {},
                'context': 'heart_disease_prediction'
            }
        }
    
    def handle_conversation_step(self, user_message, session_data, user_id):
        current_field_index = session_data['current_field']
        current_field = self.get_field(current_field_index)
        
        parsed_value = self.parse_input(user_message, current_field['type'])
        
        if parsed_value is None:
            return {
                'intent': 'heart_disease_prediction',
                'response': f"Sorry, I didn't understand that.\n\n{current_field['question']}",
                'user_id': user_id,
                'collecting_data': True,
                'progress': f"{current_field_index + 1}/{self.get_total_fields()}",
                'session_data': session_data
            }
        
        session_data['user_data'][current_field['name']] = parsed_value
        session_data['current_field'] += 1
        
        if session_data['current_field'] < self.get_total_fields():
            next_field = self.get_field(session_data['current_field'])
            progress = session_data['current_field'] + 1
            total = self.get_total_fields()
            
            response_text = (
                f"Got it!\n\n"
                f"**Question {progress} of {total}**\n\n"
                f"{next_field['question']}"
            )
            
            return {
                'intent': 'heart_disease_prediction',
                'response': response_text,
                'user_id': user_id,
                'collecting_data': True,
                'progress': f"{progress}/{total}",
                'session_data': session_data
            }
        else:
            probability, error = self.make_prediction(session_data['user_data'])
            
            if error:
                response_text = f"Sorry, there was an error: {error}"
                prediction_data = None
            else:
                response_text = self.format_prediction_response(probability)
                prediction_data = {
                    'probability': float(probability),
                    'risk_percentage': float(probability * 100)
                }
            
            return {
                'intent': 'heart_disease_prediction',
                'response': response_text,
                'user_id': user_id,
                'collecting_data': False,
                'prediction': prediction_data,
                'session_data': None 
            }
    
    @staticmethod
    def check_keywords(user_message):
        keywords = ['heart', 'cardiac', 'cardiovascular', 'chd', 'coronary']
        message_lower = user_message.lower()
        return any(word in message_lower for word in keywords)