import pickle
import numpy as np
import pandas as pd
import os
from catboost import CatBoostClassifier

class AsthmaPredictor:
    def __init__(self, model_dir=None):
        self.model = None
        self.model_dir = model_dir if model_dir else os.getcwd()
        self.fields = [
            {'name': 'Age', 'question': 'What is your age?', 'type': 'numeric'},
            {'name': 'Gender', 'question': 'What is your gender? (Male/Female)', 'type': 'categorical'},
            {'name': 'BMI', 'question': 'What is your Body Mass Index (BMI)?', 'type': 'numeric'},
            {'name': 'Smoking_Status', 'question': 'Smoking status? (Never/Former/Current)', 'type': 'categorical'},
            {'name': 'Family_History', 'question': 'Do you have a family history of asthma? (0=No, 1=Yes)', 'type': 'binary'},
            {'name': 'Allergies', 'question': 'Do you have allergies? (None/Dust/Pollen/Pet/Multiple)', 'type': 'categorical'},
            {'name': 'Air_Pollution_Level', 'question': 'What is the air pollution level in your area? (Low/Moderate/High)', 'type': 'categorical'},
            {'name': 'Physical_Activity_Level', 'question': 'What is your level of physical activity? (Sedentary/Moderate/Active)', 'type': 'categorical'},
            {'name': 'Occupation_Type', 'question': 'What type of occupation do you have? (Indoor/Outdoor)', 'type': 'categorical'},
            {'name': 'Comorbidities', 'question': 'Do you have any comorbidities? (None/Diabetes/Hypertension/Both)', 'type': 'categorical'},
            {'name': 'Medication_Adherence', 'question': 'What is your medication adherence level? (value between 0 and 1)', 'type': 'numeric'},
            {'name': 'Number_of_ER_Visits', 'question': 'How many emergency room visits have you had in the past year?', 'type': 'numeric'},
            {'name': 'Peak_Expiratory_Flow', 'question': 'Peak Expiratory Flow (PEF) - value in L/min?', 'type': 'numeric'},
            {'name': 'FeNO_Level', 'question': 'What is your FeNO level (Fractional Exhaled Nitric Oxide)?', 'type': 'numeric'}
        ]
        self.load_model()
        
    def load_model(self):
        try:
            model_path = os.path.join(self.model_dir, 'asthma_prediction.pkl')
            print(f"Looking for model at: {model_path}")
            
            if not os.path.exists(model_path):
                print(f"Error: File not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            print(f"Asthma prediction model loaded successfully!")
            return True
        
        except FileNotFoundError as e:
            print(f"Warning: Asthma model not found!")
            print(f"Model directory: {self.model_dir}")
            print(f"Error: {e}")
            print("Run asthma notebook first or check file paths.")
            return False
        except Exception as e:
            print(f"Error loading asthma model: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def parse_input(self, user_input, field_type, field_name=None):
        user_input = user_input.lower().strip()
        
        if field_type == 'binary':
            if any(word in user_input for word in ['yes', 'y', 'da', 'true', '1']):
                return 1
            elif any(word in user_input for word in ['no', 'n', 'nu', 'false', '0']):
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
            
        elif field_type == 'categorical':
            # Normalizare pentru categorii specifice
            if field_name == 'Gender':
                if 'male' in user_input or 'm' == user_input:
                    return 'Male'
                elif 'female' in user_input or 'f' == user_input:
                    return 'Female'
            elif field_name == 'Allergies':
                if 'none' in user_input in user_input:
                    return 'None'
                elif 'dust' in user_input in user_input:
                    return 'Dust'
                elif 'pollen' in user_input in user_input:
                    return 'Pollen'
                elif 'pet' in user_input in user_input:
                    return 'Pet'
                elif 'multiple' in user_input in user_input:
                    return 'Multiple'
            elif field_name == 'Comorbidities':
                if 'none' in user_input in user_input:
                    return 'None'
                elif 'diabetes' in user_input in user_input:
                    return 'Diabetes'
                elif 'hypertension' in user_input in user_input:
                    return 'Hypertension'
                elif 'both' in user_input in user_input:
                    return 'Both'
            
            return user_input.title()
        
        return None
    
    def make_prediction(self, user_data):
        if self.model is None:
            return None, "Asthma prediction model not loaded!"
        
        try:
            feature_order = [field['name'] for field in self.fields]
            features_dict = {name: [user_data.get(name, 0)] for name in feature_order}
            
            df = pd.DataFrame(features_dict)
            
            probability = self.model.predict_proba(df)[0][1]
            
            return probability, None
        
        except Exception as e:
            return None, f"Error making prediction: {str(e)}"
        
    def format_prediction_response(self, probability):
        risk_percentage = probability * 100
        
        if risk_percentage < 30:
            risk_level = "Low"
            emoji = "✅"
            advice = "Low risk of asthma. Continue to maintain a healthy lifestyle and avoid known allergens."
        elif risk_percentage < 60:
            risk_level = "Moderate"
            emoji = "⚠️"
            advice = "Moderate risk detected. Consider consulting a medical professional for evaluation and preventive measures."
        else:
            risk_level = "High"
            emoji = "🚨"
            advice = "High risk of asthma. Please consult a pulmonologist or medical professional for appropriate evaluation and management as soon as possible."
            
        response_text = (
            f"**Asthma Risk Assessment**\n\n"
            f"{emoji} **Risk Level: {risk_level}**\n"
            f"**Probability of Asthma: {risk_percentage:.1f}%**\n\n"
            f"{advice}\n\n"
            f"*Note: This is a statistical prediction based on machine learning analysis. "
            f"It is not a medical diagnosis. Always consult a medical professional for medical advice.*"
        )
        
        return response_text
    
    def get_total_fields(self):
        return len(self.fields)
    
    def get_field(self, index):
        if 0 <= index < len(self.fields):
            return self.fields[index]
        return None
    
    def is_model_loaded(self):
        return self.model is not None
    
    def start_conversation(self, user_id):
        if not self.is_model_loaded():
            return {
                'intent': 'error',
                'response': 'Asthma prediction is currently unavailable. Please contact support.',
                'user_id': user_id
            }
        
        first_field = self.get_field(0)
        total_fields = self.get_total_fields()
        
        response_text = (
            "**Asthma Risk Assessment**\n\n"
            "I will help you assess your asthma risk. "
            "I need to collect some information about your health and environment.\n\n"
            f"**Question 1 of {total_fields}**\n\n"
            f"{first_field['question']}"
        )
        
        return {
            'intent': 'asthma_prediction',
            'response': response_text,
            'user_id': user_id,
            'collecting_data': True,
            'progress': f"1/{total_fields}",
            'session_data': {
                'collecting_data': True,
                'current_field': 0,
                'user_data': {},
                'context': 'asthma_prediction'
            }
        }
    
    def handle_conversation_step(self, user_message, session_data, user_id):
        current_field_index = session_data['current_field']
        current_field = self.get_field(current_field_index)
        
        parsed_value = self.parse_input(user_message, current_field['type'], current_field['name'])
        
        if parsed_value is None:
            return {
                'intent': 'asthma_prediction',
                'response': f"Sorry, I didn't understand. Please try again.\n\n{current_field['question']}",
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
                f"Perfect!\n\n"
                f"**Question {progress} of {total}**\n\n"
                f"{next_field['question']}"
            )
            
            return {
                'intent': 'asthma_prediction',
                'response': response_text,
                'user_id': user_id,
                'collecting_data': True,
                'progress': f"{progress}/{total}",
                'session_data': session_data
            }
        else:
            probability, error = self.make_prediction(session_data['user_data'])
            
            if error:
                response_text = f"Sorry, an error occurred: {error}"
                prediction_data = None
            else:
                response_text = self.format_prediction_response(probability)
                prediction_data = {
                    'probability': float(probability),
                    'risk_percentage': float(probability * 100)
                }
            
            return {
                'intent': 'asthma_prediction',
                'response': response_text,
                'user_id': user_id,
                'collecting_data': False,
                'prediction': prediction_data,
                'session_data': None
            }
    
    @staticmethod
    def check_keywords(user_message):
        keywords = ['asthma', 'astm', 'breathing', 'wheezing', 'respiratory', 'lung']
        message_lower = user_message.lower()
        return any(word in message_lower for word in keywords)