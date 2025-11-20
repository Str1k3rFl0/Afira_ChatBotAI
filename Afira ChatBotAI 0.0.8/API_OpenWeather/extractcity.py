import re

def extract_city(text):
    text = text.lower()
    
    match = re.search(r'(weather?:\s+forecast)?\s+in\s+([a-zA-Z\s]+)', text)
    if match:
        return match.group(2).strip()
    
    match = re.search(r'in\s+([a-zA-Z\s]+)$', text)
    if match and "weather" in text:
        return match.group(1).strip()
    
    match = re.search(r'forecast\s+for\s+([a-zA-Z\s]+)', text)
    if match:
        return match.group(1).strip()
    
    return None