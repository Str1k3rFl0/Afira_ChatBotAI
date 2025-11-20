import requests

OPENWEATHER_API_KEY = "YOUR_API_CODE"

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    try:
        r = requests.get(url)
        data = r.json()
        
        if data.get("cod") != 200:
            return None
        
        return {
            "city": data["name"],
            "temp": data["main"]["temp"],
            "feels": data["main"]["feels_like"],
            "desc": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind": data["wind"]["speed"]
        }
    
    except Exception:
        return None