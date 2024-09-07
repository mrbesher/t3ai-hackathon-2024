
from typing import Dict, Any, Optional, Callable
from geopy.geocoders import Nominatim

import requests
import random

apikey= "KEY"

def get_geo_location(city_name: str):
    try:
        geolocator = Nominatim(user_agent="MyApp")
        location = geolocator.geocode("İstanbul")
        return location.latitude,  location.longitude
    except Exception as e:
        print(e)
        return None,None

def calculate_age(birth_date: str) -> int:
    """
    Calculate the age based on the given birth date.
    """
    born = datetime.strptime(birth_date, "%Y-%m-%d")
    today = datetime.now()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))



def get_city_weather(city: str) -> Optional[Dict[str, Any]]:
    """
    Get the current weather for a specified city using the OpenWeatherMap API.
    
    :param city: Name of the city
    :param api_key: Your OpenWeatherMap API key
    :return: Dictionary containing weather information or None if an error occurs
    """
    print("Calling get_city_weather")
    lat,lon = get_geo_location(city)
    if not lat or not lon:
        return {
        "temperature": round(random.uniform(20, 30), 1),  # Temperature between -5°C and 35°C
        "humidity": random.randint(30, 100),  # Humidity between 30% and 100%
        "wind_speed": round(random.uniform(0, 20), 1)  # Wind speed between 0 and 20 m/s
    }
    base_url = f"https://api.tomorrow.io/v4/weather/realtime?location={lat},{lon}&apikey={apikey}"
    try:
        response = requests.get(base_url)
        response.raise_for_status()  # Raise an exception for bad responses
        weather_data = response.json()
        weather_data = weather_data["data"]["values"]
        return {
            "temperature": weather_data["temperature"],
            "humidity": weather_data["humidity"],
            "wind_speed": weather_data["windSpeed"]
        }

    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None




AVAILABLE_FUNCTIONS: Dict[str, Callable] = {
    'calculate_age': calculate_age,
    'get_city_weather': get_city_weather,
}


available_tools = [

    {
        'name': 'get_city_weather',
        'description': 'Get the current weather for a specified city',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string',
                    'description': 'Name of the city'
                }
            },
            'required': ['city']
        }
    },
        {
        'name': 'calculate_age',
        'description': 'Calculate the age based on birth date',
        'parameters': {
            'type': 'object',
            'properties': {
                'birth_date': {
                    'type': 'string',
                    'format': 'date',
                    'description': 'The birth date of the person'
                }
            },
            'required': ['birth_date']
        }
    },
]


if __name__ == "__main__":
    data = get_city_weather("istanbul")
    print(data)
