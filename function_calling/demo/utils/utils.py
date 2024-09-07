import streamlit as st

import requests
from typing import Dict, Any, Optional, Callable
from geopy.geocoders import Nominatim
import json

from utils.functions import AVAILABLE_FUNCTIONS

def singleton(cls):
    """ decorator function for a class to make a singleton out of it """
    class_instances = {}

    def get_instance(*args, **kwargs):
        """ Return the parameter-specific unique instance of the class. Create if not exists. """
        key = (cls, args, str(kwargs))
        if key not in class_instances:
            class_instances[key] = cls(*args, **kwargs)
        return class_instances[key]

    return get_instance





def parse_and_call_function(model_output: str, available_functions: Dict[str, Callable] = AVAILABLE_FUNCTIONS):
    """
    Parse the model output and call the corresponding function from the list of available functions.
    """
    try:
        # Parse the model output
        function_call = json.loads(model_output.replace("'", '"'))
        function_name = function_call['name']
        arguments = json.loads(function_call['arguments'])
        
        # Check if the function is in the list of available functions
        if function_name in available_functions:
            result = available_functions[function_name](**arguments)
            return f"Result of {function_name}: {result}"
        else:
            return f"Error: Function '{function_name}' is not available"
    except json.JSONDecodeError:
        return "Error: Invalid JSON in model output"
    except KeyError as e:
        return f"Error: Missing key in function call - {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"




def get_help():
    # create help menu in markdown format
    help_message = """\n
    #### Hello, I'm Draft42!
    I'm a chatbot that can help you with any questions you have.\n

    Here are some commands you can use:
    - `/help`: Displays this help message.
    - `/clear`: Clears the chat history.
    - `/about`: Displays information about this app.
    or you can simply type your message and I will respond.

    I can talk to models as well as some local functions and tools, all in natural^ spoken language. 
    Here are some local tools I've been trained to use:
    - shell: to validate and execute shell commands.

    Some conversation starters:
    - What is the meaning of life?
    - what's 42+42-42*42/42
    - current cpu, memory, disk usage in the system
    """
    return help_message

def clear_chat():
    st.session_state.conversation.clear()
    st.rerun()

def about_app():
    about_message = """
    This is a chatbot app that demonstrates function-calling using local or OPENAI models.
    The app uses Streamlit for the user interface.
    Developed by [Swapnil Patel](https://autoscaler.sh/).
    """
    return about_message



import re

import re

def extract_first_content(text):
    pattern = r'(?:<eot>)*(.+?)(?:<eot>|<eo|$)'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    else:
        # Return None if no match is found
        return None

def clean_special_tokens(text):
    special_tokens = [
        "<eot>",
        "<start_header>",
        "<end_header>",
        "<tool_call>"
    ]
    
    pattern = '|'.join(re.escape(token) for token in special_tokens)
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text.strip()
