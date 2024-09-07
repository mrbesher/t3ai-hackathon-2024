import streamlit as st
import json
from utils.conversation import Conversation
from utils.utils import get_help, clear_chat, about_app, singleton, AVAILABLE_FUNCTIONS, parse_and_call_function, clean_special_tokens, extract_first_content
from utils.functions import available_tools
from tools.tools import Tools
from tools.text_response import TextResponse

from typing import Any, Dict, Union, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import torch

default_sys = "Sen yardımcı bir asistansın. Dışarıdan bir fonksiyona erişimin yok."
device_name = torch.cuda.get_device_name()

@singleton
class GenAIModel:
    def __init__(self, system_message: str = "I'm an AI assistant here to help you with any questions you have."):
        """
        Initializes a new instance of the ModelManager.
        """
        self.model_name = "t3ai-org/pt-model"
        self.lora_adapters = "mrbesher/t3ai-cogitators-fc-v3"
        self.model = None
        self.initialize_model()
        self.system_message = system_message

    def initialize_model(self):
        """
        Initializes the Model based on the configuration.
        """
        self.device = Accelerator().device
        self.model = AutoModelForCausalLM.from_pretrained(self.lora_adapters, device_map="auto")
        self.model.generation_config.temperature=None
        self.model.generation_config.top_p=None
        self.tokenizer = AutoTokenizer.from_pretrained(self.lora_adapters)


    def make_few_shot_example(self):
        messages = [
            {"role": "user", "content": "Fas'ta hava durumu nasıl?"},
            {'role': 'tool_call', 'content': {'name': 'get_city_weather', 'arguments': '{"city": "Morocco"}'}}, 
            {'role': 'tool_response', 'content': """{'temperature': 19.69, 'humidity': 93, 'wind_speed': 2}"""},
            {'role': 'assistant', "content": "Şu anda Fas'ta sıcaklık 19.69 derece, nem %93 ve rüzgar hızı 2'di"}
        ]

        return messages #self.tokenizer.apply_chat_template(messages, tokenize = False)


    def generate_response(self, data: str, response_model: Union[None, Any] = TextResponse, tools: List = None, tool_response = None, selected_tool = None):
        """
        Generates a response from the GPT model based on the provided data and model settings.
        """
        if data.startswith("User:"):
            data  = data.replace("User: ","")
        sys_dict = {"role": "system", "content": default_sys}
        
        if tools:
            sys_dict.update({"tools": tools, "content": self.system_message})
        try:
            messages = [sys_dict] + self.make_few_shot_example() +[{"role": "user", "content": data}]
            if selected_tool:
                messages.extend([{'role': 'tool_call', 'content': selected_tool}, {'role': 'tool_response', 'content': tool_response}])
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            len_prompt = len(formatted_prompt[0])
            formatted_prompt = formatted_prompt.to(self.device)
            response = self.model.generate(formatted_prompt, max_new_tokens=500, eos_token_id= 128002, do_sample = False)
            decoded_response = self.tokenizer.decode(response[0][len_prompt:], skip_special_tokens=False)
            if "<start_header>" in decoded_response:
                decoded_response = decoded_response.split("<start_header>")[0]
            decoded_response = extract_first_content(decoded_response)
            decoded_response = clean_special_tokens(decoded_response)
            return decoded_response
        except Exception as e:
            print(f"An error occurred while generating the response: {e}")
            return None

    


generic_system_message = """Sen aşağıdaki fonksiyonlara erişimi olan yardımcı bir asistansın. Kullanıcı sorusuna yardımcı olabilmek için bir veya daha fazla fonksiyon çağırabilirsin. Fonksiyon parametreleri ile ilgili varsayımlarda bulunma. Her fonksiyon çağrısı fonksiyon ismi ve parametreleri ile olmalıdır. İşte, kullanabileceğin fonksiyonlar:"""
model_avatar =  "🦙"
model_name = "t3ai-cogitators-fc-v2"
functions_list = [tool['name'] for tool in available_tools]

with st.sidebar:
    selected_functions = st.multiselect(
        "Select Functions:",
        options=functions_list,
        default=[functions_list[0]],
    )
    st.write(f"Selected Functions: {', '.join(selected_functions)}")

if "conversation" not in st.session_state:
    st.session_state.conversation = Conversation()

for message in st.session_state.conversation.messages:    
    if message["role"] == "user":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    elif message["role"] == "tool":
        with st.chat_message(message["role"], avatar='🛠️'):
            st.markdown(message['content'])
    else:
        with st.chat_message(message["role"], avatar=model_avatar):
            st.markdown(message["content"])

prompt = st.chat_input("Type something...", key="prompt")
import json

# ... (previous code remains the same)

if prompt:
    # ... (previous code for handling commands remains the same)

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.conversation.add_message("user", prompt)   
    model = GenAIModel(system_message=generic_system_message)
    full_prompt = st.session_state.conversation.format_for_gpt()
    # Generate initial response
    tools = [tool for tool in available_tools if tool["name"] in selected_functions ]
    initial_response = model.generate_response(full_prompt, tools=available_tools)
    # initial_response = """{'name': 'get_city_weather', 'arguments': '{"city": "istanbul"}'}"""          
    print(f"initial_response is: {initial_response}")
    try:
        # Try to parse the response as a function call
        function_call = eval(initial_response)
        print(f"function_call: {function_call}")
        if 'name' in function_call and function_call['name'] in selected_functions:
            print("It's a valid function call")
            function_name = function_call['name']
            arguments = eval(function_call['arguments'])
            
            # Call the function
            function_result = AVAILABLE_FUNCTIONS[function_name](**arguments)
            print(f"Function result: {function_result}")
            # Display the function result
            with st.chat_message("tool", avatar='🛠️'):
                st.markdown(str(function_result))
            st.session_state.conversation.add_message("tool", str(function_result))
            
            # Generate final response with tool output
            final_response = model.generate_response(
                full_prompt, 
                tools=available_tools, 
                tool_response=str(function_result), 
                selected_tool=function_call
            )
            
            with st.chat_message(model_name, avatar=model_avatar):
                st.markdown(final_response)
            st.session_state.conversation.add_message(model_name, final_response)
        else:
            # It's not a valid function call, treat it as a raw text response
            raise json.JSONDecodeError("Not a valid function call", initial_response, 0)
    except Exception as e:
        print(e)
        # It's a raw text response
        with st.chat_message(model_name, avatar=model_avatar):
            st.markdown(initial_response)
        st.session_state.conversation.add_message(model_name, initial_response)