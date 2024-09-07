import json
from typing import Any, Dict, Union, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import torch

# Assuming these imports and functions are available in your project structure
from utils.utils import get_help, clear_chat, about_app, singleton, AVAILABLE_FUNCTIONS, parse_and_call_function, clean_special_tokens, extract_first_content
from utils.functions import available_tools
from tools.tools import Tools

default_sys = "Sen yardımcı bir asistansın. Dışarıdan bir fonksiyona erişimin yok."
device_name = torch.cuda.get_device_name()

@singleton
class GenAIModel:
    def __init__(self, system_message: str = "I'm an AI assistant here to help you with any questions you have."):
        self.model_name = "t3ai-org/pt-model"
        self.lora_adapters = "mrbesher/t3ai-cogitators-fc-v3"
        self.model = None
        self.initialize_model()
        self.system_message = system_message

    def initialize_model(self):
        self.device = Accelerator().device
        self.model = AutoModelForCausalLM.from_pretrained(self.lora_adapters, device_map="auto")
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.lora_adapters)

    def make_few_shot_example(self):
        messages = [
            {"role": "user", "content": "Fas'ta hava durumu nasıl?"},
            {'role': 'tool_call', 'content': {'name': 'get_city_weather', 'arguments': '{"city": "Morocco"}'}}, 
            {'role': 'tool_response', 'content': """{'temperature': 19.69, 'humidity': 93, 'wind_speed': 2}"""},
            {'role': 'assistant', "content": "Şu anda Fas'ta sıcaklık 19.69 derece, nem %93 ve rüzgar hızı 2'di"}
        ]
        return messages

    def generate_response(self, data, tools = None, tool_response = None, selected_tool= None):
        if data.startswith("User:"):
            data = data.replace("User: ", "")
        sys_dict = {"role": "system", "content": default_sys}
        
        if tools:
            sys_dict.update({"tools": tools, "content": self.system_message})
        try:
            messages = [sys_dict] + self.make_few_shot_example() + [{"role": "user", "content": data}]
            if selected_tool:
                messages.extend([{'role': 'tool_call', 'content': selected_tool}, {'role': 'tool_response', 'content': tool_response}])
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            len_prompt = len(formatted_prompt[0])
            formatted_prompt = formatted_prompt.to(self.device)
            response = self.model.generate(formatted_prompt, max_new_tokens=500, eos_token_id=128002, do_sample=False)
            decoded_response = self.tokenizer.decode(response[0][len_prompt:], skip_special_tokens=False)
            if "<start_header>" in decoded_response:
                decoded_response = decoded_response.split("<start_header>")[0]
            decoded_response = extract_first_content(decoded_response)
            decoded_response = clean_special_tokens(decoded_response)
            return decoded_response
        except Exception as e:
            print(f"An error occurred while generating the response: {e}")
            return None

def main():
    generic_system_message = """Sen aşağıdaki fonksiyonlara erişimi olan yardımcı bir asistansın. Kullanıcı sorusuna yardımcı olabilmek için bir veya daha fazla fonksiyon çağırabilirsin. Fonksiyon parametreleri ile ilgili varsayımlarda bulunma. Her fonksiyon çağrısı fonksiyon ismi ve parametreleri ile olmalıdır. İşte, kullanabileceğin fonksiyonlar:"""
    model = GenAIModel(system_message=generic_system_message)

    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break

        initial_response = model.generate_response(user_input, tools=available_tools)
        print(f"Initial response: {initial_response}")

        try:
            function_call = eval(initial_response)
            if 'name' in function_call and function_call['name'] in [tool['name'] for tool in available_tools]:
                function_name = function_call['name']
                arguments = eval(function_call['arguments'])
                
                function_result = AVAILABLE_FUNCTIONS[function_name](**arguments)
                print(f"Tool used: {function_name}")
                print(f"Tool result: {function_result}")
                
                final_response = model.generate_response(
                    user_input, 
                    tools=available_tools, 
                    tool_response=str(function_result), 
                    selected_tool=function_call
                )
                print(f"Final response: {final_response}")
            else:
                raise json.JSONDecodeError("Not a valid function call", initial_response, 0)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Response: {initial_response}")

if __name__ == "__main__":
    main()