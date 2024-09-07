from pydantic import BaseModel, Field
from typing import Union
import json

# from tools.shell import ShellCommand
from tools.text_response import TextResponse
# from instructor import OpenAISchema

class Tools():
    """
    Represents a response from a GPT model, including additional metadata for context and debugging.
    """
    
    action: TextResponse = Field(
        ...,
        description="Best function to respond to user's query, use text_response for most cases and other functions for specific use cases."
    )
    # def process_tool_reponse(self, tool_reponse):
    #     try:
    #         if 
    #     except Exception as e:
    #         print(e)


    # def process(self):
    #     """Process the action."""
    #     output = self.action.process()
    #     # restrict the output to limited openai tokens length, some might be less than 4096
    #     output = output[:4096]
    #     return output