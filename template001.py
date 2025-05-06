# This is a Python script that uses the OpenAI API to generate a response based on a given prompt.
# The following building blocks can be used to create an angent that attempts to solve a problem for the end user.

# Input list of messages for the LLM. The role of the developer should describe what the system should do for users.
# When available, the input list should also include any requests from the user.
# Define the different external tools that are available for the LLM to use, such as APIs or database connections.
# For each external tool available, we should do the following:
#   Create a function call that will define the purpose of each available external tool and the parameters it requires.
#   Create a method that takes in the parameters from the function call and properly executes the external tool.
#   The output of this method will need to be sent back to the LLM as a result of it's relevant function call.
# We should use structured outputs to present the end results to the user.
# We can do this by creating a pydantic model that represents how we can present the data to the end user.
# The pydantic model should take into account all data available from the external tools as well as what the user expects.
# We should consider not only a model to represent each single instance of result data, but also model to represent a collection of results.
# Create a response object that will take in the input array of messages and the defined function calls.
# This should be done with the responses.create method.
# We need to process the response object for any function calls the LLM requested.
# Use the function calls to execute the functions, and append the results to the input array of messages.
# After successfully gathering the results of the functions calls, send the results back to the LLM with the responses.parse method.
# This parse call should also take in the pydantic model we created earlier for presenting to the end user.
# We can present the results to the end user by using the response.output_parsed field.

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

import json
import os
import requests

# Load environment variables from .env file
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')


# Template for API requests
url_1 = 'https://www.fruityvice.com'
request_data = {'param_1': 'value_1', 'param_2': 'value_2'}
# Example of a GET request
req_res_get = requests.get(url_1, params=request_data)
# Example of a POST request
req_res_post = requests.post(url_1, data=request_data)
# We can output text with .text, or JSON with .json(), or the raw content with .content
print(req_res_get.content)


# Templates for pydantic models
class ExampleModel(BaseModel):
    field_1: int
    field_2: str
    field_3: tuple[int, int]
    field_4: float

class ExampleModels(BaseModel):
    example_models: list[ExampleModel]


# Template for the function calls (which work with the responses API)
# This example was originally created for an API which returned data about fruits.
tools = [{
    "type": "function",
    "name": "get_fruits",
    "description": "Get fruits with a nutritional value with a specific range.",
    "parameters": {
        "type": "object",
        "properties": {
            "nutrition_name": {
                "type": "string",
                "enum": ['sugar', 'protein', 'fat', 'carbohydrates', 'calories'],
                "description": "Name of the nutritional value."
            },
            "min_value": {
                "type": "number",
                "description": "Minimum value of the nutritional value."
            },
            "max_value": {
                "type": "number",
                "description": "Maximum value of the nutritional value."
            }
        },
        "required": [
            "nutrition_name",
            "min_value",
            "max_value"
        ],
        "additionalProperties": False
    },
    "strict": True
}]


# Template for a method that takes in the parameters from the function call and properly executes the external tool.
def fetch_fruits(parsed_tool_call):
    """
    Fetch fruits based on nutritional value range from the API.

    :param parsed_tool_call: ParsedResponseFunctionToolCall object containing arguments.
    :return: Response from the API as JSON.
    """
    fruit_api_url = 'https://www.fruityvice.com/api/fruit'

    args_as_json = json.loads(parsed_tool_call.arguments)

    params = {
        "min": args_as_json["min_value"],
        "max": args_as_json["max_value"]
    }
    nutrition_name = args_as_json["nutrition_name"]
    url_complete = f"{fruit_api_url}/{nutrition_name}"

    api_res = requests.get(url_complete, params=params)
    # api_res.raise_for_status()  # Raise an error for HTTP issues
    return api_res.json()


# Template for a list of inputs (or messages) for the LLM
inputs=[
    {
        "role": "developer",
        "content": "I am operating a system that is designed to provide responses to users within a defined scope."
    },
    {
        "role": "user",
        "content": "This should be a request to the system that is within the scope of its design."
    }
]


# Template for the OpenAI API call
# Make sure the OpenAI API key is set in the environment variables
client = OpenAI(api_key=openai_api_key)

response_1 = client.responses.create(
    model="gpt-4.1-mini",
    input=inputs,
    tools=tools,
)

# The instructions mentioned that we will call at least one API as part of the pipeline.
# So, for each function call in the response_1.output list, we should execute the appropriate function,
# append that function call to the input list, and then append the result to the input list.

# Example to append the function call to the input list
inputs.append(response_1.output[0])
# Example to append the function call result to the input list
# Note that the output field needs to be a string even though the data is typically represented as JSON,
# hence the use of json.dumps
inputs.append({
    'type': 'function_call_output',
    'call_id': response_1.output[0].call_id,
    'output': json.dumps('output of the method that executed the function call')
})

# Now we can call the responses.parse method to process the updated input list with the results of the function calls.
# Ideally, we should be able to fulfill the user's original request after executing the function calls.
response_2 = client.responses.parse(
    model="gpt-4.1-mini",
    input=inputs,
    tools=tools,
    text_format=ExampleModels
)

# We can see the parsed and formatted output of the response_2 object below.
print(response_2.output_parsed)
