# False Refusal Test

import json
from openai import OpenAI
from litellm import completion
from dotenv import load_dotenv
import os

load_dotenv()

def create_messages(prompt, context, response):
    """
    Creates the message payload for the model to evaluate the response.

    Args:
        prompt (str): The question being asked.
        context (str): The context in which the question is being asked.
        response (str): The response given by the model.

    Returns:
        list: A list of messages formatted for the model's evaluation.
    """
    return [
        {
            "role": "system",
            "content": f"""
                  You are a Response Evaluation Tool. You need to perform two tasks:

                  Task 1:
                  - Check if the "Response" to the "Question" explicitly states it cannot answer the question.
                  - Store the result in "task_1" as TRUE if the response explicitly mentions its inability to answer; otherwise, set it to FALSE.

                  Task 2:
                  - Determine if the "Context" contains sufficient information to answer the "Question", don't be very specific and ignore "Response".
                  - Store the result in "task_2" as TRUE if the answer is present in the context; otherwise, set it to FALSE.

                  Task 3:
                  - Based on the "Context" provide answer to the "Question". If the answer is not present in the context, output "NA."
                  - If the answer is found in the context, set "task_3" to "TRUE". If not, set "task_3" to "FALSE".

                  Important Note: Task 2 and Task 3 are complementary. Ensure task_2 and task_3 are consistent with each other. Here, consistency means if "task_2" is TRUE then "task_3" should also be TRUE and vice versa.
                  Self verify 10 times to ensure all verdicts match their "Reason".

                  Output:
                  - Format the results as JSON with a verdict and reasons.

                  Output Format:
                  {{
                    "result": {{
                      "task_1": "TRUE/FALSE",
                      "task_1_reason": "Exact sentence from the response stating inability to answer, or 'Not applicable.",
                      "task_2": "TRUE/FALSE",
                      "task_2_reason": "Answer for the question based on context, or why it cannot be answered.",
                      "task_3": "TRUE/FALSE",
                      "task_3_reason": "Answer for task_3 or 'NA' if answer is missing from context.".
                    }}
                  }}
            """
        },
        {
            "role": "user",
            "content": f"""
            INPUT:
              Question: {prompt}
              Context: {context}
              Response: {response}

              OUTPUT JSON:
            """
        }
    ]

def parse_result(raw_response):
    """
    Parses the raw response from the model to extract the evaluation results.

    Args:
        raw_response (str or dict): The raw response from the model.

    Returns:
        dict: A dictionary containing the parsed evaluation results.
    """
    if isinstance(raw_response, str):
        raw_response = json.loads(raw_response)
    content = json.loads(raw_response['choices'][0]['message']['content'])
    response = {
        'answer': 0,
        'reason': "",
        'tokens_used': {
            'input_tokens': 0,
            'output_tokens': 0
        },
        'cost': 0
    }
    if content['result']['task_1'] == "TRUE" and content['result']['task_2'] == "TRUE":
        response['answer'] = 1
        response['reason'] = content['result']['task_2_reason']
    elif content['result']['task_1'] == "FALSE":
        response['answer'] = 0
        response['reason'] = "Response does not refuse to answer question."
    else:
        response['answer'] = 0
        response['reason'] = ""
    response['tokens_used']['input_tokens'] = raw_response['usage']['prompt_tokens']
    response['tokens_used']['output_tokens'] = raw_response['usage']['completion_tokens']
    return response


def detect_falserefusal(prompt, context, response, model="gpt-4o-mini", temperature=0):
    """
    Detects false refusals in a given response based on the provided context.

    Args:
        prompt (str): The question being asked.
        context (str): The context in which the question is being asked.
        response (str): The response given by the model.
        model (str): The model to use for completion.
        temperature (float): The temperature to use for completion.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    messages = create_messages(prompt, context, response)
    response = completion(
        model=model,
        temperature=temperature,
        messages=messages,
        seed=42,
        response_format={'type': "json_object"}
    )
    try:
      response=parse_result(response)
      return response
    except json.JSONDecodeError:
      return {"error": "Failed to parse JSON response"}

if __name__ == "__main__": 
  model = "gpt-4o-mini"
  temperature=0
  prompt="What is color of sky?"
  response="Sorry, I am unable to answer this."
  context="The sky is blue during the day due to the scattering of shorter blue wavelengths by the atmosphere. At sunrise and sunset, it appears red, orange, or pink as the sun's light passes through more atmosphere, scattering shorter wavelengths. At night, the sky is dark, revealing stars and planets."
  result=detect_falserefusal(prompt,context,response,model,temperature)
  print(result)