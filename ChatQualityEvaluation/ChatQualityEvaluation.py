import json
from openai import OpenAI
from litellm import completion
from dotenv import load_dotenv
import os

load_dotenv()

def create_messages(conversation_history, latest_query, latest_llm_response):
    """
    Creates the message payload for the model to evaluate the response.

    Args:
        conversation_history (str): Dialogue between User and AI.
        latest_query (str): last response by User.
        latest_llm_reponse (str): last reponse given by AI.

    Returns:
        list: A list of messages formatted for the model's evaluation.
    """
    return [
        {
            "role": "system",
            "content": f"""
                  You are a Conversation Flow Quality evaluator, acting as an LLM Judge. Your task is to assess the quality of a conversation between a user and an AI assistant based on specific criteria. You will be provided with the conversation history, the last query from the user, and the last response from the AI.
            """
        },
        {
            "role": "user",
            "content": f"""
            First, review the entire conversation history: {conversation_history}
            Now, consider the most recent interaction:

            Last user query: {latest_query}

            Last AI response: {latest_llm_response}

            Analyze the conversation thoroughly, focusing on these three criteria:

            1. Is the conversation stuck in redundant responses? (Repeated or non-progressive responses)
            2. Does the AI ignore the user's instructions or information passed?
            3. Does the AI misrepresent or misinterpret information passed by the user?
            Important Notes:
            1. If the AI cannot provide an answer because the user has denied the request, it should not be considered an issue.
            2. If the AI ends the conversation by asking more questions, offering support, or properly concluding, it is not considered an issue.
            3. If the AI asks a follow-up question (e.g., offering an alternative or different options), this effort to continue the conversation should be noted. However, the AI should still consider and provide all relevant options or alternatives related to the user's initial request for a more complete evaluation.
            Use an inner monologue to evaluate each criterion carefully before making your final judgment. Consider the entire conversation flow, not just the last interaction.

            <inner_monologue>
            Criterion 1: [Analyze for redundant responses]
            Criterion 2: [Analyze for ignoring user instructions/information]
            Criterion 3: [Analyze for misrepresentation/misinterpretation]
            </inner_monologue>

            After your thorough evaluation, provide your output in the following JSON format:

            <output>
            {{
              "answer": [true/false],
              "reason": [Concise explanation of your evaluation, touching on all three criteria]
            }}
            </output>
            The "answer" should be true if any of the three criteria are met (i.e., if there are issues with the conversation quality), and false if none of the criteria are met (i.e., if the conversation quality is good).

            Ensure that your evaluation is thorough and your reasoning is clear and concise. Do not include any text outside of the JSON format in your final output.
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
    if content['answer']=="true" or content['answer']==True:
        response['answer'] = 1
        response['reason'] = content['reason']
    else:
        response['answer'] = 0
        response['reason'] = ""
    response['tokens_used']['input_tokens'] = raw_response['usage']['prompt_tokens']
    response['tokens_used']['output_tokens'] = raw_response['usage']['completion_tokens']
    return response


def preprocessing(conversation):
    """Extracting latest user query and latest llm response from conversation"""
    latest_query=""
    latest_llm_response=""
    for itr in conversation.split("\n"):
        if "USER".lower() in itr.lower().split(":"):
            latest_query=itr
        else:
            latest_llm_response=itr
    return conversation,latest_query,latest_llm_response


def detect_chatquality(conversation, context="", model="gpt-4o-mini", temperature=0):
    """
    Detects false refusals in a given response based on the provided context.

    Args:
        conversation (str): chat conversation.
        context (str): context or knowledge used by AI/LLM to answer user.
        model (str): The model to use for completion.
        temperature (float): The temperature to use for completion.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    conversation_history, latest_query, latest_llm_response=preprocessing(conversation)
    messages = create_messages(conversation_history, latest_query, latest_llm_response)
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
  conversation=f"""
  USER: I am looking for a restaurant called the Gandhi. 
  AI: The Gandhi is at 72 Regent Street City Centre. Would you like a reservation or more information? 
  USER: I'd like a reservation for 7 people Monday at 15:30 please. 
  AI: Unable to book at that time. Go away.
  """
  context=""
  result=detect_chatquality(conversation,context,model,temperature)
  print(result)