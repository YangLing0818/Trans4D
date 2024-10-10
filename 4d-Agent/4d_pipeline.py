import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import AzureOpenAI
# from agents import ProcessAgent, FormatAgent, ExtractAgent
model="chat-v4o"
api_keys = {"api":'', "base": "", "model": ""}
client = AzureOpenAI(
  api_key = "",
  api_version = "",
  azure_endpoint = ""
)
def generate_response(client, model, sys_prompt, usr_prompt, max_tokens):

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.99,
        max_tokens=max_tokens,
    )

    response_message = response.choices[0].message

    return response_message

if __name__ == "__main__":
    from agents.object_extract_agent import ExtractAgent, SYSTEM_PROMPT, USER_PROMPT
    extract_agent = ExtractAgent(client, model, SYSTEM_PROMPT, USER_PROMPT, max_tokens=4096)
    from agents.process_agent import ProcessAgent, SYSTEM_PROMPT, USER_PROMPT
    process_agent = ProcessAgent(client, model, SYSTEM_PROMPT, USER_PROMPT, max_tokens=4096)
    from agents.format_agent import FormatAgent, SYSTEM_PROMPT, USER_PROMPT
    format_agent = FormatAgent(client, model, SYSTEM_PROMPT, USER_PROMPT, max_tokens=4096)

    # user_input = "An ice cube melting to water"
    user_input = "A dog chase a ball"
    subject = extract_agent.parse(user_input)
    print(subject)
    description = process_agent.parse(user_input, subject)
    print(description)
    result = format_agent.parse(description)
    print(result)
