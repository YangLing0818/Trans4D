import pandas as pd
import json
import openai
import abc

import yaml


class BaseAgent(abc.ABC):
    def __init__(self, client, model, sys_prompt, usr_prompt, json_format=None, max_tokens=1024):
        """

        :param config
        :param prompt: system prompt of this agents
        :param json_format: the except json format
        """
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.sys_prompt = sys_prompt
        self.usr_prompt = usr_prompt
        self.json_format = json_format


    def check_format_recursive(self, output, expected_format):

        if isinstance(expected_format, dict):
            if not isinstance(output, dict):
                return False
            for key, expected_type in expected_format.items():
                if key not in output:
                    print(f"Missing key: {key}")
                    return False
                if isinstance(expected_type, list):
                    if not all(isinstance(item, expected_type[0]) for item in output[key]):
                        print(f"List item type mismatch for key: {key}")
                        return False
                elif isinstance(expected_type, dict):
                    if not self.check_format_recursive(output[key], expected_type):
                        return False
                else:
                    if not isinstance(output[key], expected_type):
                        print(f"Type mismatch for key: {key}")
                        return False
        elif isinstance(expected_format, list):
            if output not in expected_format:
                return False
        elif expected_format is None:
            return True
        else:
            if not isinstance(output, expected_format):
                return False
        return True

    def generate(self, sys_prompt, usr_prompt):
        client = self.client
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ]
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.99,
                max_tokens=self.max_tokens,
            )
            response_message = response.choices[0].message
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            return response_message, prompt_tokens, completion_tokens
        except Exception as e:
            print(e)
            return None, 0, 0


    @abc.abstractmethod
    def parse(self, input):
        pass

    @abc.abstractmethod
    def get_prompt(self, input):
        pass
