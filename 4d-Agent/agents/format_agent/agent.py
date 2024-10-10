import ast
import json

from agents.base_agent import BaseAgent
from .prompt import SYSTEM_PROMPT, USER_PROMPT


def extract_text_or_number(str_input):
    if str_input.startswith("[") and str_input.endswith("]"):
        try:
            parsed = ast.literal_eval(str_input)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0]
        except:
            pass

    try:
        parsed = ast.literal_eval(str_input)
        return parsed
    except:
        return str_input


class FormatAgent(BaseAgent):
    def get_prompt(self, input):
        usr_prompt = self.usr_prompt.replace("{{prompt}}", input)
        return self.sys_prompt, usr_prompt

    def parse(self, input):
        attempts = 0
        input_tokens = 0
        output_tokens = 0
        sys_prompt, usr_prompt = self.get_prompt(input)
        output = self.generate(sys_prompt, usr_prompt)
        return output[0].content
        # while attempts < 3:
        #     output, prompt_tokens, completion_tokens = self.generate(sys_prompt, usr_prompt)
        #     input_tokens += prompt_tokens
        #     output_tokens += completion_tokens
        #     try:
        #         temp = json.loads(output.content)
        #         if self.check_format_recursive(temp, self.json_format):
        #             return temp, input_tokens, output_tokens
        #         else:
        #             print(f"Error Format {temp}")
        #     except Exception as e:
        #         print(e)
        #         # print(f"Error Json {output}")
        #         attempts += 1
        # return None, input_tokens, output_tokens

#
# process_agent = ProcessAgent("../assets/config.yaml", sys_prompt=SYSTEM_PROMPT, usr_prompt=USER_PROMPT)
