import ast

from agents.base_agent import BaseAgent
from .prompt import SYSTEM_PROMPT, USER_PROMPT


class ExtractAgent(BaseAgent):
    def get_prompt(self, input):
        user_prompt = self.usr_prompt.replace("{prompt}", str(input))
        return self.sys_prompt, user_prompt

    def parse(self, input):
        attempts = 0
        sys_prompt, usr_prompt = self.get_prompt(input)
        output = self.generate(sys_prompt, usr_prompt)
        return output[0].content


#
# process_agent = ProcessAgent("../assets/config.yaml", sys_prompt=SYSTEM_PROMPT, usr_prompt=USER_PROMPT)
