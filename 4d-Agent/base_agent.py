import re
import abc
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

class BaseAgent(abc.ABC):
    def __init__(self, model, processor, sys_prompt, json_format=None, max_tokens=1024):
        self.model = model
        self.processor = processor
        self.max_tokens = max_tokens
        self.sys_prompt = sys_prompt
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

    def generate(self, input):
        messages = self.get_prompt(input)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process the vision information (images and videos)
        image_inputs, video_inputs = process_vision_info(messages)

        # Create the input tensors for the model
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text

    @abc.abstractmethod
    def parse(self, input):
        pass

    @abc.abstractmethod
    def get_prompt(self, input):
        pass


if __name__ == "__main__":

    # Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct"
    ).to("cuda:0")

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")