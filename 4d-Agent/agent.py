from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct"
).to("cuda:0")

# Enable flash_attention_2 if needed (commented out for default settings)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# Load the processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Define the input messages
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Prepare input text using the processor
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
)

# Move the inputs to GPU
inputs = inputs.to("cuda:0")

# Forward pass to get hidden states
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# Extract hidden states
# Hidden states are returned as a list of tuples, with each tuple containing the hidden states from each layer
hidden_states = outputs.hidden_states

# Print hidden states information (optional)
for i, layer_hidden_states in enumerate(hidden_states):
    print(f"Layer {i}: Shape of hidden states {layer_hidden_states.shape}")

# Example: Accessing the hidden states of the last layer
last_layer_hidden_states = hidden_states[-1]

# If you want to process these hidden states further, you can continue here
print("Shape of the last layer hidden states:", last_layer_hidden_states.shape)
print(last_layer_hidden_states)
print("mean", torch.mean(last_layer_hidden_states))
print("var", torch.var(last_layer_hidden_states))