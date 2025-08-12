# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import os

IMAGE_FOLDER = "test_images"
V1_256M = "HuggingFaceTB/SmolVLM-256M-Instruct"
V1_500M = "HuggingFaceTB/SmolVLM-500M-Instruct"
V2_256M = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
V2_500M = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

MODEL_NAME = V2_500M

torch.set_default_device("cuda")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME)

#processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
#model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")

model.to("cuda")

print("Using Device: ", model.device)



def generate_text(image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": ""}
            ]
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])

if __name__ == "__main__": 
    for image_path in os.listdir(IMAGE_FOLDER):
        print(f"Processing {image_path}")
        image_path = os.path.join(IMAGE_FOLDER, image_path)
        print(generate_text(image_path))
        print("="*100)