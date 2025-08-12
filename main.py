from transformers import pipeline

pipe = pipeline("image-text-to-text", model="HuggingFaceTB/SmolVLM2-2.2B-Instruct", device="cuda")
pipe = pipeline("image-text-to-text", model="HuggingFaceTB/SmolVLM2-256M-Video-Instruct", device="cuda")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
print(pipe(text=messages))