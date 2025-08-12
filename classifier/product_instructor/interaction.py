"""
This is a simple script to interact with the gemma3 model.
It uses the ollama library to send a message to the model and print the response.
"""

import random
import os
from ollama import chat
from PIL import Image


def convert_image_to_valid_type(img_path: str) -> str:
    """
    Convert the image to a valid type(.png and .jpg)
    Details in: https://ollama.com/blog/vision-models
    """
    img = Image.open(img_path)
    suffix = img_path.split(".")[-1]
    if suffix in ["png", "jpg", "jpeg"]:
        return img_path
    else:
        # save image in a temp directory
        tmp_path = os.path.join(
            "tmp", os.path.basename(img_path).replace(suffix, "png")
        )
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        img.save(tmp_path)
        return tmp_path


def pick_a_random_image(kind: str) -> str:
    """
    Pick a random image from the kind
    """
    img_path = os.path.abspath(
        f"train_data/gallery/{kind}/{random.choice(os.listdir(f'train_data/gallery/{kind}'))}"
    )
    return convert_image_to_valid_type(img_path)


def delete_temp_image(img_path: str) -> None:
    """
    Delete the temp image
    """
    if os.path.dirname(img_path) == os.path.abspath("tmp"):  # is path is under /tmp
        if os.path.exists(img_path):
            os.remove(img_path)


image_path = pick_a_random_image("wind")

stream = chat(
    model="gemma3:4b",
    messages=[
        {
            "role": "user",
            "content": "Describe the image in detail.",
            "images": [image_path],
        },
    ],
    stream=True,
)

print(f"The image is {image_path}")
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)

delete_temp_image(image_path)
