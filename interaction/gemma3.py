"""
This is a simple script to interact with the gemma3 model.
It uses the ollama library to send a message to the model and print the response.
"""

from ollama import chat

stream = chat(
    model='gemma3:4b',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
