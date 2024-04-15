import os
from openai import OpenAI

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

completion = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "system", "content": "Always answer in rhymes."},
        {"role": "user", "content": "Coding is fun"}
    ],
    temperature=0.7,
)

print(completion.choices[0].message)
