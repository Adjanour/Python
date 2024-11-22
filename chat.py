from openai import OpenAI

client = OpenAI()
messages = [
    {"role": "system", "content": "You are a helpful and friendly assistant."},
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "It's sunny and warm today."},
    {"role": "user", "content": "Great! What about tomorrow?"}
]
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
