from google.adk import Agent

def get_hued(text: str) -> dict:
    text = text.split(' ')
    for i, word in enumerate(text):
        if len(word) > 3:
            word = f'hui{word[3:].lower()}'
        text[i] = word
    text = ' '.join(text)
    return {"status": "success", "report": text}


root_agent = Agent(
    name="hui_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent to huificate"
    ),
    instruction=(
        "You are a helpful agent who can huificate."
    ),
    tools=[get_hued],
)
