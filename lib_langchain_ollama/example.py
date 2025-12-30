import asyncio

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


async def run():
    model = ChatOllama(
        model="mistral",
        validate_model_on_init=True,
        temperature=0.8,
        num_predict=256,
    )
    # messages = [
    #     ("system", "You are a helpful translator. Translate the user sentence to French."),
    #     ("human", "I love programming."),
    # ]
    # async for chunk in model.astream(messages):
    #     print(chunk.content)

    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    result = ''
    messages = agent.invoke({"messages": [
        {"role": "user", "content": "what is the weather in sf"},
        {"role": "user", "content": "what is the weather in philadelphia"},
    ]})
    for message in messages.get('messages'):
        if isinstance(message, HumanMessage):
            continue
        if isinstance(message, ToolMessage):
            print(message.content)
            continue
        if message.content:
            result = f"{result}\n{message.content}"
    print(result)


if __name__ == '__main__':
    asyncio.run(run())
