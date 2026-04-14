
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage

# 2a Write a function for invoking model- client connection with Bedrock
def demo_chatbot(messages):
    demo_llm = ChatBedrockConverse(
        credentials_profile_name='default',
        model="us.deepseek.r1-v1:0",
        region_name="us-east-1",
        temperature=0.1,
        max_tokens=1000
    )
    return demo_llm.invoke(messages)

# 2b Test out the LLM with invoke method
messages = [
    HumanMessage(content="What is an LLM")
]

response = demo_chatbot(messages)
print(response)
