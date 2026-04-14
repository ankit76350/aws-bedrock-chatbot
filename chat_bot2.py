
from langchain_aws import ChatBedrockConverse
from langchain_classic.memory import ConversationSummaryBufferMemory

# 2a Return the LLM object (not an invoked response) so it can be reused
def demo_chatbot():
    demo_llm = ChatBedrockConverse(
        credentials_profile_name='default',
        model="us.deepseek.r1-v1:0",
        region_name="us-east-1",
        temperature=0.1,
        max_tokens=1000
    )
    return demo_llm

# 3 Create a Function for ConversationSummaryBufferMemory
def demo_memory():
    llm_data = demo_chatbot()
    memory = ConversationSummaryBufferMemory(llm=llm_data, max_token_limit=2000)
    return memory

response = demo_memory()
print(response)
