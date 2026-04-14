
from langchain_aws import ChatBedrockConverse
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_classic.chains import ConversationChain

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


#4 Create a Function for Conversation Chain - Input text + Memory
def demo_conversation(input_text,memory):
    llm_chain_data=demo_chatbot()
    llm_conversation = ConversationChain(
    llm=llm_chain_data, 
    memory=memory,
    verbose=True
    )
    
#5 Chat response using invoke (Prompt template)
    chat_reply=llm_conversation.invoke(input_text)
    return chat_reply['response']