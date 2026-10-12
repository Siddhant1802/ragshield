from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Load the API key from the .env file into the environment
load_dotenv()

# Create the model. temperature=0 means consistent, non-random answers
llm = ChatAnthropic(model="claude-haiku-5-5")

# Send a question and wait for the answer
response = llm.invoke("In one sentence, what is RAG in AI?")

# Print just the text of the answer
print(response.content)