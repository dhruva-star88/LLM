API_KEY ="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"
USER_INPUT = 'Paris'

from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain import PromptTemplate

llm = HuggingFaceHub(repo_id = "microsoft/Phi-3-mini-4k-instruct", huggingfacehub_api_token = API_KEY)

template = """ I am travelling to {location}. What are the top 3 things I can do while I am there. Be very specific and respond as three bullet points """

prompt = PromptTemplate(

input_variables=["location"],

template=template,

)

final_prompt = prompt.format(location=USER_INPUT )

print(f"LLM Output: {llm(final_prompt)}")