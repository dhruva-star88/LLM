from langchain.llms.huggingface_hub import HuggingFaceHub
API_KEY ="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"

from langchain.llms.huggingface_hub import HuggingFaceHub
llm = HuggingFaceHub(repo_id = "microsoft/Phi-3-mini-4k-instruct", huggingfacehub_api_token = API_KEY)

print(llm("what is datascientist?"))