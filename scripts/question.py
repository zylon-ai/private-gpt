import sys
from gradio_client import Client

client = Client("http://localhost:8001/")
result = client.predict(
		sys.argv[1],	# str  in 'Message' Textbox component
		"Query Files",	# Literal['Query Files', 'Search Files', 'LLM Chat (no context from files)']  in 'Mode' Radio component
		[],	# List[filepath]  in 'Upload File(s)' Uploadbutton component
		"You can only answer questions about the provided context. If you know the answer but it is not based in the provided context, don't provide the answer, just state the answer is not in the context provided.",	# str  in 'System Prompt' Textbox component
		api_name="/chat"
)
print(result)
