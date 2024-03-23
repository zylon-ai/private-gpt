from gradio_client import Client

with open('question.txt', 'r') as file:
    # Read the content of the file
    content = file.read()

with open('system.txt', 'r') as file:
    # Read the content of the file
    systemcontent = file.read()

client = Client("http://localhost:8001/")
result = client.predict(
		content,	# str  in 'Message' Textbox component
		"Query Files",	# Literal['Query Files', 'Search Files', 'LLM Chat (no context from files)']  in 'Mode' Radio component
		[],	# List[filepath]  in 'Upload File(s)' Uploadbutton component
		systemcontent,	# str  in 'System Prompt' Textbox component
		api_name="/chat"
)
print(result)
