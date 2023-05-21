import requests

url="http://localhost:5000"

prompt="I want you to act as a security engineer. Your task is to security review the code and find potential security bugs.\n    Your input would be a git diff, please only give suggestion on only the edited content. Consider the context for better suggestion.\n    Find and fix any bugs and typos. If no bug is found, just output \\\"No obvious bug found.\\\"\n    Do not include any personal opinions or subjective evaluations in your response.\n    Your output should looks like:\n\n      [\n        {\n        \\\"line\\\": 66,\n        \\\"finding\\\":\\\"...(your suggestion)\\\"\n        },\n        {\n        \\\"line\\\": 77,\n        \\\"finding\\\":\\\"...(your suggestion)\\\"\n        }\n\n     ]"
code="test"
jsondata={}
jsondata["question"] = prompt+"\n"+code

x = requests.post(url, json = jsondata)

print(x.text)
