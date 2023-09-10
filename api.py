from flask import Flask, request, jsonify
from private_gpt_api import PrivateGPTQueryInterface, Config

app = Flask(__name__)
conf = Config()
private_gpt_query_interface = PrivateGPTQueryInterface(conf)


@app.route('/ask', methods=['POST'])
def ask():
    query = request.json.get('query', '')
    response = private_gpt_query_interface.get_answer(query)
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
