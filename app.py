import os
from privateGPT import get_answer
from ingest import train_model
from flask import Flask,request, jsonify,send_from_directory
from flask_cors import CORS

relative_path = '../client/build'
absolute_path = os.path.join(os.path.dirname(__file__), relative_path)
app = Flask(__name__,static_folder=absolute_path )

config = {
    "DEBUG": True  # run app in debug mode
}
app.config.from_mapping(config)
CORS(app)

@app.route('/api/answer', methods=['POST'])
def answer():
    question = request.json.get('question')
    print(question)
    answer = get_answer(query=question)
    return jsonify({'answer': answer})

@app.route('/api/train', methods=['POST'])
def train():
    result = train_model()
    return jsonify({"message":"Completed Successfully!"})
  
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == "__main__":
  app.run(port=5000)