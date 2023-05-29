export LLAMA_CUBLAS=1
#check if venv exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt