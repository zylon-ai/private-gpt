# start a fastapi server with uvicorn
import os

import uvicorn

from private_gpt.main import app

port = int(os.environ.get("PORT", 8001))
uvicorn.run(app, host="0.0.0.0", port=port)
