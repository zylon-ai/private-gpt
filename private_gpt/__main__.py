# start a fastapi server with uvicorn
import os

import uvicorn

from private_gpt.main import app
from private_gpt.settings import settings

port = int(os.environ.get("PORT", settings.server.port))
uvicorn.run(app, host="0.0.0.0", port=port)
