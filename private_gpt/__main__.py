# start a fastapi server with uvicorn

import uvicorn

from private_gpt.main import app
from private_gpt.settings.settings import settings

uvicorn.run(app, host="0.0.0.0", port=settings.server.port)
