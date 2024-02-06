# start a fastapi server with uvicorn

import uvicorn

from private_gpt.main import app
from private_gpt.settings.settings import settings
from fastapi.staticfiles import StaticFiles

# Set log_config=None to do not use the uvicorn logging configuration, and
# use ours instead. For reference, see below:
# https://github.com/tiangolo/fastapi/discussions/7457#discussioncomment-5141108
app.mount("/static", StaticFiles(directory=r"C:\Users\Dbuser\QuickGPT\backend\privateGPT\private_gpt\static"), name="static")

uvicorn.run(app, host="0.0.0.0", port=settings().server.port, log_config=None)
