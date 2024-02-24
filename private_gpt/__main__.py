# start a fastapi server with uvicorn

from datetime import datetime
from fastapi.middleware import Middleware
from private_gpt.users.db.session import SessionLocal
from private_gpt.users.models import Audit, User, Department, Document
from private_gpt.users.api.deps import get_audit_logger, get_db
import uvicorn

from private_gpt.main import app
from private_gpt.settings.settings import settings
from fastapi.staticfiles import StaticFiles
from private_gpt.constants import UPLOAD_DIR

# Set log_config=None to do not use the uvicorn logging configuration, and
# use ours instead. For reference, see below:
# https://github.com/tiangolo/fastapi/discussions/7457#discussioncomment-5141108

app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

uvicorn.run(app, host="0.0.0.0", port=settings().server.port, log_config=None)
