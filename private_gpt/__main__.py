# start a fastapi server with uvicorn

import uvicorn
import argparse
from private_gpt.main import app
from private_gpt.settings.settings import settings

def start_server(use_https):
    if use_https:
        # Set the paths for SSL key and certificate
        ssl_keyfile_path = "/etc/ssl/privateGPT/private.key"
        ssl_certfile_path = "/etc/ssl/privateGPT/certificate.crt"
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=settings().server.port,
            ssl_keyfile=ssl_keyfile_path,
            ssl_certfile=ssl_certfile_path,
            log_config=None,
        )
    else:
        # Set log_config=None to do not use the uvicorn logging configuration, and
        # use ours instead. For reference, see below:
        # https://github.com/tiangolo/fastapi/discussions/7457#discussioncomment-5141108
        uvicorn.run(app, host="0.0.0.0", port=settings().server.port, log_config=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the FastAPI server.")
    parser.add_argument(
        "--https",
        dest="use_https",
        action="store_true",
        help="Use HTTPS for the server (default is HTTP)",
    )

    args = parser.parse_args()
    start_server(args.use_https)
