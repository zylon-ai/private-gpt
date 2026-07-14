import sys

from private_gpt.worker import run_private_gpt_worker

if __name__ == "__main__":
    run_private_gpt_worker(sys.argv[1:])
