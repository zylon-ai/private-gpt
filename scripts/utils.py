import argparse
import os
import shutil


def wipe():
    path = "local_data"
    print(f"Wiping {path}...")
    all_files = os.listdir(path)

    files_to_remove = [file for file in all_files if file != ".gitignore"]
    for file_name in files_to_remove:
        file_path = os.path.join(path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            print(f" - Deleted {file_path}")
        except PermissionError:
            print(
                f"PermissionError: Unable to remove {file_path}. It is in use by another process."
            )
            continue


if __name__ == "__main__":
    commands = {
        "wipe": wipe,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", help="select a mode to run", choices=list(commands.keys())
    )
    args = parser.parse_args()
    commands[args.mode.lower()]()
