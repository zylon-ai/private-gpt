import os
import hashlib

def calculate_hash(file_path):
    """Calculate the SHA256 hash of a file."""
    BUF_SIZE = 65536
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

def remove_duplicates(directory):
    """Remove duplicate files in a directory."""
    file_hashes = {}
    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_hash = calculate_hash(file_path)
            if file_hash in file_hashes:
                print(f"Removing duplicate file: {file_path}")
                os.remove(file_path)
            else:
                file_hashes[file_hash] = file_path

# Replace with your directory 
directory = "./source_documents"
remove_duplicates(directory)