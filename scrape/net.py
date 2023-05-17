import os
import subprocess
import datetime

# Ask user for search terms
print("Enter your search terms. This is the query to make on Google Scholar.")
search_terms = input("Search terms: ")

# Set other parameters
pages_to_download = "10"
min_year = "2018"
scihub_mirror = "https://sci-hub.do"

# Generate timestamped directory
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = os.path.join("C:\\User\\example\\papers", timestamp)

# Check if the directory exists, if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Log the search terms
with open(os.path.join(output_dir, 'search_terms.txt'), 'w') as f:
    f.write(search_terms)

# Construct the command
command = f'python3 -m PyPaperBot --query="{search_terms}" --scholar-pages={pages_to_download}  --min-year={min_year} --dwn-dir="{output_dir}" --scihub-mirror="{scihub_mirror}"'

# Execute the command
subprocess.run(command, shell=True)
