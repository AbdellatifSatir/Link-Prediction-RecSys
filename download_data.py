import urllib.request
import zipfile
import os
import shutil

url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
zip_path = "ml-100k.zip"
extract_path = "."
final_path = "data"

print(f"Downloading {url}...")
urllib.request.urlretrieve(url, zip_path)

print("Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Cleanup and rename
if os.path.exists(final_path):
    shutil.rmtree(final_path)
os.rename("ml-100k", final_path)
os.remove(zip_path)

print(f"Dataset ready in '{final_path}/' directory.")
