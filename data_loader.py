import pandas as pd
from tqdm import tqdm
import requests
import os
import zipfile

def download(url, output_folder):
    print(f"Dowloading {zip_filename}...")
    response = requests.get(url)
    zip_filename = url.split("/")[-1]
    zip_path = os.path.join(output_folder, zip_filename)

    with open(zip_path, "wb") as file:
        file.write(response.content)

    print("Extracting the zip file...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_folder)

    print("Done.")

    return zip_path

def extract_triplets(input_file, output_file, n):
    print(f"Writing first {n} triplets of {input_file} to {output_file}...")
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for i, line in enumerate(infile):
            if i > n:
                break
            outfile.write(line)
    print(f"Done.")

def load_user_data(path):
    print("Loading users metadata...")
    df = pd.concat([chunk for chunk in tqdm(pd.read_csv(path, sep="\t", header=None, names=["user_id", "song_id", "play_count"], chunksize=1000))])
    print("Users metadata loaded.")

    return df




