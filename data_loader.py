import pandas as pd
from tqdm import tqdm
import requests
import os
import zipfile

def download(url, output_folder):
    """
    Download a file from a given URL and save it to the specified output folder.

    Arguments:
    - url (str): The URL of the file to download.
    - output_folder (str): The path to the folder where the downloaded file will be saved.

    Returns:
    - str: The path to the downloaded file.
    """
    print(f"Dowloading {url} to {output_folder}...")
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
    """
    Extract the first n triplets from the input file and write them to the output file.

    Arguments:
    - input_file (str): The path to the input file.
    - output_file (str): The path to the output file.
    - n (int): The number of triplets to extract.
    """
    print(f"Writing first {n} triplets of {input_file} to {output_file}...")
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for i, line in enumerate(infile):
            if i > n:
                break
            outfile.write(line)
    print(f"Done.")

def load_user_data(path):
    """
    Load user data from a given path.

    Arguments:
    - path (str): The path to the user data file.

    Returns:
    - pandas.DataFrame: A DataFrame containing the user data.
    """
    print("Loading users metadata...")
    df = pd.concat([chunk for chunk in tqdm(pd.read_csv(path, sep="\t", header=None, names=["user_id", "song_id", "play_count"], chunksize=1000))])
    print("Users metadata loaded.")

    return df