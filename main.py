from data_loader import *
import pandas as pd
from tqdm import tqdm
import hdf5_getters

### MAIN ###
if __name__ == "__main__":
    #millionsongsubset_path = "./data/MillionSongSubset"
    #msd_summary_path = "./data/msd_summary_file.h5"
    #tasteprofilesubset_path = "./data/TasteProfileSubset/train_triplets.txt"
    tasteprofilesubset_path = "./data/TasteProfileSubset/user_subset.txt"

    print("Loading users metadata...")
    metadata_users = pd.concat([chunk for chunk in tqdm(pd.read_csv(tasteprofilesubset_path, sep="\t", header=None, names=["user_id", "song_id", "play_count"], chunksize=1000), desc='Loading data')])
    print(metadata_users)
    print("Users metadata loaded.")

    song_ids = set(metadata_users["song_id"])
    print(f"Nombre de chansons dans le subset : {len(song_ids)}")

    unique_users = metadata_users["user_id"].unique()
    unique_songs = metadata_users["song_id"].unique()
    print(f"Nombre d'utilisateurs dans le subset user : {len(unique_users)}")
    print(f"Nombre de chansons dans le subset user : {len(unique_songs)}")

