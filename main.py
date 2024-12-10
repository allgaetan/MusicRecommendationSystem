from data_loader import *
import pandas as pd

### MAIN ###
if __name__ == "__main__":
    millionsongsubset_path = "./data/MillionSongSubset"
    tasteprofilesubset_path = "./data/TasteProfileSubset/train_triplets.txt"

    print("Loading songs metadata...")
    h5_files = get_all_files(millionsongsubset_path)
    metadata_songs = [get_all_metadata(file) for file in h5_files] 
    metadata_songs = pd.DataFrame(metadata_songs)
    print("Songs metadata loaded.")

    print("Loading users metadata...")
    metadata_users = pd.read_csv(tasteprofilesubset_path, sep="\t", header=None, names=["user_id", "song_id", "play_count"])
    print("Users metadata loaded.")

    unique_users = metadata_users["user_id"].unique()
    unique_songs = metadata_users["song_id"].unique()
    print(f"Nombre d'utilisateurs : {len(unique_users)}")
    print(f"Nombre de chansons : {len(unique_songs)}")

    # Overflow : matrice de plus de 4e11 elmts (1e6 users x 4e5 songs)
    #user_song_matrix = metadata_users.pivot(index="user_id", columns="song_id", values="play_count").fillna(0)
    #print(user_song_matrix)