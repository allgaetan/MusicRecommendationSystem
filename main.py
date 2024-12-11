import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def precision_recall_at_k(predicted_ratings, test_matrix, k=10):
    precisions, recalls = [], []

    for user in range(predicted_ratings.shape[0]):
        top_k_items = np.argsort(-predicted_ratings[user, :])[:k]
        
        relevant_items = test_matrix[user, :].nonzero()[1]
        
        hits = len(set(top_k_items) & set(relevant_items))
        precisions.append(hits / k if k > 0 else 0)
        recalls.append(hits / len(relevant_items) if len(relevant_items) > 0 else 0)

    return np.mean(precisions), np.mean(recalls)

### MAIN ###
if __name__ == "__main__":
    tasteprofilesubset_path = "./data/TasteProfileSubset/train_triplets.txt"
    #tasteprofilesubset_path = "./data/TasteProfileSubset/user_subset.txt"
    #tasteprofilesubset_path = "./data/TasteProfileSubset/user_microset.txt"

    print("Loading users metadata...")
    metadata_users = pd.concat([chunk for chunk in tqdm(pd.read_csv(tasteprofilesubset_path, sep="\t", header=None, names=["user_id", "song_id", "play_count"], chunksize=1000))])
    print("Users metadata loaded.")

    unique_users = metadata_users["user_id"].unique()
    unique_songs = metadata_users["song_id"].unique()
    print(f"Nombre d'utilisateurs dans le subset user : {len(unique_users)}")
    print(f"Nombre de chansons dans le subset user : {len(unique_songs)}")

    print("Mapping data with user index and song index...")
    user_mapping = {user_id: index for index, user_id in enumerate(unique_users)}
    song_mapping = {song_id: index for index, song_id in enumerate(unique_songs)}

    metadata_users["user_index"] = metadata_users["user_id"].map(user_mapping)
    metadata_users["song_index"] = metadata_users["song_id"].map(song_mapping)
    print("Data mapped.")


    sparse_matrix = csr_matrix(
        (metadata_users["play_count"], 
        (metadata_users["user_index"], metadata_users["song_index"])),
        shape=(len(unique_users), len(unique_songs)),
        dtype=np.float64
    )

    print(f"Sparse matrix size : {sparse_matrix.size} = {100*sparse_matrix.size/(len(unique_users)*len(unique_songs)):.2f} % of dense matrix size.")
    

