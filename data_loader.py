from surprise import Dataset, Reader, SVD, accuracy, NormalPredictor
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV, KFold
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import zipfile
from sklearn.metrics import accuracy_score
from collections import defaultdict
import scipy.stats as stats


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

def assign_quantile_ratings(user_data, n_quantiles):
    quantiles = user_data["play_count"].quantile([i/n_quantiles for i in range(1, n_quantiles + 1)]).values
    user_data["rating"] = user_data["play_count"].apply(lambda x: sum(x > q for q in quantiles)/n_quantiles)
    return user_data

def normalize_user_playcounts(user_data):
    mean, var, min, max = get_field_info(user_data, "play_count")
    if min < max:
        user_data["rating"] = user_data["play_count"].apply(lambda x: (x - min) / (max - min))
    else:
        user_data["rating"] = user_data["play_count"].apply(lambda x: 0.5)
    get_field_info(user_data, "rating")
    return user_data

def process_user_ratings(df, min_count=1, process=None, n_quantiles=None):
    df = df.drop(df[df["play_count"] < min_count].index)
    #print(df)

    if process == "n_quantiles":
        if n_quantiles == None:
            raise ValueError(f"Value of n_quantiles must be passed with process={process}")
        else:
            df = df.groupby("user_id", group_keys=False).apply(assign_quantile_ratings, n_quantiles=n_quantiles)
    elif process == "normalize":
        df = df.groupby("user_id", group_keys=False).apply(normalize_user_playcounts)
    elif process == "listened_twice":
        df["rating"] = df["play_count"] >= 2
    elif process == None:
        raise ValueError(f"Preprocess needed to generate ratings from play counts")
    
    if binary_rating:
        mean, _, _, _ = get_field_info(df, "rating")
        df["rating"] = df["rating"].apply(lambda x: x > mean)
    
    return df


def load_user_data(path):
    print("Loading users metadata...")
    df = pd.concat([chunk for chunk in tqdm(pd.read_csv(path, sep="\t", header=None, names=["user_id", "song_id", "play_count"], chunksize=1000))])
    print("Users metadata loaded.")

    return df

def get_field_info(df, field, verbose=False):
    field_mean, field_var = df[field].mean(), df[field].var()
    field_min, field_max = df[field].min(), df[field].max()

    if verbose:
        print(f"Distribution de {field} : \n\tMoyenne : {field_mean} \n\tVariance : {field_var} \n\tMin : {field_min} \n\tMax : {field_max}")

    return field_mean, field_var, field_min, field_max

def eval(df, model, process, n_quantiles=None):
    if process == "n_quantiles":
        if n_quantiles == None:
            raise ValueError(f"Value of n_quantiles must be passed with process={process}")
        else:
            df = process_user_ratings(df, min_count=1, process=process, n_quantiles=n_quantiles)
    else:
        df = process_user_ratings(df, min_count=1, process=process)

    _, _, min, max = get_field_info(df, "rating")

    reader = Reader(rating_scale=(min, max), sep="\t")
    data = Dataset.load_from_df(df[["user_id", "song_id", "rating"]], reader=reader)

    train_data, test_data = train_test_split(data, test_size=0.25)

    model.fit(train_data)
    predictions = model.test(test_data)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    actual_ratings = [pred.r_ui for pred in predictions]
    predicted_ratings = [pred.est for pred in predictions]

    threshold = 0.5
    actual_ratings_bin = [1 if r > threshold else 0 for r in actual_ratings]
    predicted_ratings_bin = [1 if r > threshold else 0 for r in predicted_ratings]

    for idx, pred in enumerate(predictions[:10]):
        print(f"User: {pred.uid}, Item: {pred.iid}, Actual: {pred.r_ui}, Predicted: {pred.est:.2f}")
        print(f"\t{actual_ratings_bin[idx]}")
        print(f"\t{predicted_ratings_bin[idx]}")

    acc = accuracy_score(actual_ratings_bin, predicted_ratings_bin)

    results = {
        "accuracy": acc,
        "RMSE": rmse,
        "MAE": mae
    }

    return results

if __name__ == "__main__":
    #url = "http://labrosa.ee.columbia.edu/~dpwe/tmp/train_triplets.txt.zip"
    output_folder = "./data/TasteProfileSubset/"
    #os.makedirs(output_folder, exist_ok=True)
    #zip_path = download(url, output_folder)

    n_triplets = 10000
    extracted_file = os.path.join(output_folder, "train_triplets.txt")
    output_file = os.path.join(output_folder, f"train_{n_triplets}_triplets.txt")
    extract_triplets(extracted_file, output_file, n_triplets)

    df = load_user_data(output_file)
    print(df)
    get_field_info(df, "play_count", verbose=True)

    preprocesses = [
        "n_quantiles",
        "normalize",
        "listened_twice"]  
    N_quantiles = [5, 10, 25, 50, 100, 200]
    min_counts = [1, 2, 5, 10]
    threshold = [i/10 for i in range(1, 10)]

    params = {
        "preprocesses": preprocesses,
        "N_quantiles": N_quantiles,
        "min_counts": min_counts,
        "threshold": threshold
    }

    binary_rating=True
    
    results = {}
    
    for process in preprocesses:
        model = SVD()
        print(f"Preprocessing method used : {process}")
        if process == "n_quantiles":
            for n_quantiles in N_quantiles:
                print(f"\tNumber of quantiles : {n_quantiles}")
                results[f"{n_quantiles}_quantiles"] = eval(df, model, process, n_quantiles)
        else:
            results[process] = eval(df, model, process)

    results_df = pd.DataFrame(results)
    print(results_df)