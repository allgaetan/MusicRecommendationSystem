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
    return user_data

def process_user_ratings(df, process):
    df = df.drop(df[df["play_count"] < 2].index)
    print(df)

    if process == "percentiles":
        return df.groupby("user_id", group_keys=False).apply(assign_quantile_ratings, n_quantiles=100)
    if process == "deciles":
        return df.groupby("user_id", group_keys=False).apply(assign_quantile_ratings, n_quantiles=10)
    if process == "normalize":
        mean, var, min, max = get_field_info(df, "play_count")
        if min < max:
            df["rating"] = df["play_count"].apply(lambda x: (x - min) / (max - min))
        else:
            df["rating"] = df["play_count"].apply(lambda x: 0.5)
        return df
    if process == "normalize_per_user":
        return df.groupby("user_id", group_keys=False).apply(normalize_user_playcounts)
    if process == "listened_twice":
        df["rating"] = df["play_count"] >= 2
        return df
    if process == 0:
        df["rating"] = df["play_count"] 
        return df

def load_user_data(path):
    print("Loading users metadata...")
    df = pd.concat([chunk for chunk in tqdm(pd.read_csv(path, sep="\t", header=None, names=["user_id", "song_id", "play_count"], chunksize=1000))])
    print("Users metadata loaded.")

    return df

def get_field_info(df, field):
    field_mean, field_var = df[field].mean(), df[field].var()
    field_min, field_max = df[field].min(), df[field].max()
    print(f"Distribution de {field} : \n\tMoyenne : {field_mean} \n\tVariance : {field_var} \n\tMin : {field_min} \n\tMax : {field_max}")

    return field_mean, field_var, field_min, field_max

def precision_recall_at_k(predictions, k, threshold):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls

def precision_recall(model):
    kf = KFold(n_splits=5)
    for trainset, testset in kf.split(data):
        model.fit(trainset)
        predictions = model.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=2)
        print(sum(prec for prec in precisions.values()) / len(precisions))
        print(sum(rec for rec in recalls.values()) / len(recalls))


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
    get_field_info(df, "play_count")

    df = process_user_ratings(df, process="percentiles")
    print(df)
    _, _, min, max = get_field_info(df, "rating")

    reader = Reader(rating_scale=(min, max), sep="\t")
    data = Dataset.load_from_df(df[["user_id", "song_id", "rating"]], reader=reader)

    train_data, test_data = train_test_split(data, test_size=0.25)

    model = SVD()
    model.fit(train_data)
    predictions = model.test(test_data)

    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    for pred in predictions[:10]:
        print(f"User: {pred.uid}, Item: {pred.iid}, Actual: {pred.r_ui}, Predicted: {pred.est:.2f}")

    actual_ratings = [pred.r_ui for pred in predictions]
    predicted_ratings = [pred.est for pred in predictions]
    actual_ratings_int = [int(r) for r in actual_ratings]
    predicted_ratings_int = [int(r) for r in predicted_ratings]

    binary_accuracy = accuracy_score(actual_ratings_int, predicted_ratings_int)
    print(f"Binary Classification Accuracy: {binary_accuracy * 100:.2f}%")

    plt.hist(actual_ratings, bins=100, alpha=0.5, label="Ratings réels", density=True)
    plt.hist(predicted_ratings, bins=100, alpha=0.5, label="Ratings prédits", density=True)
    plt.hist(df["rating"], bins=100, alpha=0.5, label="Ratings dataset", density=True)
    plt.legend(loc="upper right")
    plt.xlabel("Ratings")
    plt.ylabel("Fréquence")
    plt.title("Distribution des Ratings Réels vs Prédits")
    plt.show()
        
    #results = cross_validate(model, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

    