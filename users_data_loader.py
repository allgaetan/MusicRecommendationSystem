from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def assign_quantile_ratings(user_data):
        quantiles = user_data["play_count"].quantile([0.2, 0.4, 0.6, 0.8]).values
        user_data["rating"] = user_data["play_count"].apply(
            lambda x: 0.2 if x <= quantiles[0] else
                      0.4 if x <= quantiles[1] else
                      0.6 if x <= quantiles[2] else
                      0.8 if x <= quantiles[3] else 1.0
        )
        return user_data

def process_user_ratings(df):
    return df.groupby("user_id", group_keys=False).apply(assign_quantile_ratings)

def load_user_data(path):
    print("Loading users metadata...")
    df = pd.concat([chunk for chunk in tqdm(pd.read_csv(path, sep="\t", header=None, names=["user_id", "song_id", "play_count"], chunksize=1000))])
    print("Users metadata loaded.")

    return df

def get_field_info(field):
    field_mean, field_var = df[field].mean(), df[field].var()
    field_min, field_max = df[field].min(), df[field].max()
    print(f"Distribution de {field} : \n\tMoyenne : {field_mean} \n\tVariance : {field_var} \n\tMin : {field_min} \n\tMax : {field_max}")

    return field_mean, field_var, field_min, field_max

if __name__ == "__main__":
    path = "./user_subset.txt"

    df = load_user_data(path)
    print(df)
    get_field_info("play_count")

    """
    unique_users = df["user_id"].unique()
    unique_songs = df["song_id"].unique()
    print(f"Nombre d'utilisateurs dans le subset user : {len(unique_users)}")
    print(f"Nombre de chansons dans le subset user : {len(unique_songs)}")
    """
    
    df = process_user_ratings(df)
    print(df)
    get_field_info("rating")

    reader = Reader(rating_scale=(0.0, 1.0), sep="\t")
    data = Dataset.load_from_df(df[["user_id", "song_id", "rating"]], reader=reader)

    train_data, test_data = train_test_split(data, test_size=0.25)

    param_grid = {"n_epochs": [5, 10], "lr_all": [0.002, 0.005], "reg_all": [0.4, 0.6]}
    gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)

    model = SVD()
    model.fit(train_data)
    predictions = model.test(test_data)

    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    for pred in predictions[:10]:
        print(f"User: {pred.uid}, Item: {pred.iid}, Actual: {pred.r_ui}, Predicted: {pred.est:.2f}")

    actual_ratings = [pred.r_ui for pred in predictions]
    predicted_ratings = [pred.est for pred in predictions]

    plt.hist(actual_ratings, bins=10, alpha=0.5, label="Ratings Réels")
    plt.hist(predicted_ratings, bins=10, alpha=0.5, label="Ratings Prédits")
    plt.legend(loc="upper right")
    plt.xlabel("Ratings")
    plt.ylabel("Fréquence")
    plt.title("Distribution des Ratings Réels vs Prédits")
    plt.show()
        
    results = cross_validate(model, data, measures=["RMSE", "MAE"], cv=5, verbose=True)