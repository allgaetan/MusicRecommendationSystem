from data_loader import download, extract_triplets, load_user_data
from preprocessing import get_field_info, assign_quantile_ratings, normalize_user_playcounts, process_user_ratings

from surprise import Dataset, Reader, accuracy, SVD, KNNBasic
from surprise.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
import pandas as pd
import os

def eval(df, model, process, threshold=0.5, n_quantiles=None):
    """
    Evaluate a model on a given DataFrame using a given preprocessing method.

    Arguments:
    - df (pandas.DataFrame): The DataFrame to evaluate the model on.
    - model (surprise.model_base.AlgoBase): The model to evaluate.
    - process (str): The type of preprocessing method to generate ratings :
        - "n_quantiles" : assign the ratings based on quantiles. When this process is used, parameter n_quantiles needs to be input with an int
        - "normalize" : rating based on a normalization of the user play counts
        - "listened_twice" : naive boolean approach consisting of considering a song listened at least twice as "good rating" (1) and 0 else
    - threshold (float): The decision threshold (default 0.5)
    - n_quantiles (int): The number of quantiles to use for rating.
    """
    # Data preprocessing (generating ratings)
    if process == "n_quantiles":
        if n_quantiles == None:
            raise ValueError(f"Value of n_quantiles must be passed with process={process}")
        else:
            df = process_user_ratings(df, min_count=1, process=process, n_quantiles=n_quantiles)
    else:
        df = process_user_ratings(df, min_count=1, process=process)

    # Read the data 
    _, _, min, max = get_field_info(df, "rating", verbose=False)
    reader = Reader(rating_scale=(min, max), sep="\t")
    data = Dataset.load_from_df(df[["user_id", "song_id", "rating"]], reader=reader)

    # Train test split
    train_data, test_data = train_test_split(data, test_size=0.25)

    # Model training
    model.fit(train_data)

    # Model predictions
    predictions = model.test(test_data)

    # Converting predicted ratings to binary recommendations
    actual_ratings = [pred.r_ui for pred in predictions]
    predicted_ratings = [pred.est for pred in predictions]
    actual_ratings_bin = [1 if r > threshold else 0 for r in actual_ratings]
    predicted_ratings_bin = [1 if r > threshold else 0 for r in predicted_ratings]

    # Compute sklearn metrics
    precision = precision_score(actual_ratings_bin, predicted_ratings_bin)
    recall = recall_score(actual_ratings_bin, predicted_ratings_bin)
    accuracy = accuracy_score(actual_ratings_bin, predicted_ratings_bin)
    results = {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy
    }

    return results

def eval_preprocess(df, preprocesses, N_quantiles):
    """
    Evaluate a model on a given DataFrame using different preprocessing methods.

    Arguments:
    - df (pandas.DataFrame): The DataFrame to evaluate the model on.
    - preprocesses (list): The list of preprocessing methods to evaluate.
    - N_quantiles (list): The list of numbers of quantiles to use for rating when "n_quantiles" is used.
    """
    results = {}
    for process in preprocesses:
        print(f"\tEvaluation using {process} preprocessing")
        model = SVD()
        if process == "n_quantiles":
            for n_quantiles in N_quantiles:
                print(f"\t\tNumber of quantiles : {n_quantiles}")
                results[f"{n_quantiles}_quantiles"] = eval(df, model, process, n_quantiles=n_quantiles)
        else:
            results[process] = eval(df, model, process)
    
    return results

def eval_n_triplets(path, N_triplets):
    """
    Evaluate a model on a given DataFrame using different numbers of triplets of the whole dataset.

    Arguments:
    - path (str): The path to the folder containing the extracted triplets.
    - N_triplets (list): The list of numbers of triplets to evaluate.
    """
    results = {}
    for n_triplets in N_triplets:
        print(f"\tEvaluation with {n_triplets} triplets")
        extracted_file = os.path.join(path, "train_triplets.txt")
        output_file = os.path.join(path, f"train_{n_triplets}_triplets.txt")
        extract_triplets(extracted_file, output_file, n_triplets)
        df = load_user_data(output_file)
        model = SVD()
        results[n_triplets] = eval(df, model, "normalize")

    return results

def eval_threshold(df, thresholds):
    """
    Evaluate a model on a given DataFrame using different decision thresholds.

    Arguments:
    - df (pandas.DataFrame): The DataFrame to evaluate the model on.
    - thresholds (list): The list of decision thresholds to evaluate.
    """
    results = {}
    for threshold in thresholds:
        print(f"\tEvaluation with a decision threshold of {threshold}")
        model = SVD()
        results[threshold] =  eval(df, model, "normalize", threshold=threshold)

    return results