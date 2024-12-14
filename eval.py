from data_loader import download, extract_triplets, load_user_data
from preprocessing import get_field_info, assign_quantile_ratings, normalize_user_playcounts, process_user_ratings

from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

def eval(df, model, process, threshold=0.5, n_quantiles=None):
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