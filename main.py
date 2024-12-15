from data_loader import download, extract_triplets, load_user_data
from preprocessing import get_field_info, assign_quantile_ratings, normalize_user_playcounts, process_user_ratings
from eval import eval, eval_preprocess, eval_n_triplets, eval_threshold

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    ###################################
    # DATA IMPORT
    ###################################
    data_folder = "./data/TasteProfileSubset/"

    # Download the data if not already dowloaded
    alreadyDowloaded = True # Change to false when running for the first time
    if not alreadyDowloaded:
        url = "http://labrosa.ee.columbia.edu/~dpwe/tmp/train_triplets.txt.zip"
        os.makedirs(data_folder, exist_ok=True)
        zip_path = download(url, data_folder)

    # Parameters 
    N_triplets = [100, 300, 500, 1000, 3000, 5000, 10000, 30000, 50000, 100000, 300000, 500000, 1000000]
    preprocesses = [
        "n_quantiles",
        "normalize",
        "listened_twice"]  
    N_quantiles = [5, 10, 50, 100]
    thresholds = [i/10 for i in range(1, 10)]

    params = {
        "N_triplets": N_triplets,
        "preprocesses": preprocesses,
        "N_quantiles": N_quantiles,
        "threshold": thresholds
    }

    # Extract n triplets from the data in a file
    n_triplets = 10000
    extracted_file = os.path.join(data_folder, "train_triplets.txt")
    output_file = os.path.join(data_folder, f"train_{n_triplets}_triplets.txt")
    extract_triplets(extracted_file, output_file, n_triplets)

    # Load users data from the file
    df = load_user_data(output_file)
    print(df)
    get_field_info(df, "play_count", verbose=True)
    

    ###################################
    # EVALUATION
    ###################################
    evaluate_preprocess = True
    evaluate_thresholds = True
    evaluate_n_triplets = True
    show_plots = True

    # Evaluation by type of preprocessing
    if evaluate_preprocess:
        print(f"Evaluation by type of preprocessing :")
        preprocesses_evaluation = eval_preprocess(df, preprocesses, N_quantiles)
        preprocesses_evaluation = pd.DataFrame(preprocesses_evaluation)
        print(preprocesses_evaluation)

    # Evaluation by decision threshold
    if evaluate_thresholds:
        print(f"Evaluation by decision threshold :")
        thresholds_evaluation = eval_threshold(df, thresholds)
        precision = []
        recall = []
        accuracy = []
        for key, value in thresholds_evaluation.items():
            precision.append(value["precision"])
            recall.append(value["recall"])
            accuracy.append(value["accuracy"])

        thresholds_evaluation = pd.DataFrame(thresholds_evaluation)
        print(thresholds_evaluation)

        if show_plots:
            plt.figure()
            plt.title("Scores by decision threshold")
            plt.plot(thresholds, precision, label="Precision")
            plt.plot(thresholds, recall, label="Recall")
            plt.plot(thresholds, accuracy, label="Accuracy")
            plt.xlabel("Decision threshold")
            plt.ylabel("Score")
            plt.legend()
            plt.show()

    # Evaluation by number of triplets used
    if evaluate_n_triplets:
        print(f"Evaluation by number of triplets used :")
        n_triplets_evaluation = eval_n_triplets(data_folder, N_triplets)
        precision = []
        recall = []
        accuracy = []
        for key, value in n_triplets_evaluation.items():
            precision.append(value["precision"])
            recall.append(value["recall"])
            accuracy.append(value["accuracy"])

        n_triplets_evaluation = pd.DataFrame(n_triplets_evaluation)
        print(n_triplets_evaluation)

        if show_plots:
            plt.figure()
            plt.title("Scores by number of triplets used")
            plt.plot(np.log10(N_triplets), precision, label="Precision")
            plt.plot(np.log10(N_triplets), recall, label="Recall")
            plt.plot(np.log10(N_triplets), accuracy, label="Accuracy")
            plt.xlabel("log(n_triplets)")
            plt.ylabel("Score")
            plt.legend()
            plt.show()