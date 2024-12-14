from data_loader import download, extract_triplets, load_user_data
from preprocessing import get_field_info, assign_quantile_ratings, normalize_user_playcounts, process_user_ratings
from eval import eval

import os
from surprise import SVD
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    data_folder = "./data/TasteProfileSubset/"

    # Download the data if not already dowloaded
    alreadyDowloaded = True
    if not alreadyDowloaded:
        url = "http://labrosa.ee.columbia.edu/~dpwe/tmp/train_triplets.txt.zip"
        os.makedirs(data_folder, exist_ok=True)
        zip_path = download(url, data_folder)

    # Parameters 
    N_triplets = [100, 500, 1000, 5000, 10000, 50000, 100000]
    preprocesses = [
        "n_quantiles",
        "normalize",
        "listened_twice"]  
    N_quantiles = [5, 10, 50, 100]
    min_counts = [1, 2, 5, 10]
    threshold = [i/10 for i in range(1, 10)]

    params = {
        "N_triplets": N_triplets,
        "preprocesses": preprocesses,
        "N_quantiles": N_quantiles,
        "min_counts": min_counts,
        "threshold": threshold
    }
    """
    # Evaluation by sample size
    sample_size_eval = {}
    for n_triplets in N_triplets:
        print(f"Evaluation with {n_triplets} triplets")

        # Extract n triplets from the data in a file
        extracted_file = os.path.join(data_folder, "train_triplets.txt")
        output_file = os.path.join(data_folder, f"train_{n_triplets}_triplets.txt")
        extract_triplets(extracted_file, output_file, n_triplets)

        # Load users data from the file
        df = load_user_data(output_file)
        print(df)
        get_field_info(df, "play_count", verbose=True)
        
        # Evaluation by type of preprocessing
        results = {}
        for process in preprocesses:
            model = SVD()
            print(f"Preprocessing method used : {process}")
            if process == "n_quantiles":
                for n_quantiles in N_quantiles:
                    print(f"\tNumber of quantiles : {n_quantiles}")
                    results[f"{n_quantiles}_quantiles"] = eval(df, model, process, n_quantiles=n_quantiles)
            else:
                results[process] = eval(df, model, process)

        results_df = pd.DataFrame(results)
        print(results_df)

        best_acc = 0
        best_process = None
        for process, stats in results.items():
            if stats["accuracy"] >= best_acc:
                best_acc = stats["accuracy"]
                best_process = process
        
        sample_size_eval[n_triplets] = [best_acc, best_process]

    print(sample_size_eval)
    """
    sample_size_eval = {100: [0.8, '100_quantiles'], 500: [0.76, 'normalize'], 1000: [0.752, 'normalize'], 5000: [0.7088, 'normalize'], 10000: [0.7488, 'normalize'], 50000: [0.76552, 'normalize'], 100000: [0.75008, 'normalize']}
    best_acc = []
    for (key, value) in sample_size_eval.items():
        best_acc.append(value[0])

    plt.figure()
    plt.plot(np.log10(N_triplets), best_acc)
    for (key, value) in sample_size_eval.items():
        plt.scatter(np.log10(key), value[0], label=f"{value[1]} preprocessing")
    plt.legend()
    plt.show()