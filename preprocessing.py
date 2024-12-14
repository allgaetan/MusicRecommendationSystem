def get_field_info(df, field, verbose=False):
    field_mean, field_var = df[field].mean(), df[field].var()
    field_min, field_max = df[field].min(), df[field].max()

    if verbose:
        print(f"Distribution de {field} : \n\tMoyenne : {field_mean} \n\tVariance : {field_var} \n\tMin : {field_min} \n\tMax : {field_max}")

    return field_mean, field_var, field_min, field_max

def assign_quantile_ratings(user_data, n_quantiles):
    quantiles = user_data["play_count"].quantile([i/n_quantiles for i in range(1, n_quantiles + 1)]).values
    user_data["rating"] = user_data["play_count"].apply(lambda x: sum(x > q for q in quantiles)/n_quantiles)
    return user_data

def normalize_user_playcounts(user_data):
    _, _, min, max = get_field_info(user_data, "play_count")
    if min < max:
        user_data["rating"] = user_data["play_count"].apply(lambda x: (x - min) / (max - min))
    else:
        user_data["rating"] = user_data["play_count"].apply(lambda x: 0.5)
    get_field_info(user_data, "rating")
    return user_data

def process_user_ratings(df, min_count=1, binary_rating=True, process=None, n_quantiles=None, n_listen=None):
    df = df.drop(df[df["play_count"] < min_count].index)

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


