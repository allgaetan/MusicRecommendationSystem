# MusicRecommendationSystem

Group members : Gaétan Allaire, Ramzi Khezzar, Romain Desmidt, François Duguay-Giguère, Logan Leconte

## Description

This project explores a subset of the Taste Profile dataset and implements a collaborative filtering system using the Surprise library. The system aims to predict user preferences for songs based on their listening history and the preferences of similar users.

## Installation

1. **Clone the repository:**

    `git clone https://github.com/allgaetan/MusicRecommendationSystem.git`

2. **Install the required libraries:**

    `pip install -r requirements.txt`

## Usage

Run the "main.py" Python script. The script will perform the following :

1. **Download the Taste Profile Subset:**
If the dataset is not already downloaded, the script will automatically download and extract it. The zip file is 500Mo in size so it may take some time to dowload and extract.

2. **Data Preprocessing:**
One of the main challenge is to use the play counts given in the dataset and generate ratings for each user, since the prediction is made on a rating value rather than a listening count.
The project includes various data preprocessing techniques to generate user ratings, including:
    - Quantile-based ratings
    - Normalized play counts
    - Binary ratings based on listening frequency

3. **Collaborative Filtering:**
The project uses the SVD algorithm from the Surprise library to train a collaborative filtering model.

4. **Evaluation:**
The model is evaluated using sklearn metrics : precision, recall, and accuracy.

## Project Structure

- `preprocessing.py`: Includes functions for data preprocessing and rating generation.
- `eval.py`: Contains functions for evaluating the collaborative filtering model.
- `main.py`: The main script to run the project.
- `README.md`: This file.
- `requirements.txt`: Lists the required Python libraries.

