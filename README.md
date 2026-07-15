# Amharic Idiom Classification

This repository contains an Amharic idiom classification project that combines a Flask-based web dashboard with text preprocessing, model training, and model evaluation.

## Project Overview

- A Flask application for browsing an Amharic idiom dataset, filtering idioms, and predicting idiom sentiment labels.
- A `sklearn` pipeline model using `TfidfVectorizer` and `LogisticRegression` saved as `classifier_model.joblib`.
- A labeled Amharic idiom dataset in `labeled_idiom.csv` and a results summary in `model_results.json`.
- Experimental Jupyter notebooks for idiom classification without and with word embedding approaches.

## Current Status

- `app.py` is a working Flask web app with:
  - `/login` authentication page (dummy credentials: `admin` / `admin123`)
  - `/dashboard` dataset summary and idiom browsing UI
  - `/predict` prediction endpoint for new idiom text
  - `/download` CSV export of the current filtered idiom view
- `classifier_model.joblib` is built from `labeled_idiom.csv`.
- `model_results.json` contains the current evaluation metrics:
  - Accuracy: `0.92`
  - Precision: `0.90`
  - Recall: `0.88`
  - F1 score: `0.89`

## Repository Contents

- `app.py` - Flask application implementing login, dashboard, prediction, filtering, and download.
- `classifier_model.joblib` - saved trained model used by the app.
- `labeled_idiom.csv` - labeled Amharic idiom dataset used for training and dashboard display.
- `model_results.json` - JSON file with evaluation metrics.
- `README.md` - project documentation.
- `tests/test_app.py` - unit tests for the Flask app routes and basic functionality.
- `Amharic_idiom_dictionary.txt` - dictionary resource file for Amharic idiom preprocessing.
- `idiom_and_meaning.csv` and `Amharic_idiom_preprocessing.ipynb` - dataset and preprocessing notebook resources.
- `idiom_classification_without word embedding.ipynb` - notebook exploring classification without pretrained embeddings.
- `idiom_classification_using word embedding.ipynb` - notebook exploring classification with word embeddings.

## How to Run

1. Install dependencies:
   ```bash
   pip install flask scikit-learn joblib
   ```
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Open the browser at `http://127.0.0.1:5000/login`.
4. Use credentials: `admin` / `admin123`.

## Notes and Next Steps

- The current classification model is a simple TF-IDF + Logistic Regression pipeline.
- The web app is functional, but it uses inline HTML templates and basic login logic.
- The notebooks show research experiments on neural network approaches and one-hot encoding.
- Future improvements could include:
  - adding a proper registration/login system
  - supporting Amharic-specific preprocessing and spell correction in the Flask app
  - integrating the neural models from the notebooks into the dashboard
  - cleaning and expanding the dataset for better coverage and model generalization
