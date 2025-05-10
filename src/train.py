from utils.dataset_utils import convert_numpy_types, load_from_config
import os
import yaml
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tqdm import tqdm
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import logging
import time  # Import the time module

# Load constants from config.yaml
cfg = load_from_config()

RANDOM_STATE = cfg.get("random_state", 42)
TEST_SIZE = cfg.get("test_size", 0.2)
VALIDATION_SIZE = cfg.get("validation_size", 0.25)
N_JOBS = cfg.get("n_jobs", -1)
OUTPUT_DIR = cfg["output_dir"]

# Define target and categorical columns
TARGET = cfg["TARGET"]
CATEGORICAL = cfg["CATEGORICAL"]
IGNORE = cfg["IGNORE"]


def create_run_folder(output_dir):
    """Create a unique folder for the current training run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(output_dir, f"model_results_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    return run_folder


def save_metrics(metrics, run_folder):
    """Save metrics to a YAML file in the run folder."""
    metrics_path = os.path.join(run_folder, "metrics.yaml")
    with open(metrics_path, "w") as f:
        yaml.safe_dump(metrics, f)
    print(f"Metrics saved to {metrics_path}")


def save_model(model, model_name, run_folder):
    """Save a model to the run folder."""
    model_path = os.path.join(run_folder, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model {model_name} saved to {model_path}")
    return model_path


def save_encoder_path_to_config(encoders_path):
    """Save the path to the encoders in config.yaml."""
    with open("options/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["encoders_path"] = encoders_path

    with open("options/config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    print(f"Best model path saved to config.yaml: {encoders_path}")


def preprocess_data(run_folder):
    """Preprocess the data and return X_train, y_train."""
    unique_sets = {cat: set() for cat in CATEGORICAL}

    # Load previous batches and collect unique categories
    for batch_path in tqdm(os.listdir(cfg["batch_storage"]), desc="Scanning unique categories..."):
        batch = pd.read_csv(os.path.join(cfg["batch_storage"], batch_path))
        for cat in CATEGORICAL:
            arr = batch[cat].apply(ast.literal_eval)
            for item in arr:
                unique_sets[cat].update(item)

    # Initialize encoders
    encoders = {cat: MultiLabelBinarizer(classes=sorted(
        list(unique_sets[cat]))) for cat in CATEGORICAL}

    # Save encoders to a file for future use
    for cat in CATEGORICAL:
        encoders[cat].fit([])  # Dummy fit to set classes_

    # Encode previous batches
    X_train = []
    for batch_path in tqdm(os.listdir(cfg["batch_storage"]), desc="Encoding batches..."):
        batch = pd.read_csv(os.path.join(cfg["batch_storage"], batch_path))
        encoded_dfs = []
        for cat in CATEGORICAL:
            arr = batch[cat].apply(ast.literal_eval)
            enc = encoders[cat].transform(arr)
            encoded_dfs.append(pd.DataFrame(
                enc, columns=encoders[cat].classes_))
        batch = pd.concat(
            [batch.drop(columns=CATEGORICAL)] + encoded_dfs, axis=1)
        X_train.append(batch)
    X_train = pd.concat(X_train, axis=0)

    # Prepare features and target
    features = [i for i in X_train.columns if i not in IGNORE and i != TARGET]
    X = X_train[features]
    y = X_train[TARGET]

    # # Remove features with low correlation to the target
    # numerical_columns = X_train.select_dtypes(
    #     include=["int64", "int32", "float64"]).columns
    # correlation_matrix = X_train[numerical_columns].corr()
    # threshold = cfg.get("correlation_threshold", 0.1)
    # low_correlation_features = correlation_matrix[TARGET][abs(
    #     correlation_matrix[TARGET]) < threshold].index
    # columns_to_drop = [
    #     col for col in low_correlation_features if col in X.columns]
    # X = X.drop(columns=columns_to_drop)
    # print(
    #     f"Removed features with correlation below {threshold}: {list(low_correlation_features)}")
    encoders_path = os.path.join(run_folder, "encoders.pkl")
    with open(encoders_path, "wb") as f:
        pickle.dump(encoders, f)
    save_encoder_path_to_config(encoders_path)

    return X, y


def define_models():
    """Define models and their hyperparameter grids."""
    return {
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {
                "fit_intercept": [True, False],
                "positive": [True, False]
            }
        },
        "KNN": {
            "model": KNeighborsRegressor(),
            "params": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "p": [1, 2]
            }
        },
        "DecisionTree": {
            "model": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "params": {
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        },
        # "RandomForest": {
        #     "model": RandomForestRegressor(random_state=RANDOM_STATE),
        #     "params": {
        #         "n_estimators": [100, 200],
        #         "max_depth": [None, 10, 20],
        #         "min_samples_split": [2, 5],
        #         "min_samples_leaf": [1, 2]
        #     }
        # },
        # "GradientBoosting": {
        #     "model": GradientBoostingRegressor(random_state=RANDOM_STATE),
        #     "params": {
        #         "n_estimators": [100, 200],
        #         "learning_rate": [0.01, 0.1, 0.2],
        #         "max_depth": [3, 5, 7]
        #     }
        # },
    }


def train_and_evaluate(models, X_train, y_train, X_test, y_test, run_folder):
    """Train and evaluate models, and save metrics."""
    best_models = {}
    metrics = {}

    for name, config in models.items():
        print(f"\nTraining and tuning {name}...")

        # Start timing the training process
        start_time = time.time()

        # Grid search with 5-fold cross-validation
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=5,
            n_jobs=N_JOBS,
            scoring="neg_mean_squared_error",
            verbose=1
        )

        grid.fit(X_train, y_train)

        # End timing the training process
        end_time = time.time()
        training_time = end_time - start_time

        # Get best model
        best_model = grid.best_estimator_
        best_models[name] = best_model

        # Evaluate
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n{name} Results:")
        print(f"Best Parameters: {grid.best_params_}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")

        # Save metrics
        metrics[name] = {
            "best_params": grid.best_params_,
            "mae": mae,
            "mse": mse,
            "r2": r2,
            "training_time": training_time  # Add training time to metrics
        }

        save_model(best_model, f'{name}_best_model', run_folder)

    # Save metrics to YAML
    metrics = convert_numpy_types(metrics)
    save_metrics(metrics, run_folder)

    # Choose the best model based on MAE
    best_model_name = min(metrics, key=lambda x: metrics[x]["mae"])
    best_model_path = save_model(
        best_models[best_model_name], "best_model", run_folder)
    print(
        f"Best model: {best_model_name} with MAE = {metrics[best_model_name]['mae']:.4f}")

    return best_model_path


def save_best_model_path_to_config(best_model_path):
    """Save the path to the best model in config.yaml."""
    with open("options/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["best_model_path"] = best_model_path

    with open("options/config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    print(f"Best model path saved to config.yaml: {best_model_path}")


def train():
    # Preprocess data
    run_folder = create_run_folder(OUTPUT_DIR)
    X, y = preprocess_data(run_folder)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Define models
    models = define_models()
    # Train and evaluate models
    best_model_path = train_and_evaluate(
        models, X_train, y_train, X_test, y_test, run_folder)

    # Save the best model path to config.yaml
    save_best_model_path_to_config(best_model_path)


if __name__ == "__main__":
    train()
