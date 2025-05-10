import ast
import os
import pickle
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import yaml
from utils.dataset_utils import load_from_config


def load_best_model(config_path="options/config.yaml"):
    """Load the best model from the path specified in config.yaml."""
    config = load_from_config()

    best_model_path = config.get("best_model_path")
    if not best_model_path or not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Best model not found at {best_model_path}. Check config.yaml.")

    with open(best_model_path, "rb") as f:
        model = pickle.load(f)

    print(f"Loaded best model from {best_model_path}")
    return model


def run_inference(input_file, output_file, config_path="options/config.yaml"):
    """Run inference on the input file and save the results."""
    # Load the best model
    model = load_best_model(config_path)

    # Load the input data
    df = pd.read_csv(input_file)
    print(f"Loaded input data from {input_file}")

    # Keep a reference to the original DataFrame
    original_df = df.copy()

    # Ensure the input data has the required features
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    ignore_columns = config.get("IGNORE", [])
    target_column = config.get("TARGET", "price")
    categorical_columns = config.get("CATEGORICAL", [])

    unique_sets = {cat: set() for cat in categorical_columns}
    for cat in categorical_columns:
        arr = df[cat].apply(ast.literal_eval)
        for item in arr:
            unique_sets[cat].update(item)

    # Initialize encoders
    encoders_path = config.get("encoders_path")
    if not encoders_path or not os.path.exists(encoders_path):
        raise FileNotFoundError(
            f"Encoders file not found at {encoders_path}. Check config.yaml."
        )

    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)

    encoded_dfs = []
    for cat in categorical_columns:
        arr = df[cat].apply(ast.literal_eval)
        enc = encoders[cat].transform(arr)
        encoded_dfs.append(pd.DataFrame(enc, columns=encoders[cat].classes_))
    df = pd.concat([df.drop(columns=categorical_columns)] +
                   encoded_dfs, axis=1)

    features = [
        col for col in df.columns if col not in ignore_columns and col != target_column
    ]
    if not all(feature in df.columns for feature in features):
        missing_features = [
            feature for feature in features if feature not in df.columns
        ]
        raise ValueError(
            f"Input data is missing required features: {missing_features}"
        )

    # Run predictions
    predictions = model.predict(df[features])
    print("Predictions added to the DataFrame.")

    # Attach predictions to the original DataFrame
    original_df["predict"] = predictions

    # Save the results
    original_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    return output_file
