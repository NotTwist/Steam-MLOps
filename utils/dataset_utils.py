from scipy.stats import ks_2samp
import os
import time
import pandas as pd
import json
import numpy as np
import yaml
import logging
import ast
from monitoring.auto_eda import auto_eda
from utils.load_config import load_from_config

def get_json_data(dataset_location: str) -> dict:
    if os.path.exists(dataset_location):
        with open(dataset_location, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    else:
        raise FileNotFoundError(f"File {dataset_location} not found.")
    return json_data


def create_df(json) -> pd.DataFrame:
    unnecessary_vars = [
        'packages', 'screenshots', 'movies', 'score_rank', 'header_image',
        'reviews', 'website', 'support_url', 'notes', 'support_email',
        'user_score', 'required_age', 'metacritic_score',
        'metacritic_url',  'detailed_description', 'about_the_game',
        'windows', 'mac', 'linux', 'achievements', 'full_audio_languages',
        'dlc_count', 'supported_languages', 'developers', 'publishers', 'discount'
    ]
    games = [{
        **{k: v for k, v in game_info.items() if k not in unnecessary_vars},
        'tags': list(tags.keys()) if isinstance((tags := game_info.get('tags', {})), dict) else [],
        'tag_frequencies': list(tags.values()) if isinstance(tags, dict) else [],
        'app_id': app_id
    } for app_id, game_info in json.items()]

    # Create a DataFrame from the processed list
    df = pd.DataFrame(games)
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:

    # remove games with zero owners or zero reviews and categories
    df = df[~((df['estimated_owners'] == "0 - 0") | (df['positive'] +
                                                     df['negative'] == 0) | (df['categories'].str.len() == 0))]

    # Split estimated_owners into two: min_owners and max_owners
    df = df.copy()
    df[['min_owners', 'max_owners']] = df['estimated_owners'].str.split(
        ' - ', expand=True)

    # Remove the original field
    df = df.drop('estimated_owners', axis=1)

    df['positive_percent'] = df['positive'] / \
        (df['positive'] + df['negative']) * 100

    return df


def stream_data_chronologically(file_path, batch_size, output_dir, date_column, delay=0, verbose=True):
    if verbose:
        logging.info(
            f"Streaming data from {file_path} in batches of {batch_size}...")
    df = pd.read_csv(file_path)

    df[date_column] = pd.to_datetime(df['release_date'], errors='coerce')

    df = df.sort_values(by=date_column)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        batch.to_csv(f"{output_dir}/batch_{i // batch_size}.csv", index=False)
        time.sleep(delay)  # Имитируем задержку в 1 секунду между пакетами
        if verbose:
            print(f"Batch {i // batch_size} saved.", end='\r')


def get_batch(file_path, batch_size, date_column, batch_number=0, output_dir=None, verbose=True):
    if verbose:
        logging.info(
            f"Fetching batch number {batch_number} from {file_path}...")
    df = pd.read_csv(file_path)
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.sort_values(by=date_column)

    start_index = batch_number * batch_size
    end_index = start_index + batch_size
    batch = df.iloc[start_index:end_index]

    if batch.empty:
        if verbose:
            logging.info(
                f"No more batches available. Batch number {batch_number} is empty.")
        return None  # No more batches available

    if not validate_batch(batch, batch_number, verbose=verbose):
        batch_number += 1
        update_batch_number('options/config.yaml', batch_number)
        return None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        batch_file_path = f"{output_dir}/batch_{batch_number}.csv"
        batch.to_csv(batch_file_path, index=False)
        if verbose:
            logging.info(f"Batch {batch_number} saved to {batch_file_path}.")

    batch_number += 1
    update_batch_number('options/config.yaml', batch_number)

    return batch


def update_batch_number(config_path, new_batch_number):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['current_batch_number'] = new_batch_number

    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)




# Dataset quality check


def data_quality_metrics(df):
    metrics = {
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "outliers": {col: ((df[col] < df[col].quantile(0.01)) | (df[col] > df[col].quantile(0.99))).sum()
                     for col in df.select_dtypes(include=["float64", "int64"]).columns},
        "zero_values": {col: (df[col] == 0).sum() for col in df.select_dtypes(include=["float64", "int64"]).columns}
    }
    return metrics


def validate_batch(df, batch_numder, verbose=True):
    config = load_from_config()
    max_missing = config.get("max_missing", 0.1)
    max_duplicates = config.get("max_duplicates", 0.05)
    max_outliers = config.get("max_outliers", 0.1)
    # max_zero = config.get("max_zeros", 0.1)
    metrics = data_quality_metrics(df)
    total_rows = len(df)

    # Calculate thresholds
    missing_threshold = total_rows * max_missing
    # zero_threshold = total_rows * max_zero
    duplicate_threshold = total_rows * max_duplicates
    outlier_threshold = total_rows * max_outliers
    if verbose:
        save_data_quality_report(metrics, batch_numder)
    # Check if metrics exceed thresholds
    if any(value > missing_threshold for value in metrics["missing_values"].values()):
        logging.warning("Batch skipped due to excessive missing values.")
        return False
    if metrics["duplicates"] > duplicate_threshold:
        logging.warning("Batch skipped due to excessive duplicates.")
        return False
    if any(value > outlier_threshold for value in metrics["outliers"].values()):
        logging.warning("Batch skipped due to excessive outliers.")
        return False
    # if any(value > zero_threshold for value in metrics["zero_values"].values()):
    #     logging.warning("Batch skipped due to excessive zero values.")
    #     return False

    return True


def convert_numpy_types(data):
    """
    Recursively convert numpy types to native Python types.
    """
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, (np.integer, np.int64)):
        return int(data)
    elif isinstance(data, (np.floating, np.float64)):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.bool_, bool)):
        return bool(data)
    else:
        return data


def save_data_quality_report(metrics: dict, batch_number: int):
    """
    Save data quality metrics to a JSON file.
    """
    config = load_from_config()
    output_dir = config["report_storage"]
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/data_quality_report_batch_{batch_number}.yaml"
    metrics = convert_numpy_types(metrics)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(metrics, f, indent=4)
    logging.info(f"Data quality report saved to {output_path}")

def detect_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, columns: list) -> dict:
    """
    Detect data drift using KS-test for numerical columns.
    """
    drift_results = {}
    for col in columns:
        stat, p_value = ks_2samp(
            reference_df[col].dropna(), current_df[col].dropna())
        drift_results[col] = {"p_value": p_value,
                              "drift_detected": p_value < 0.05}
    return drift_results


def create_reference_data(batch_storage: str, batch_number: int, output_path: str):
    """
    Create reference data by combining all previous batches.
    """
    all_batches = []
    for i in range(batch_number):
        batch_file = os.path.join(batch_storage, f"batch_{i}.csv")
        if os.path.exists(batch_file):
            batch_df = pd.read_csv(batch_file)
            all_batches.append(batch_df)
        else:
            logging.warning(f"Batch file {batch_file} not found. Skipping.")

    if all_batches:
        reference_df = pd.concat(all_batches, ignore_index=True)
        return reference_df
    else:
        logging.warning("No batches found to create reference data.")


def monitor_and_handle_data_drift(
    batch_storage: str,
    batch_number: int,
    current_df: pd.DataFrame,
    output_dir: str,
    update_reference: bool = False
):
    """
    Monitor and handle data drift by comparing current data with reference data.

    Args:
        batch_storage (str): Path to the directory containing batch files.
        batch_number (int): Current batch number.
        current_df (pd.DataFrame): Current batch of data.
        output_dir (str): Directory to save drift reports.
        update_reference (bool): Whether to update reference data with the current batch.
    """
    # Step 1: Create reference data from previous batches
    logging.info("Creating reference data from previous batches...")
    reference_df = create_reference_data(
        batch_storage, batch_number, output_path=None)
    if reference_df is None or reference_df.empty:
        logging.warning(
            "No reference data available. Skipping data drift detection.")
        return

    # Step 2: Automatically detect numerical columns
    numerical_columns = current_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if not numerical_columns:
        logging.warning("No numerical columns found for data drift detection.")
        return

    # Step 3: Detect data drift using KS-test
    logging.info("Detecting data drift...")
    drift_results = detect_data_drift(
        reference_df, current_df, numerical_columns)

    # Step 4: Handle data drift
    os.makedirs(output_dir, exist_ok=True)
    drift_report_path = os.path.join(
        output_dir, f"data_drift_report_{batch_number}.yaml")
    drift_results = convert_numpy_types(drift_results)
    with open(drift_report_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(drift_results, f, indent=4)
    logging.info(f"Data drift report saved to {drift_report_path}")

    # Log warnings for detected drift
    for col, result in drift_results.items():
        if result["drift_detected"]:
            logging.warning(
                f"Data drift detected in column '{col}' (p-value: {result['p_value']:.5f})")

    # # Step 5: Optionally update reference data
    # if update_reference:
    #     logging.info("Updating reference data with the current batch...")
    #     updated_reference_df = pd.concat(
    #         [reference_df, current_df], ignore_index=True)
    #     reference_data_path = os.path.join(output_dir, "reference_data.csv")
    #     updated_reference_df.to_csv(reference_data_path, index=False)
    #     logging.info(f"Updated reference data saved to {reference_data_path}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("dataset_utils.log"),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting the dataset processing script.")
    config = load_from_config()
    json_data = get_json_data(config['dataset_location'])
    logging.info(f"Loaded JSON data from {config['dataset_location']}.")
    df = create_df(json_data)
    df = clean_df(df)
    logging.info("DataFrame created and cleaned.")
    df.to_csv(config['csv_file'], index=False)
    stream_data_chronologically(
        file_path=config['csv_file'],
        batch_size=config['batch_size'],
        output_dir=config['batch_storage'],
        date_column="release_date"
    )

    # batch = get_batch(
    #     file_path=config['csv_file'],
    #     batch_size=config['batch_size'],
    #     date_column="release_date",
    #     batch_number=config["current_batch_number"],
    #     output_dir=config['batch_storage'],
    # )
    # auto_eda(batch)
    logging.info("Data streaming completed.")
