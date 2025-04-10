import argparse
import logging
import os
import yaml
import pandas as pd
from auto_eda import auto_eda
from train import train
from infer import run_inference
from dataset_utils import create_df, clean_df, get_batch, get_json_data, load_from_config, monitor_and_handle_data_drift, data_quality_metrics


def generate_summary_report(config):
    """Generate a summary report of data quality, model metrics, and hyperparameters."""
    report_storage = config["report_storage"]
    os.makedirs(report_storage, exist_ok=True)
    report_path = os.path.join(report_storage, "monitoring_report.txt")

    with open(report_path, "w", encoding='utf-8') as report_file:
        # 1. Data Quality Changes Over Time
        report_file.write("### Data Quality Changes Over Time ###\n")
        batch_storage = config["batch_storage"]
        batch_files = sorted(os.listdir(batch_storage))
        for batch_file in batch_files:
            batch_path = os.path.join(batch_storage, batch_file)
            df = pd.read_csv(batch_path)
            metrics = data_quality_metrics(df)

            report_file.write(f"Batch: {batch_file}\n")
            report_file.write(
                f"  Missing Values: {metrics['missing_values']}\n")
            report_file.write(f"  Duplicates: {metrics['duplicates']}\n")
            report_file.write(f"  Outliers: {metrics['outliers']}\n")
            report_file.write(f"  Zero values: {metrics['zero_values']}\n")
            report_file.write("\n")

        # 2. Best Model Metrics Over Time
        report_file.write("### Best Model Metrics Over Time ###\n")
        results_dir = config["output_dir"]
        for root, dirs, files in os.walk(results_dir):
            if "metrics.yaml" in files:
                metrics_path = os.path.join(root, "metrics.yaml")
                with open(metrics_path, "r") as metrics_file:
                    metrics = yaml.safe_load(metrics_file)
                    report_file.write(f"Metrics from: {metrics_path}\n")
                    for model_name, model_metrics in metrics.items():
                        report_file.write(f"Model: {model_name}\n")
                        report_file.write(f"  MAE: {model_metrics['mae']}\n")
                        report_file.write(f"  MSE: {model_metrics['mse']}\n")
                        report_file.write(f"  R²: {model_metrics['r2']}\n")
                        report_file.write(
                            f"  Training Time: {model_metrics.get('training_time', 'N/A')} seconds\n")
                        report_file.write(
                            f"  Best Params: {model_metrics['best_params']}\n")
                        report_file.write("\n")

        # 3. Selected Hyperparameters
        report_file.write("### Selected Hyperparameters ###\n")
        for root, dirs, files in os.walk(results_dir):
            if "metrics.yaml" in files:
                metrics_path = os.path.join(root, "metrics.yaml")
                with open(metrics_path, "r") as metrics_file:
                    metrics = yaml.safe_load(metrics_file)
                    report_file.write(
                        f"Hyperparameters from: {metrics_path}\n")
                    for model_name, model_metrics in metrics.items():
                        report_file.write(f"Model: {model_name}\n")
                        report_file.write(
                            f"  Best Params: {model_metrics['best_params']}\n")
                        report_file.write("\n")

    logging.info(f"Summary report generated at {report_path}")
    return report_path
