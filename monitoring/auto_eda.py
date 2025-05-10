import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
from utils.load_config import load_from_config
import dataframe_image as dfi


def auto_eda(df):
    """Perform automatic EDA and save results to a log file."""
    config = load_from_config()
    output_dir = config["eda_storage"]
    os.makedirs(output_dir, exist_ok=True)

    # Redirect output to a log file
    log_file_path = os.path.join(output_dir, "eda_report.txt")
    with open(log_file_path, "w", encoding='utf-8') as log_file:
        def log_message(message):
            log_file.write(message + "\n")

        # 1. General information
        # log_message("### General Information ###")
        # log_message(str(df.info()))
        log_message("\n### First 5 Rows ###")
        log_message(str(df.head()))
        log_message("\n### Last 5 Rows ###")
        log_message(str(df.tail()))

        # 2. Missing values
        log_message("\n### Missing Values ###")
        missing_values = df.isnull().sum()
        log_message(str(missing_values[missing_values > 0]))

        # 3. Basic statistics
        log_message("\n### Basic Statistics ###")
        log_message(str(df.describe()))

        # 4. Distribution of numerical data
        log_message("\n### Distribution of Numerical Data ###")
        numerical_columns = df.select_dtypes(
            include=['int64', 'float64']).columns
        for col in numerical_columns:
            plt.figure(figsize=(16, 16))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.savefig(f"{output_dir}/distribution_{col}.png")
            plt.close()

        # 5. Distribution of categorical data
        log_message("\n### Distribution of Categorical Data ###")
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            plt.figure(figsize=(16, 16))
            df[col].value_counts().head(10).plot(kind='bar')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.savefig(f"{output_dir}/distribution_{col}.png")
            plt.close()

        # 6. Correlation matrix
        log_message("\n### Correlation Matrix ###")
        plt.figure(figsize=(10, 10))
        correlation_matrix = df[numerical_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.savefig(f"{output_dir}/correlation_matrix.png")
        plt.close()

        # 7. Outliers (based on IQR)
        log_message("\n### Outliers in Numerical Data ###")
        for col in numerical_columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            log_message(f"{col}: {len(outliers)} outliers")

        # 8. Proportions of categorical data
        log_message("\n### Proportions of Categorical Data ###")
        for col in categorical_columns:
            log_message(f"\n{col}:")
            log_message(str(df[col].value_counts(normalize=True).head(10)))

        # 9. Distribution of list-like categorical data
        log_message("\n### Distribution of List-like Categorical Data ###")
        list_like_columns = ['tags', 'genres', 'categories']
        for col in list_like_columns:
            if col in df.columns:
                plt.figure(figsize=(16, 16))
                exploded = df[col].apply(ast.literal_eval).explode()
                exploded.value_counts().head(10).plot(kind='bar')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.savefig(f"{output_dir}/distribution_{col}.png")
                plt.close()

    print(f"EDA report saved to {log_file_path}")

    dfi.export(df.describe(), f"{output_dir}/stats_table.png")

    correlation_matrix.to_csv(f"{output_dir}/correlation_matrix.csv")


if __name__ == "__main__":
    df = pd.read_csv('raw_batches/batch_17.csv')
    auto_eda(df)
