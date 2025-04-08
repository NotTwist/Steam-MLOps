import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function for automatic EDA


def auto_eda(df, output_dir="eda_images"):
    import os
    # Create directory for saving images
    os.makedirs(output_dir, exist_ok=True)

    # 1. General information
    print("### General Information ###")
    print(df.info())
    print("\n### First 5 Rows ###")
    print(df.head())
    print("\n### Last 5 Rows ###")
    print(df.tail())

    # 2. Missing values
    print("\n### Missing Values ###")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    # 3. Basic statistics
    print("\n### Basic Statistics ###")
    print(df.describe())

    # 4. Distribution of numerical data
    print("\n### Distribution of Numerical Data ###")
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_columns:
        plt.figure(figsize=(16, 8))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(f"{output_dir}/distribution_{col}.png")  # Save the plot
        plt.close()  # Close the plot to avoid displaying it

    # 5. Distribution of categorical data
    print("\n### Distribution of Categorical Data ###")
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        plt.figure(figsize=(16, 8))
        df[col].value_counts().head(10).plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(f"{output_dir}/distribution_{col}.png")  # Save the plot
        plt.close()  # Close the plot to avoid displaying it

    # 6. Correlation matrix
    print("\n### Correlation Matrix ###")
    plt.figure(figsize=(10, 6))
    correlation_matrix = df[numerical_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig(f"{output_dir}/correlation_matrix.png")  # Save the plot
    plt.close()  # Close the plot to avoid displaying it

    # 7. Outliers (based on IQR)
    print("\n### Outliers in Numerical Data ###")
    for col in numerical_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"{col}: {len(outliers)} outliers")

    # 8. Proportions of categorical data
    print("\n### Proportions of Categorical Data ###")
    for col in categorical_columns:
        print(f"\n{col}:")
        print(df[col].value_counts(normalize=True).head(10))
