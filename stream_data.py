import os
import pandas as pd
import time



def stream_data_chronologically(file_path, batch_size, output_dir, date_column):
    df = pd.read_csv(file_path)

    df[date_column] = pd.to_datetime(df['release_date'], errors='coerce')

    df = df.sort_values(by=date_column)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        batch.to_csv(f"{output_dir}/batch_{i // batch_size}.csv", index=False)
        # time.sleep(1)  # Имитируем задержку в 1 секунду между пакетами
        print(f"Batch {i // batch_size} saved.")




if __name__ == "__main__":
    stream_data_chronologically(
        file_path="games.csv",
        batch_size=100,
        output_dir="raw_batches",
        date_column="release_date"
    )

