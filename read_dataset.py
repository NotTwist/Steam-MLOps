import pandas as pd
import json
import numpy as np

dataset_location = "games.json"


def create_df(json) -> pd.DataFrame:
    unnecessary_vars = [
        'packages', 'screenshots', 'movies', 'score_rank', 'header_image',
        'reviews', 'website', 'support_url', 'notes', 'support_email',
        'user_score', 'required_age', 'metacritic_score',
        'metacritic_url',  'detailed_description', 'about_the_game',
        'windows', 'mac', 'linux', 'achievements', 'full_audio_languages',
        'dlc_count', 'supported_languages', 'developers', 'publishers'
    ]
    games = [{
        **{k: v for k, v in game_info.items() if k not in unnecessary_vars},
        'tags': list(tags.keys()) if isinstance((tags := game_info.get('tags', {})), dict) else [],
        'tag_frequencies': list(tags.values()) if isinstance(tags, dict) else [],
        'app_id': app_id
    } for app_id, game_info in json_data.items()]

    # Create a DataFrame from the processed list
    df = pd.DataFrame(games)
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:


    # remove games with zero owners or zero reviews and categories
    df = df[~((df['estimated_owners'] == "0 - 0") | (df['positive'] +
                                                     df['negative'] == 0) | (df['categories'].str.len() == 0))]

    # Split estimated_owners into two: min_owners and max_owners
    df[['min_owners', 'max_owners']] = df['estimated_owners'].str.split(
        ' - ', expand=True)

    # Remove the original field
    df = df.drop('estimated_owners', axis=1)

    return df


if __name__ == "__main__":
    with open(dataset_location, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    df = create_df(json_data)
    df = clean_df(df)
    from auto_eda import auto_eda
    auto_eda(df)
    df.to_csv('games.csv', index=False)
