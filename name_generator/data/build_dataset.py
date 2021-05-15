from pathlib import Path

import pandas as pd


def run(data_path, output_path):

    with (DATA_PATH / 'descriptions.txt').open(mode='r') as f:
        text = f.read()

    names = []
    descriptions = []
    for definition in text.split('===========================\n'):
        business_name, _, business_description = definition.partition('\n')
        names.append(business_name)
        descriptions.append(business_description.strip())

    df = pd.DataFrame({
        'name': names,
        'description': descriptions
    })

    df = df.loc[df.description.str.len() > 0]

    df.to_pickle(DATA_PATH / 'dataset.pkl')
    df.to_csv(DATA_PATH / 'dataset.csv')


if __name__ == '__main__':
    DATA_PATH = Path('./data/text_descriptions')
    OUTPUT_PATH = Path('./data/dataset')
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    run(DATA_PATH, OUTPUT_PATH)
