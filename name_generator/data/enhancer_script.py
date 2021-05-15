from pathlib import Path

import pandas as pd
import wikipedia


def run(data_path, output_path):

    df = pd.read_csv(data_path, usecols=['name', 'industry'])

    def get_summary(x):
        summary = ''
        try:
            summary = wikipedia.summary(x)
        except:
            pass
        return summary

    for index, name in df.name.iteritems():
        summary = get_summary(name)
        if isinstance(summary, str):
            with output_path.open(mode='a+') as f:
                f.write(name)
                f.write('\n')
                f.write(summary)
                f.write('\n===========================\n')


if __name__ == '__main__':
    DATA_PATH = Path('./data/kaggle_dataset/companies_sorted.csv')
    OUTPUT_PATH = Path('./data/text_descriptions')
    OUTPUT_PATH.mkdir(parents=True)

    run(DATA_PATH, OUTPUT_PATH / 'descriptions.txt')
