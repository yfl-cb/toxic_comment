import pandas as pd


def read_csv(file):
    df = pd.read_csv('data/' + file)
    return df


if __name__ == '__main__':
    print(read_csv('train.csv'))
