import os
import pandas as pd
import pandas.api.types as types
from sklearn.model_selection import train_test_split

def get_quantitatives(variables_family: str = None, excluded_variables: list = list()):
    variables = {
        "fac": list(map(lambda i: "fac_{}".format(i), range(1, 217))),
        "fou": list(map(lambda i: "fou_{}".format(i), range(1, 77))),
        "kar": list(map(lambda i: "kar_{}".format(i), range(1, 65))),
        "mor": list(map(lambda i: "mor{}".format(i), range(1, 7))),
        "pix": list(map(lambda i: "pix_{}".format(i), range(1, 241))),
        "zer": list(map(lambda i: "zer_{}".format(i), range(1, 48))),
    }

    final_variables = set(variables[variables_family]) if variables_family in variables else set(
        variables["fac"]
        + variables["fou"]
        + variables["kar"]
        + variables["mor"]
        + variables["pix"]
        + variables["zer"]
    )

    final_variables = final_variables.difference(excluded_variables)

    return final_variables


def load_train_dataset(variables_family: str = None, return_dataframe:bool = False, excluded_variables: list = list()):
    path_train = "{}/datasets/data_train.csv".format(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(path_train)

    quantitatives = get_quantitatives(variables_family=variables_family, excluded_variables=excluded_variables)
    categorical = "class"
    df[categorical] = df[categorical].astype(types.CategoricalDtype(ordered=False))

    if return_dataframe:
        quantitatives.add(categorical)
        return df[quantitatives]

    return df[quantitatives].values, df[categorical].values, quantitatives


def load_test_dataset():
    path_train = "{}/datasets/data_test.csv".format(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(path_train)
    quantitatives = get_quantitatives()

    return df[quantitatives].values


def find_best_train_dataset(test_size: float):
    X, y = load_train_dataset()

    lowest_std = 1.56347
    random_state = 13255

    for i in range(50000, 100000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)

        current_std = y_test.describe().counts.std()

        if lowest_std > current_std:
            random_state = i
            lowest_std = current_std
            print("Found lowest std {:.2f} than before for random_state = {}".format(lowest_std, random_state))

    return random_state, lowest_std
