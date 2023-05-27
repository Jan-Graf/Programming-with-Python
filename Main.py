import os
import pandas as pd
import MyMath

# define dataset directory based on file location
dataset_dir = os.path.join(os.path.dirname(__file__), "Dataset")
# define the train functions
train_functions = []
# define the ideal functions
ideal_functions = []

def main():
    # read main and ideal data
    train_data = pd.read_csv(os.path.join(dataset_dir, "train.csv"), sep=",")
    ideal_data = pd.read_csv(os.path.join(dataset_dir, "ideal.csv"), sep=",")

    # get covariances
    train_cov = train_data.cov(ddof=0, numeric_only=True)
    ideal_cov = ideal_data.cov(ddof=0, numeric_only=True)
    # get variances
    train_var = train_data.var(axis=0, skipna= True, ddof=0, numeric_only=True)
    ideal_var = ideal_data.var(axis=0, skipna= True, ddof=0, numeric_only=True)
    
    train_avg_x = train_data.loc[:, "x"].mean()
    for y_axes in list(train_data.columns.values):
        # skip the x axis
        if y_axes == "x":
            continue
        # calculate the params for the function
        b = train_cov.loc["x"][y_axes] / train_var["x"]
        a = train_data.loc[:, y_axes].mean() - b * train_avg_x

        # store function object in list
        train_functions.append(MyMath.LinearFunction(a, b))

    ideal_avg_x = ideal_data.loc[:, "x"].mean()
    for y_axes in list(ideal_data.columns.values):
        # skip the x axis
        if y_axes == "x":
            continue
        # calculate the params for the function
        b = ideal_cov.loc["x"][y_axes] / ideal_var["x"]
        a = ideal_data.loc[:, y_axes].mean() - b * ideal_avg_x

        # store function object in list
        ideal_functions.append(MyMath.LinearFunction(a, b))

if __name__ == "__main__":
    main()