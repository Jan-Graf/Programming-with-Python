import os
import pandas as pd

# define dataset directory based on file location
dataset_dir = os.path.join(os.path.dirname(__file__), "Dataset")

def main():
    # read main and ideal data
    train_data = pd.read_csv(os.path.join(dataset_dir, "train.csv"), sep=",")
    ideal_data = pd.read_csv(os.path.join(dataset_dir, "ideal.csv"), sep=",")

    # get covariances
    train_cov = train_data.cov(ddof=0, numeric_only=True)
    ideal_cov = ideal_data.cov(ddof=0, numeric_only=True)
    # get variances
    train_var = train_data.var()
    ideal_var = ideal_data.var()

if __name__ == "__main__":
    main()