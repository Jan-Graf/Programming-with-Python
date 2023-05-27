import os
import pandas as pd

# define dataset directory based on file location
dataset_dir = os.path.join(os.path.dirname(__file__), "Dataset")

def main():
    # read main and ideal data
    train_data = pd.read_csv(os.path.join(dataset_dir, "train.csv"), sep=",")
    ideal_data = pd.read_csv(os.path.join(dataset_dir, "ideal.csv"), sep=",")

if __name__ == "__main__":
    main()