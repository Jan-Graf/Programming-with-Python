import functions
import os
import pandas as pd

# define dataset directory based on file location
dataset_dir = os.path.join(os.path.dirname(__file__), "Dataset")
# define the train functions
train_functions = {}
# define the ideal functions
ideal_functions = {}

def main():
    get_funcs("train.csv")
    get_funcs("ideal.csv")

    set_ideal_functions()

def get_funcs(file_name: str) -> None:
    '''
    Read the given *.csv file and store the linear functions in a dict
    
    :param file_name str: The file name of the csv-file with file extension
    '''
    try:
        # check if the given file name has the correct extension --> raise exception if not
        if not file_name.endswith(".csv"):
            raise Exception("The file name has no or the wrong file extension.")
        # check if the file name is a name and no path --> raise exception if not
        if "/" in file_name or "\\" in file_name:
            raise Exception("Only the name if the file is expected. No information about the file path.")
        
        # store the complete path to the file in a variable and check, if the file exists --> raise exception if not
        complete_path = os.path.join(dataset_dir, file_name)
        if not os.path.isfile(complete_path):
            raise FileNotFoundError
        
        # read csv file and store it in a data frame
        data = pd.read_csv(complete_path, sep=",")
        # get covariances and variances
        cov = data.cov(ddof=0, numeric_only=True)
        var = data.var(axis=0, skipna= True, ddof=0, numeric_only=True)
        # get mean of x values
        x_mean = data.loc[:, "x"].mean()

        for y_axes in list(data.columns.values):
            # skip the x-axis
            if y_axes == "x":
                continue
            # calculate the params for the function
            b = cov.loc["x"][y_axes] / var["x"]
            a = data.loc[:, y_axes].mean() - b * x_mean

            # depending on the file, add function to dict
            if file_name == "train.csv":
                train_functions[y_axes] = functions.train_function(a, b)
            elif file_name == "ideal.csv":
                ideal_functions[y_axes] = functions.linear_function(a, b)
            else:
                raise Exception("The function for the test data is not neccessary.")
    except:
        pass

def set_ideal_functions():
    try:
        # check if there are 4 training functions as expected in the dict --> raise exception if not
        if len(train_functions) < 4:
            raise Exception("There are not enough training functions")
        elif len(train_functions) > 4:
            raise Exception("There are too much training functions")
        # check if there are 50 ideal functions as expected in the dict --> raise exception if not
        if len(ideal_functions) < 50:
            raise Exception("There are not enough training functions")
        elif len(ideal_functions) > 50:
            raise Exception("There are too muach training functions")
        
        # iterate through all training functions
        for train_func in train_functions:
            # default: set first ideal function as ideal function
            train_functions[train_func].ideal_function = ideal_functions["y1"]

            # iterate through ideal functions
            for ideal_func in ideal_functions:
                # check, if the current ideal function fits better to the training function as the actual one
                train_functions[train_func].check_ideal_function(ideal_functions[ideal_func])
    except:
        pass

if __name__ == "__main__":
    main()