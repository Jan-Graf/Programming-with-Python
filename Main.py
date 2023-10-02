import functions
import math
import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# define dataset directory based on file location
dataset_dir = os.path.join(os.path.dirname(__file__), "Dataset")
# store the raw data for each function to draw the functions
raw_train_data: pd.DataFrame
raw_ideal_data: pd.DataFrame

# define the train functions
train_functions = {}
# define the ideal functions
ideal_functions = {}

def main():
    get_funcs("train.csv")
    get_funcs("ideal.csv")

    set_ideal_functions()

    map_test_function()

    visualize_functions()

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
        # store the raw data in DataFrames
        if file_name == "train.csv":
            global raw_train_data
            raw_train_data = data
        elif file_name == "ideal.csv":
            global raw_ideal_data
            raw_ideal_data = data

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
                train_functions[train_func].check_ideal_function(ideal_functions[ideal_func], ideal_func)
    except:
        pass

def map_test_function():
    try:
        # check if there are 4 training functions as expected in the dict --> raise exception if not
        if len(train_functions) < 4:
            raise Exception("There are not enough training functions")
        elif len(train_functions) > 4:
            raise Exception("There are too much training functions")
        
        # load test function
        test_functions = pd.read_csv(os.path.join(dataset_dir, "test.csv"))
        # load training data (again)
        data = pd.read_csv(os.path.join(dataset_dir, "train.csv"), sep=",")
        # iter through test functions
        for index, row in test_functions.iterrows():
            # get y value of test function
            test_y = row["y"]

            for train_func in train_functions:
                # get y value of train function
                train_y = data.loc[data["x"] == row["x"], train_func].values[0]
                # if criterion is matched, add point to list
                if abs(train_y - test_y) < math.sqrt(2):
                    train_functions[train_func].mapped_points.append(functions.mapping_point(row["x"], train_y, train_y - test_y, train_functions[train_func].ideal_no))
    except:
        pass

def visualize_functions():
    # create a figure object
    fig = make_subplots(rows=2, cols=2, shared_xaxes=False, shared_yaxes=False, horizontal_spacing=0.1, vertical_spacing=0.1, subplot_titles=['Training Function y1', 'Training Function y2', 'Training Function y3', 'Training Function y4'])
    
    # iterate through all training functions
    for train_func in train_functions:
        # get row and col index of the function
        if train_func == "y1":
            row_index = 1
            col_index = 1
        elif train_func == "y2":
            row_index = 1
            col_index = 2
        elif train_func == "y3":
            row_index = 2
            col_index = 1
        else:
            row_index = 2
            col_index = 2
        
        # draw graph of the raw data
        fig.add_trace(
            go.Scatter(x=raw_train_data["x"], y=raw_train_data[train_func], mode="lines", name=train_func, legendgroup=train_func),
            row=row_index,
            col=col_index
        )

        # calucalte x and y values based on the ideal function
        ideal_function_no = train_functions[train_func].ideal_no
        x_values = np.linspace(raw_ideal_data["x"].min(), raw_ideal_data["x"].max(), raw_ideal_data["x"].count() + 1)
        y_values = ideal_functions[ideal_function_no].slope * x_values + ideal_functions[ideal_function_no].y_intercept
        # add ideal function
        fig.add_trace(
            go.Scatter(x=x_values, y=y_values, mode="lines", name="Ideal Function", legendgroup=train_func),
            row=row_index,
            col=col_index
        )
    
    fig.show()

if __name__ == "__main__":
    main()