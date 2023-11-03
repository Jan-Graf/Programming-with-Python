import functions
import math
import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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
            raise Exception("Only the name of the file is expected. No information about the file path.")
        
        # store the complete path to the file in a variable and check, if the file exists --> raise exception if not
        complete_path = os.path.join(dataset_dir, file_name)
        if not os.path.isfile(complete_path):
            raise FileNotFoundError
        
        # read csv file and store it in a data frame
        data = pd.read_csv(complete_path, sep=",")

        for y_axes in list(data.columns.values):
            # skip the x-axis
            if y_axes == "x":
                continue

            # calculate the params for the function with degree 1
            coefficients, residuals, rank, singular_values, rcond = np.polyfit(data["x"], data[y_axes], 1, full=True)
            
            # define residuals
            min_residuals = residuals[0]
            # while the residuals are equal, the min residual has been updated --> another iteration
            while min_residuals == residuals:
                # get parameters for the function of the next degree
                updated_coefficients, residuals, rank, singular_values, rcond = np.polyfit(data["x"], data[y_axes], len(coefficients), full=True)
                
                # if no residuals can be found, the function can't be fitted --> exit loop
                if len(residuals) == 0:
                    break

                # use rounded residuals because if not really big functions will be returned (degree of 36 and coeffiecents like 2e-46)
                if round(residuals[0], 3) < round(min_residuals, 3) and rank >= len(updated_coefficients) - 1:
                    min_residuals = residuals[0]
                    coefficients = updated_coefficients
            
            if file_name == "train.csv":
                train_functions[y_axes] = functions.train_function(data[["x", y_axes]], coefficients)
            elif file_name == "ideal.csv":
                ideal_functions[y_axes] = functions.function(data[["x", y_axes]], coefficients)
            else:
                raise Exception("The function for the test data is not neccessary.")
    except Exception as error:
        # handle the exception
        print(error)
        exit()

def set_ideal_functions():
    '''
    Find the ideal function for every training function
    '''
    try:
        # check if there are 50 ideal functions as expected in the dict --> raise exception if not
        if len(ideal_functions) < 50:
            raise Exception("There are not enough training functions")
        elif len(ideal_functions) > 50:
            raise Exception("There are too much training functions")
        
        # iterate through all training functions
        for train_func in train_functions:
            # iterate through ideal functions
            for ideal_func in ideal_functions:
                # check, if the current ideal function fits better to the training function as the actual one
                train_functions[train_func].check_ideal_function(ideal_functions[ideal_func], ideal_func)
            
            # predict the values for the ideal function
            ideal_functions[train_functions[train_func].ideal_no].predict_values()
    except Exception as error:
        # handle the exception
        print(error)
        exit()

def map_test_function():
    try:
        # check if there are 4 training functions as expected in the dict --> raise exception if not
        if len(train_functions) < 4:
            raise Exception("There are not enough training functions")
        elif len(train_functions) > 4:
            raise Exception("There are too much training functions")
        
        # iterate through training functions
        for train_func in train_functions:
            # get ideal function
            ideal_func = ideal_functions[train_functions[train_func].ideal_no]
            # iterate through ideal values
            for index, row in ideal_func.predicted_data.iterrows():
                # get y deviation
                delta_y = abs(row["y"] - train_functions[train_func].raw_data.at[index, "y"])
                # if the deviation is smaller than zero add mapping point
                if delta_y < math.sqrt(2):
                    train_functions[train_func].mapped_points.append(functions.mapping_point(row["x"], row["y"], delta_y))
    except Exception as error:
        # handle the exception
        print(error)
        exit()

def visualize_functions():
    # create a figure object
    fig = make_subplots(rows=2, cols=2, shared_xaxes=False, shared_yaxes=False, horizontal_spacing=0.1, vertical_spacing=0.1, subplot_titles=['Training Function y1', 'Training Function y2', 'Training Function y3', 'Training Function y4'])
    
    index_mapping = {"y1": (1, 1), "y2": (1, 2), "y3": (2, 1), "y4": (2, 2)}
    # iterate through all training functions
    for train_func in train_functions:
        # get row and col index of the function
        row_index, col_index = index_mapping[train_func]
        
        # draw graph of the raw data
        fig.add_trace(
            go.Scatter(x=train_functions[train_func].raw_data["x"], y=train_functions[train_func].raw_data["y"], mode="lines", name=train_func),
            row=row_index,
            col=col_index
        )

        # define ideal values
        ideal_x = ideal_functions[train_functions[train_func].ideal_no].predicted_data["x"]
        ideal_y = ideal_functions[train_functions[train_func].ideal_no].predicted_data["y"]
        
        # add ideal function
        fig.add_trace(
            go.Scatter(x=ideal_x, y=ideal_y, mode="lines", name="Ideal Function for "+ train_func),
            row=row_index,
            col=col_index
        )
    
    fig.show()

if __name__ == "__main__":
    main()