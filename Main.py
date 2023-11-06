import functions
import math
import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from plotly.subplots import make_subplots
from sqlalchemy import create_engine

# define dataset directory based on file location
working_dir = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(working_dir, "Dataset")

# define the train functions
train_functions = {}
# define the ideal functions
ideal_functions = {}

def main():
    global train_functions, ideal_functions

    create_database()

    get_funcs(train_functions, "train.csv", "TrainingFunction")
    get_funcs(ideal_functions, "ideal.csv", "IdealFunctions")

    set_ideal_functions()

    map_test_function(train_functions)

    # visualize_functions()

def create_database() -> None:
    '''
    Create an empty database file. If a file exists, clear the database

    Args:
        db_engine Engine: The engine object for the connection to the database
    '''
    db_path = os.path.join(working_dir, "PyTask.db")
    # if the db file already exists, delete
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # create engine object
    db_engine = create_engine("sqlite:///" + db_path)
    # connect to database
    con = db_engine.connect()
    con.close()

def get_funcs(funcs: dict, file_name: str, table_name: str) -> None:
    '''
    Read the given *.csv file, store the data inside the database and store the functions in a dict
    
    Args:
        functions dict: Dictionary to store the functions object
        file_name str: The file name of the csv-file with file extension
        table_name str: The name of the SQL table
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

        # store raw data in database
        db_path = os.path.join(working_dir, "PyTask.db")
        engine = create_engine('sqlite:///' + db_path)
        data.to_sql(table_name, engine, index=False, if_exists="replace")

        for y_axes in list(data.columns.values[1:]):
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
            
            # create a new function object to store
            if file_name == "train.csv":
                funcs[y_axes] = functions.TrainingFunction(data[["x", y_axes]], coefficients)
            elif file_name == "ideal.csv":
                funcs[y_axes] = functions.Function(data[["x", y_axes]], coefficients)
            else:
                raise Exception("The function for the test data is not neccessary.")
    except Exception as error:
        # handle the exception
        print(error)
        exit()

def set_ideal_functions():
    '''
    Find the ideal function for every training function

    Args:
        train_functions dict: A dictionary with the training functions
        ideal_functions dict: A dictionary with the ideal functions
    '''    
    global train_functions, ideal_functions

    try:
        # check if there are 50 ideal functions as expected in the dict --> raise exception if not
        if len(ideal_functions) < 50:
            raise Exception("There are not enough training functions")
        elif len(ideal_functions) > 50:
            raise Exception("There are too much training functions")
        
        # list with the ideal function nos
        ideal_func_nos = []
        # iterate through all training functions
        for train_func in train_functions:
            # iterate through ideal functions
            for ideal_func in ideal_functions:
                # check, if the current ideal function fits better to the training function as the actual one
                train_functions[train_func].check_ideal_function(ideal_functions[ideal_func], ideal_func)
            # store ideal no
            ideal_func_nos.append(train_functions[train_func].ideal_no)

        # if the ideal function was not mapped to a training function, delete it from the dict
        ideal_functions = {key: value for key, value in ideal_functions.items() if key in ideal_func_nos}
    except Exception as error:
        # handle the exception
        print(error)
        exit()

def map_test_function(train_functions: dict):
    '''
    Map the test function to the training functions

    Args:
        train_functions dict: A dictionary with the training functions
    '''
    try:
        # check if there are 4 training functions as expected in the dict --> raise exception if not
        if len(train_functions) < 4:
            raise Exception("There are not enough training functions")
        elif len(train_functions) > 4:
            raise Exception("There are too much training functions")
        
        # read test data
        complete_path = os.path.join(dataset_dir, "test.csv")
        test_function = pd.read_csv(complete_path, sep=",")
        test_function["delta y"] = None
        test_function["No of ideal function"] = None
        
        # add y* of the training functions
        data = pd.merge(test_function[["x", "y"]], train_functions["y1"].data[["x", "y*"]], on="x", how="inner", suffixes=("", "1"))
        data = pd.merge(data, train_functions["y2"].data[["x", "y*"]], on="x", how="inner", suffixes=["", "2"])
        data = pd.merge(data, train_functions["y3"].data[["x", "y*"]], on="x", how="inner", suffixes=["", "3"])
        data = pd.merge(data, train_functions["y4"].data[["x", "y*"]], on="x", how="inner", suffixes=["", "4"])
        # renaming columns
        data.columns = ["x", "y", "y1", "y2", "y3", "y4"]
        
        for index, row in data.iterrows():
            delta_y, ideal_no = "", ""
            already_mapped = False
            if abs(row["y"] - row["y1"]) < math.sqrt(2):
                delta_y = str(abs(row["y"] - row["y1"]))
                ideal_no = train_functions["y1"].ideal_no
                already_mapped = True
            if abs(row["y"] - row["y2"]) < math.sqrt(2):
                delta_y += (" / " if already_mapped else "") + str(abs(row["y"] - row["y2"]))
                ideal_no += (" / " if already_mapped else "") + train_functions["y2"].ideal_no
                already_mapped = True
            if abs(row["y"] - row["y3"]) < math.sqrt(2):
                delta_y += (" / " if already_mapped else "") + str(abs(row["y"] - row["y3"]))
                ideal_no += (" / " if already_mapped else "") + train_functions["y3"].ideal_no
                already_mapped = True
            if abs(row["y"] - row["y4"]) < math.sqrt(2):
                delta_y += (" / " if already_mapped else "") + str(abs(row["y"] - row["y4"]))
                ideal_no += (" / " if already_mapped else "") + train_functions["y4"].ideal_no
            
            # add values to data frame
            test_function.loc[test_function["x"] == row["x"], "delta y"] = delta_y
            test_function.loc[test_function["x"] == row["x"], "No of ideal function"] = ideal_no
        
        # store raw data in database
        db_path = os.path.join(working_dir, "PyTask.db")
        engine = create_engine('sqlite:///' + db_path)
        test_function.to_sql("TestFunction", engine, index=False, if_exists="replace")
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