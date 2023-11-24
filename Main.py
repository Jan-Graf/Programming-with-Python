import functions
import math
import numpy as np
import os
import pandas as pd
import plotly.graph_objs as go

from colorama import Fore
from datetime import datetime
from enum import Enum
from plotly.subplots import make_subplots
from sqlalchemy import create_engine

class MessageType(Enum):
    Info = 1
    Warn = 2
    Error = 3

# define dataset directory based on file location
working_dir = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(working_dir, "Dataset")

# variable to proof, if the database should be updated --> If an exception occurs, update variable
update_database = False

# define the train functions
train_functions = {}
# define the ideal functions
ideal_functions = {}
# define the test function
test_function = None

def main():
    global train_functions, ideal_functions

    create_database()

    get_funcs(train_functions, "train.csv", "TrainingFunctions")
    get_funcs(ideal_functions, "ideal.csv", "IdealFunctions")

    set_ideal_functions()

    map_test_function()

    visualize_functions()

def print_log(msg_type: MessageType, method: str, msg: str):
    '''
    Print a log message to the console

    Args:
        msg_type MessageType: The type of the logged message
        method str: The calling method
        msg str: The message that should be printed
    '''
    # get the current timestamp and generate log message
    log_msg = f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} {method} [{msg_type.name}]: {str(msg)}"
    # print log, color depends on msg_type
    if msg_type == MessageType.Info:
        print(Fore.WHITE + log_msg + Fore.WHITE)
    elif msg_type == MessageType.Warn:
        print(Fore.YELLOW + log_msg + Fore.WHITE)
    else:
        print(Fore.RED + log_msg + Fore.WHITE)

def create_database() -> None:
    '''
    Create an empty database file. If a file exists, clear the database

    Args:
        db_engine Engine: The engine object for the connection to the database
    '''

    global update_database
    try:
        db_path = os.path.join(working_dir, "PyTask.db")
        # if the db file already exists, delete
        if os.path.exists(db_path):
            os.remove(db_path)
            print_log(MessageType.Info, "create_database", "Successfully removed existing database file!")
        
        # create engine object
        db_engine = create_engine("sqlite:///" + db_path)
        # connect to database
        con = db_engine.connect()
        con.close()
        print_log(MessageType.Info, "create_database", "Successfully created the PyTask.db!")
        # after the creation of the file, allow update
        update_database = True
    except PermissionError as error:
        print_log(MessageType.Warn, "create_database", "The database file couldn't be updated, because it is in use. The database won't be updated!")
    except Exception as error:
        print_log(MessageType.Error, "create_database", error)

def get_funcs(funcs: dict, file_name: str, table_name: str) -> None:
    '''
    Read the given *.csv file, store the data inside the database and store the functions in a dict
    
    Args:
        functions dict: Dictionary to store the functions object
        file_name str: The file name of the csv-file with file extension
        table_name str: The name of the SQL table
    '''    
    
    global update_database

    try:
        if file_name == "test.csv" or file_name == "test":
            raise ValueError("The function for the test data is not neccessary.")
        elif file_name != "train.csv" and file_name != "train" and file_name != "ideal.csv" and file_name != "ideal":
            raise ValueError("Given a completly wrong / empty file or a directorys.")
        
        # store the complete path to the file in a variable
        complete_path = os.path.join(dataset_dir, file_name)
        # check file to be correct
        file_extension = os.path.splitext(file_name)[1]
        if file_extension == "":
            print_log(MessageType.Warn, "get_funcs", "Missing file extension. Added "".csv"" automatically!")
            complete_path += ".csv"
        if not os.path.exists(complete_path):
            raise FileNotFoundError("The given file can't be found in the dataset directory!")
        
        # read csv file and store it in a data frame
        data = pd.read_csv(complete_path, sep=",")
        print_log(MessageType.Info, "get_funcs", "Successfully read \"" + file_name + "\"!")
    except PermissionError as error:
        print_log(MessageType.Error, "get_funcs", "Cannot open the given csv file!")
        return
    except ValueError as error:
        print_log(MessageType.Error, "get_funcs", error)
        return
    except Exception as error:
        print_log(MessageType.Error, "get_funcs", error)
        return

    try:
        if update_database:
            # store raw data in database
            db_path = os.path.join(working_dir, "PyTask.db")
            engine = create_engine('sqlite:///' + db_path)
            data.to_sql(table_name, engine, index=False, if_exists="replace")
            engine.dispose()
    except PermissionError as error:
        update_database = False
        # log warning
        print_log(MessageType.Warn, "get_funcs", "The database file couldn't be updated, because it is in use. The database won't be updated!")
    except Exception as error:
        update_database = False
        # log exception
        print_log(MessageType.Error, "get_funcs", "The database file couldn't be updated, because it is in use. The database won't be updated!")
        print_log(MessageType.Error, "get_funcs", error)

    try:
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

                # use rounded residuals because if not, really big functions will be returned (degree of 36 and coeffiecents like 2e-46)
                if round(residuals[0], 3) < round(min_residuals, 3) and rank >= len(updated_coefficients) - 1:
                    min_residuals = residuals[0]
                    coefficients = updated_coefficients
                
            # create a new function object to store
            if file_name == "train.csv" or file_name == "train":
                funcs[y_axes] = functions.TrainingFunction(data[["x", y_axes]], coefficients)
            elif file_name == "ideal.csv" or file_name == "ideal":
                funcs[y_axes] = functions.Function(data[["x", y_axes]], coefficients)
                
            print_log(MessageType.Info, "get_funcs", "Successfully created new function " + y_axes + "!")
    except Exception as error:
        # log error
        print_log(MessageType.Error, "get_funcs", error)
        return
    
def set_ideal_functions():
    '''
    Find the ideal function for every training function
    '''
    global train_functions, ideal_functions

    try:
        # check if there are 4 training functions as expected in the dict --> raise exception if not
        if len(train_functions) < 4:
            print_log(MessageType.Warn, "set_ideal_functions", "There are not enough training functions")
        elif len(train_functions) > 4:
            print_log(MessageType.Warn, "set_ideal_functions", "There are too much training functions")
        # check if there are 50 ideal functions as expected in the dict --> raise exception if not
        if len(ideal_functions) < 50:
            print_log(MessageType.Warn, "set_ideal_functions", "There are not enough ideal functions in the dict...")
        elif len(ideal_functions) > 50:
            print_log(MessageType.Warn, "set_ideal_functions", "There are too much ideal functions in the dict...")
        
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
            # log mapping
            print_log(MessageType.Info, "set_ideal_function", "Mapped " + train_functions[train_func].ideal_no + " to " + train_func + "!")

        # if the ideal function was not mapped to a training function, delete it from the dict
        ideal_functions = {key: value for key, value in ideal_functions.items() if key in ideal_func_nos}
    except Exception as error:
        print_log(MessageType.Error, "set_ideal_function", error, True)

def map_test_function():
    '''
    Map the test function to the training functions
    '''
    
    global train_functions, test_function, update_database
    try:
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

        # try the mapping and store deltas
        data.apply(calculate_deltas, axis=1)
        print_log(MessageType.Info, "map_test_functions", "Successfully mapped test function points to train functions!")
    except Exception as error:
        print_log(MessageType.Error, "map_test_functions", error)

    try:
        if update_database:
            # store raw data in database
            db_path = os.path.join(working_dir, "PyTask.db")
            engine = create_engine('sqlite:///' + db_path)
            test_function.to_sql("TestFunction", engine, index=False, if_exists="replace")    
    except PermissionError as error:
        update_database = False
        # log warning
        print_log(MessageType.Warn, "map_test_functions", "The database file couldn't be updated, because it is in use. The database won't be updated!")
    except Exception as error:
        update_database = False
        # log exception
        print_log(MessageType.Error, "map_test_functions", "The database file couldn't be updated, because it is in use. The database won't be updated!")
        print_log(MessageType.Error, "map_test_functions", error)

def calculate_deltas(row: pd.Series):
    '''
    Calculate the y-deviation of the test function to the ideal functions

    Args:
        row pd.Series: A row of a DataFrame containing the x and y values to calculate the deviation
    '''

    global train_functions, test_function
    try:
        # define the treshold of sqrt(2) and two variables for the deviation and the ideal no
        treshold = math.sqrt(2)
        delta_y, ideal_no = None, None

        for col in ["y1", "y2", "y3", "y4"]:
            # get current difference
            difference = abs(row["y"] - row[col])
            # update variables if the difference is smaller than the treshold
            if difference < treshold:
                delta_y = str(difference) if delta_y is None else delta_y + " / " + str(difference)
                ideal_no = col if ideal_no is None else ideal_no + " / " + col
                train_functions[col].add_test_point(row["x"], row["y"])

        # set value in the test_function DataFrame
        test_function.loc[row.name, "delta y"] = delta_y
        test_function.loc[row.name, "No of ideal function"] = ideal_no
    except Exception as error:
        print_log(MessageType.Error, "calculate deltas", error)
        

def visualize_functions():
    try:
        print_log(MessageType.Info, "visualize_functions", "Starting the visualization of the graphs...")
        # create a figure object
        fig = make_subplots(rows=2, cols=2, shared_xaxes=False, shared_yaxes=False, horizontal_spacing=0.1, vertical_spacing=0.1, subplot_titles=['Training Function y1', 'Training Function y2', 'Training Function y3', 'Training Function y4'])
        
        row_index, col_index = 1, 1
        index_mapping = {"y1": (1, 1, "blue", "royalblue", "deepskyblue"), "y2": (1, 2, "red", "tomato", "orangered"), "y3": (2, 1, "green", "limegreen", "forestgreen"), "y4": (2, 2, "purple", "mediumorchid", "violet")}
        try:            
            # iterate through all training functions
            for train_func in train_functions:
                # get row and col index of the function
                row_index, col_index, train_color, ideal_color, test_color = index_mapping[train_func]
                
                # draw graph of the raw data
                fig.add_trace(
                    go.Scatter(x=train_functions[train_func].data["x"], y=train_functions[train_func].data["y"], mode="lines", line=dict(color=train_color), name=train_func),
                    row=row_index,
                    col=col_index
                )

                # define ideal values
                ideal_x = ideal_functions[train_functions[train_func].ideal_no].data["x"]
                ideal_y = ideal_functions[train_functions[train_func].ideal_no].data["y*"]
                ideal_function = str(ideal_functions[train_functions[train_func].ideal_no])
                
                # add ideal function
                fig.add_trace(
                    go.Scatter(x=ideal_x, y=ideal_y, mode="lines", line=dict(color=ideal_color), name="Ideal Function for "+ train_func),
                    row=row_index,
                    col=col_index
                )        

                # add mapped test points
                fig.add_trace(
                    go.Scatter(x=train_functions[train_func].mapped_points["x"], y=train_functions[train_func].mapped_points["y"], mode="markers", marker=dict(color=test_color), name="Mapped points (" + train_func +")"),
                    row=row_index,
                    col=col_index
                )

                # set coordinates        
                x = -17.5 if train_functions[train_func].data.at[(train_functions[train_func].data["y"].idxmax()), "x"] < -19 else -20
                y = max(train_functions[train_func].data["y"])
                fig.add_annotation(
                    go.layout.Annotation(text=ideal_function, font=dict(size=14), x=x, y=y, xanchor="left", yanchor="middle", xref="paper", yref="paper", showarrow=False),
                    row=row_index,
                    col=col_index
                )
        except Exception as error:
            print_log(MessageType.Error, "calculate deltas", error)
            print_log(MessageType.Warn, "calculate deltas", "Cleared the corresponding plot!")
            # clear traces in the subplot
            fig.update_traces(go.Scatter(), row=row_index, col=col_index)
        
        fig.show()
    except Exception as error:
        print_log(MessageType.Error, "calculate deltas", error)

if __name__ == "__main__":
    main()