import math
import pandas as pd

# dict with unicodes for the numbers in superscript
superscript = {0: '\u2070',
               1: '\u00b9',
               2: '\u00b2',
               3: '\u00b3',
               4: '\u2074',
               5: '\u2075',
               6: '\u2076',
               7: '\u2077',
               8: '\u2078',
               9: '\u2079'
               }

class function:
    '''
    A simple class to store functions and its parameters
    '''

    raw_data: pd.DataFrame
    '''
    An empty DataFrame for the raw values
    '''
    coefficients: list = []
    '''
    A list of the coefficients of the function
    '''
    predicted_data: pd.DataFrame
    '''
    A data frame for the predicted values of the function
    '''

    def __init__(self, raw_data: pd.DataFrame, coefficients: list):
        '''
        Initalize a new function

        Args:
            raw_data pd.DataFrame: A DataFrame containing the raw data
            coeffiecents list: The coefficients of the functions            
        '''
        # set raw data
        self.raw_data = raw_data
        # rename column to only y
        self.raw_data.columns.values[1] = "y"
        # add coefficents to the list
        self.coefficients = coefficients
    
    def predict_values(self):
        '''
        Get all predicted y-values
        '''
        # calcualte the y-values
        y_values = []
        for x in self.raw_data["x"]:
            y_values.append([x, self._get_predicted_y(x)])
        # set predicted values
        self.predicted_data = pd.DataFrame(data=y_values, columns=["x", "y"])

    def _get_predicted_y(self, x_value: float) -> float:
        '''
        Calculate the y-value for a given x-value

        Args:
            x_value float: The x-value to get the y-value for

        Return:
            float: The value for y at x
        '''

        predicted_y = 0.0
        # enumerate through all coeffiecents and sum it up
        for power, coefficent in enumerate(self.coefficients):
            predicted_y += coefficent * (x_value ** (len(self.coefficients) - 1 - power))
        return predicted_y
    
    def __str__(self):
        '''
        return function as string
        '''
        # get the degree of the fuction
        degree = len(self.coefficients) - 1

        result = "y = "
        for i in range(degree, 1, -1):
            # add every power to the function
            result += self.__FloatToString(self.coefficients[(i + 1) * -1]) + "x" + self.__IntToSuperscript(i) + (" + " if self.coefficients[i * -1] >= 0 else " - ")
        # add x and slope to function
        result += self.__FloatToString(self.coefficients[-2]) + "x" + (" + " if self.coefficients[-1] >= 0 else " - ")
        result += self.__FloatToString(self.coefficients[-1])
            
        return result
    
    def __FloatToString(self, number: float):
        '''
        Convert a float to a string with max. two decimal places and no sign

        Args:
            float: The float number to convert

        Returns:
            str: The given float as string
        '''
        return f"{abs(number):.2f}"
    
    def __IntToSuperscript(self, number: int):
        '''
        Convert a number to superscript

        Args:
            int: The int number to convert
        
        Returns:
            str: A string with the given number as unicode superscript
        '''

        result = ""
        # split the number in single integers and get superscript unicode
        for single_int in list(map(int, str(number))):
            result += superscript[single_int]
        return result

class train_function(function):
    '''
    A class for the training functions to add ideal functions annd mapped points
    '''
    
    ideal_no: str = None
    '''
    The number of the ideal function
    '''
    predicted_data: pd.DataFrame
    '''
    DataFrame with the predicted values ideal function
    '''
    mapped_points: list = []
    '''
    The mapped points from the test set
    '''
    __min_error: float = float("inf")
    '''
    The minimum error of the ideal function
    '''

    def __init__(self, raw_data: pd.DataFrame, coefficents: list):
        '''
        Initalize a new training function
        '''
        super().__init__(raw_data, coefficents)
        self.mapped_points = []
        
    def check_ideal_function(self, ideal_func: function, func_no: str):
        '''
        Compare the given function to ideal function and checks the squarred sum error.

        Args:
            ideal_func: Ideal Function to compare with
            func_no str: Number of the ideal function
        '''
        error = 0
        # get summed square error
        for index, row in self.raw_data.iterrows():
            error += (row["y"] - ideal_func._get_predicted_y(row["x"])) ** 2

        # if the error is smaller, update the ideal function
        if error < self.__min_error:
            self.ideal_no = func_no
            self.__min_error = error

class mapping_point:
    '''
    Simple class for the mapped points
    '''
    
    x_value: float = None
    '''
    x-value of the point
    '''
    y_value: float = None
    '''
    y-value of the point
    '''
    delta_y: float = None
    '''
    The deviation of the raw y-value and the predicted y-value
    '''
    
    def __init__(self, x: float, y: float, delta: float) -> None:
        '''
        Initialize a new Mapping Point

        Args:
            x float: The x-value of the point
            y float: The y-value of the point
            delta float: The devation of the raw y-value and the predicted y-value
        '''
        # set attributes
        self.x_value = x
        self.y_value = y
        self.delta_y = delta
