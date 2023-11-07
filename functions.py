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

class Function:
    '''
    A basic class to store functions and its parameters
    '''

    data: pd.DataFrame
    '''
    A DataFrame for the raw values
    '''
    coefficients: list = []
    '''
    A list of the coefficients of the function
    '''

    def __init__(self, raw_data: pd.DataFrame, coefficients: list):
        '''
        Initalize a new function

        Args:
            raw_data pd.DataFrame: A DataFrame containing the raw data
            coeffiecents list: The coefficients of the functions            
        '''
        # set raw data (important to copy. If not, the calculation of the predicted y-values throws a warning)
        self.data = raw_data.copy()
        # rename column to only y
        self.data.columns.values[1] = "y"
        # add coefficents to the list
        self.coefficients = coefficients
        # calculate all predicted values
        self.__set_predicted_values()
    
    def __set_predicted_values(self):
        '''
        Get all predicted y-values
        '''
        # set predicted values
        self.data["y*"] = self.data.apply(lambda row: sum(coefficient * (row["x"] ** (len(self.coefficients) - 1 - i)) for i, coefficient in enumerate(self.coefficients)), axis=1)

    def __str__(self):
        '''
        return function as string
        '''
        # get the degree of the fuction
        degree = len(self.coefficients) - 1

        result = "y ="
        for i in range(degree, 1, -1):
            # add every power to the function, if coeffiecent is not less than 0.00
            coeffiecent = self.__FloatToString(self.coefficients[(i + 1) * -1])
            if coeffiecent[3:] != "0.00":
                result += self.__FloatToString(self.coefficients[(i + 1) * -1]) + "x" + self.__IntToSuperscript(i)
        
        # add x and slope, if it is not 0.00
        coeffiecent = self.__FloatToString(self.coefficients[-2])
        slope = self.__FloatToString(self.coefficients[-1])
        if coeffiecent[3:] != "0.00" and slope[3:] != "0.00":
            result += coeffiecent + "x" + slope
        elif coeffiecent[3:] != "0.00" and slope[3:] == "0.00":
            result += coeffiecent + "x"
        elif coeffiecent[3:] == "0.00" and slope[3:] != "0.00":
            result += slope

        # if first coeffiecent is positive, remove first sign
        if result.startswith("y = +"):
            result = "y =" + result[5:]
            
        return result
    
    def __FloatToString(self, number: float):
        '''
        Convert a float to a string with max. two decimal places

        Args:
            float: The float number to convert

        Returns:
            str: The given float as string
        '''
        return " + " + f"{number:.2f}" if number > 0 else " - " + f"{abs(number):.2f}"
    
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

class TrainingFunction(Function):
    '''
    A basic class for a training function
    '''
    
    ideal_no: str = None
    '''
    The number of the ideal function
    '''
    mapped_points: pd.DataFrame
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
        # add columns for the mapped test points
        self.mapped_points = pd.DataFrame(columns=["x", "y"])
        
    def check_ideal_function(self, ideal_func: Function, func_no: str):
        '''
        Compare the given function to ideal function and checks the squarred sum error.

        Args:
            ideal_func: Ideal Function to compare with
            func_no str: Number of the ideal function
        '''
        error = ((self.data["y*"] - ideal_func.data["y*"]) ** 2).sum()

        # if the error is smaller, update the ideal function
        if error < self.__min_error:
            self.ideal_no = func_no
            self.__min_error = error
    
    def add_test_point(self, x: float, y: float) -> None:
        '''
        Add a point from the test function to the DataFrame

        Args:
            x float: The x-value
            y float: The y-value
        '''

        self.mapped_points.loc[len(self.mapped_points.index)] = [x, y]