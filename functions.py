import math

class linear_function:
    '''
    The value for the y intercept
    '''    
    y_intercept: float = None
    '''
    The slope of the function
    '''
    slope: float = None

    def __init__(self, a: float, b: float):
        '''
        Initalize a new linear function of the form y = a + b * x

        :param float a: The y intercept of the function
        :param float b: The slope of the function
        '''
        self.y_intercept = a
        self.slope = b

    def __str__(self):
        # return function in the form: y = a + b * x
        return "y = " + str(self.y_intercept) + " + " + str(self.slope) + " * x"

class train_function(linear_function):
    '''
    The ideal function for the training function
    '''
    ideal_function: linear_function
    '''
    The mapped points from the test set
    '''
    mapped_points = []

    def __init__(self, a: float, b: float):
        super().__init__(a, b)
        
    def check_ideal_function(self, ideal_func: linear_function) -> bool:
        '''
        Compare the given function to ideal function. If the given function fits besser to the function itself, the ideal function gets updated.
        '''
        # compare absolut deviations of slopes (given ideal function and actual ideal function)
        # if the slope of the given function is less than the actual ideal function...
        if abs(self.slope - ideal_func.slope) < abs(self.slope - self.ideal_function.slope):
            # if the y intercept is less than the actual ideal function, update the ideal function
            if abs(self.y_intercept - ideal_func.y_intercept) < abs(self.y_intercept - self.ideal_function.y_intercept):
                self.ideal_function = ideal_func
            # if the y intercept is greater but the slope is lower than the actual ideal function, check mapping criterion
            # if the criterion is matched, update ideal function, because slope has more impact than the y intercept
            elif abs(self.y_intercept - ideal_func.y_intercept) < math.sqrt(2):
                self.ideal_function = ideal_func
