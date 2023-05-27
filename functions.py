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