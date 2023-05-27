class LinearFunction:
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