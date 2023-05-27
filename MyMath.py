class LinearFunction:    
    y_intercept = None
    slope = None

    def __init__(self, a: float, b: float):
        '''
        Initalize a new linear function of the form y = a + b * x

        :param float a: The y intercept of the function
        :param float b: The slope of the function
        '''
        self.y_intercept = a
        self.slope = b