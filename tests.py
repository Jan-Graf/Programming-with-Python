import functions
import io
import main
import numpy as np
import os
import pandas as pd
import unittest
import unittest.mock

from colorama import Fore
from datetime import datetime
from sqlalchemy import create_engine, inspect

class UnitTestPythonTask(unittest.TestCase):
    def test_database_creation(self):
        '''
        Check, if the database file gets created
        '''
        main.create_database()
        self.assertEqual(os.path.exists(os.path.join(main.working_dir, "PyTask.db")), True)
    
    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    @unittest.mock.patch('main.os.remove', side_effect=PermissionError("PermissionError"))
    def test_file_in_use(self, mock_remove, mock_stdout):
        '''
        Check, if the correct message is printed, if the db file is in use
        '''
        with unittest.mock.patch('main.datetime') as mock_datetime:
            # mock datetime now to return always the 1st January 1900
            mock_datetime.now.return_value = datetime(1900, 1, 1, 0, 0, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            # define a variable with the expected output
            expected_output = str(Fore.YELLOW) + str("01/01/1900 00:00:00 create_database [Warn]: The database file couldn't be updated, because it is in use. The database won't be updated!") + str(Fore.WHITE) + "\n"

            main.create_database()
            # check if the formatted output equals the expected output
            self.assertEqual(mock_stdout.getvalue(), expected_output)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_test_function(self, mock_stdout):
        '''
        Checks if the correct log is printed, if the function for the test function should be calculated
        '''
        with unittest.mock.patch('main.datetime') as mock_datetime:
            # mock datetime now to return always the 1st January 1900
            mock_datetime.now.return_value = datetime(1900, 1, 1, 0, 0, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # define a variable with the expected output
            expected_output = str(Fore.RED) + str("01/01/1900 00:00:00 get_funcs [Error]: The function for the test data is not neccessary.") + str(Fore.WHITE) + "\n"

            main.get_funcs({}, "test.csv", "TestFunction")
            # check if the formatted output equals the expected output
            self.assertEqual(mock_stdout.getvalue(), expected_output)
        
    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_wrong_file_name(self, mock_stdout):
        '''
        Checks if the correct log is printed, if the a wrong file name is given
        '''
        with unittest.mock.patch('main.datetime') as mock_datetime:
            # mock datetime now to return always the 1st January 1900
            mock_datetime.now.return_value = datetime(1900, 1, 1, 0, 0, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            # define a variable with the expected output
            expected_output = str(Fore.RED) + str("01/01/1900 00:00:00 get_funcs [Error]: Given a completly wrong / empty file or a directorys.") + str(Fore.WHITE) + "\n"

            main.get_funcs({}, "some_name.csv", "MyTable")
            # check if the formatted output equals the expected output
            self.assertEqual(mock_stdout.getvalue(), expected_output)
        
    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_empty_file_name(self, mock_stdout):
        '''
        Checks if the correct log is printed, if the an empty file name is given
        '''
        with unittest.mock.patch('main.datetime') as mock_datetime:
            # mock datetime now to return always the 1st January 1900
            mock_datetime.now.return_value = datetime(1900, 1, 1, 0, 0, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
            # define a variable with the expected output
            expected_output = str(Fore.RED) + str("01/01/1900 00:00:00 get_funcs [Error]: Given a completly wrong / empty file or a directorys.") + str(Fore.WHITE) + "\n"

            main.get_funcs({}, "", "MyTable")
            # check if the formatted output equals the expected output
            self.assertEqual(mock_stdout.getvalue(), expected_output)
    
    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    @unittest.mock.patch('main.os.path.exists', return_value=False)
    def test_check_file_existing(self, mock_exist, mock_stdout):
        '''
        Checks if the correct log is printed, if the a csv file doesn't exist
        '''
        with unittest.mock.patch('main.datetime') as mock_datetime:
            # mock datetime now to return always the 1st January 1900
            mock_datetime.now.return_value = datetime(1900, 1, 1, 0, 0, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            # define a variable with the expected output
            expected_output = str(Fore.RED) + str("01/01/1900 00:00:00 get_funcs [Error]: The given file can't be found in the dataset directory!") + str(Fore.WHITE) + "\n"

            main.get_funcs({}, "train.csv", "TrainingFunctions")
            # check if the formatted output equals the expected output
            self.assertEqual(mock_stdout.getvalue(), expected_output)
    
    def test_check_function_dicts(self):
        '''
        Check, if 4 training and 50 ideal functions are created
        '''
        # load and generate functions
        main.get_funcs(main.train_functions, "train.csv", "TrainingFunction")
        main.get_funcs(main.ideal_functions, "ideal.csv", "IdealFunctions")
        # check the length of the dicts
        self.assertEqual(len(main.train_functions), 4)
        self.assertEqual(len(main.ideal_functions), 50)
    
    def test_check_table_creation(self):
        '''
        Check, if the tables are created in the datatabase
        '''
        # generate functions and update database
        main.update_database = True
        main.get_funcs(main.train_functions, "train.csv", "TrainingFunctions")
        main.get_funcs(main.ideal_functions, "ideal.csv", "IdealFunctions")

        # connect to database
        db_engine = create_engine("sqlite:///" + os.path.join(main.working_dir, "PyTask.db"))        
        # inspect database to check if the tables exists --> store result in variable
        inspector = inspect(db_engine)
        training_table_exists = inspector.has_table("TrainingFunctions")
        ideal_table_exists = inspector.has_table("IdealFunctions")
        # release database
        db_engine.dispose()

        self.assertEqual(training_table_exists, True)
        self.assertEqual(ideal_table_exists, True)

    def test_set_ideal_functions(self):
        '''
        Check, if the ideal functions are corretly mapped
        '''
        # create raw data for a dummy training function
        x = np.arange(-20, 20, step=0.1)
        y_train = 2 * x**2 + 5 * x - 2
        # create dummy values for ideal functions
        y1 = -3 * x
        y2 = 7 * x**3 + x - 5
        y3 = 2 * x**2 + 4.5 * x - 3
        y4 = x + 9
        y5 = 5 * x**2 + 2 * x - 2

        # clear training functions and add a new dummy funtion
        main.train_functions.clear()
        main.train_functions["y1"] = functions.TrainingFunction(pd.DataFrame(data={"x": x, "y1": y_train}), [2, 5, -2])
        # clear ideal functions and add a new dummy funtions
        main.ideal_functions.clear()
        main.ideal_functions["y1"] = functions.Function(pd.DataFrame(data={"x": x, "y1": y1}), [-3, 0])
        main.ideal_functions["y2"] = functions.Function(pd.DataFrame(data={"x": x, "y2": y2}), [7, 0, 0, -5])
        main.ideal_functions["y3"] = functions.Function(pd.DataFrame(data={"x": x, "y3": y3}), [2, 4.5, -3])
        main.ideal_functions["y4"] = functions.Function(pd.DataFrame(data={"x": x, "y4": y4}), [1, 9])
        main.ideal_functions["y5"] = functions.Function(pd.DataFrame(data={"x": x, "y5": y5}), [5, 2, -2])

        # map ideal function to the training function and check, if mapping is as expected
        main.set_ideal_functions()
        self.assertEqual(main.train_functions["y1"].ideal_no, "y3")

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_print_log(self, mock_stdout):
        '''
        Checks if the print_log formats the log message correctly
        '''
        with unittest.mock.patch('main.datetime') as mock_datetime:
            # mock datetime now to return always the 1st January 1900
            mock_datetime.now.return_value = datetime(1900, 1, 1, 0, 0, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            # print a sample log
            main.print_log(main.MessageType.Info, "test_print_log", "This is a sample log!")
            # check log with expected output
            self.assertEqual(mock_stdout.getvalue(), str(Fore.WHITE) + str("01/01/1900 00:00:00 test_print_log [Info]: This is a sample log!") + str(Fore.WHITE) + "\n")

    def test_set_predicted_values(self):
        '''
        Checks if the y-values are predicted correctly for a sample function
        '''        
        # Assumption: valid input for the function
        raw_data = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2.5, 3.5, 6.5, 7.5]})
        coefficients = [2, 0]  # y = 2x +/- 0.5

        # create a function
        test_function = functions.Function(raw_data, coefficients)

        # check if the expected values y = 2x are predicted
        expected_predicted_values = [2, 4, 6, 8]
        self.assertListEqual(list(test_function.data["y*"]), expected_predicted_values)
        
    def test_set_predicted_negative_values(self):
        '''
        Checks if the y-values are predicted correctly for a negative sample function
        '''        
        # Assumption: valid input for the function
        raw_data = pd.DataFrame({"x": [1, 2, 3, 4], "y": [-2, -11, -12, -21]})
        coefficients = [-5, 1]  # y = -5x + 1 +/- 2

        # create a function
        test_function = functions.Function(raw_data, coefficients)

        # check if the expected values y = -5x + 1 are predicted
        expected_predicted_values = [-4, -9, -14, -19]
        self.assertListEqual(list(test_function.data["y*"]), expected_predicted_values)

if __name__ == "__main__":
    unittest.main(exit=False)