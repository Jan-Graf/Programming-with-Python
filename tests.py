import io
import main
import os
import unittest
import unittest.mock

from colorama import Fore
from datetime import datetime

class UnitTestPythonTask(unittest.TestCase):
    def format_stoud(self, stdout: str, color: str, length: int):
        '''
        Format the standard output of a log message

        Args:
            stdout str: The standard output as unformatted string
            color str: The color of the log message
            length int: The length of the printed messaged
        
        Returns:
            str: A string without color and timestamp
        '''
        # format the output (remove color and timestamp at the beginning as well as the color and linebreak at the end)
        start_index = len(color) + 20
        return stdout[start_index : start_index + length]

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
        # define a variable with the expected output (without timestamp)
        expected_output = "create_database [Warn]: The database file couldn't be updated, because it is in use. The database won't be updated!"

        main.create_database()
        # check if the formatted output equals the expected output
        self.assertEqual(self.format_stoud(mock_stdout.getvalue(), Fore.YELLOW, len(expected_output)), expected_output)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_test_function(self, mock_stdout):
        '''
        Checks if the correct log is printed, if the function for the test function should be calculated
        '''
        # define a variable with the expected output (without timestamp)
        expected_output = "get_funcs [Error]: The function for the test data is not neccessary."

        main.get_funcs({}, "test.csv", "TestFunction")
        # check if the formatted output equals the expected output
        self.assertEqual(self.format_stoud(mock_stdout.getvalue(), Fore.RED, len(expected_output)), expected_output)
        
    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_wrong_file_name(self, mock_stdout):
        '''
        Checks if the correct log is printed, if the a wrong file name is given
        '''
        # define a variable with the expected output (without timestamp)
        expected_output = "get_funcs [Error]: Given a completly wrong / empty file or a directorys."

        main.get_funcs({}, "some_name.csv", "MyTable")
        # check if the formatted output equals the expected output
        self.assertEqual(self.format_stoud(mock_stdout.getvalue(), Fore.RED, len(expected_output)), expected_output)
        
    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_empty_file_name(self, mock_stdout):
        '''
        Checks if the correct log is printed, if the an empty file name is given
        '''
        # define a variable with the expected output (without timestamp)
        expected_output = "get_funcs [Error]: Given a completly wrong / empty file or a directorys."

        main.get_funcs({}, "", "MyTable")
        # check if the formatted output equals the expected output
        self.assertEqual(self.format_stoud(mock_stdout.getvalue(), Fore.RED, len(expected_output)), expected_output)
            
    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    @unittest.mock.patch('main.os.path.exists', return_value=False)
    def test_check_file_existing(self, mock_exist, mock_stdout):
        '''
        Checks if the correct log is printed, if the a csv file doesn't exist
        '''
        
        # define a variable with the expected output (without timestamp)
        expected_output = "get_funcs [Error]: The give file can't be found in the dataset directory!"

        main.get_funcs({}, "train.csv", "TrainingFunctions")
        # check if the formatted output equals the expected output
        self.assertEqual(self.format_stoud(mock_stdout.getvalue(), Fore.RED, len(expected_output)), expected_output)

if __name__ == "__main__":
    unittest.main(exit=False)