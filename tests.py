import io
import main
import os
import unittest
import unittest.mock

from colorama import Fore
from datetime import datetime

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
            expected_output = str(Fore.RED) + str("01/01/1900 00:00:00 get_funcs [Error]: The give file can't be found in the dataset directory!") + str(Fore.WHITE) + "\n"

            main.get_funcs({}, "train.csv", "TrainingFunctions")
            # check if the formatted output equals the expected output
            self.assertEqual(mock_stdout.getvalue(), expected_output)

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

if __name__ == "__main__":
    unittest.main(exit=False)