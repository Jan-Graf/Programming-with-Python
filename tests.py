import main
import os
import unittest
from unittest.mock import patch

class UnitTestPythonTask(unittest.TestCase):
    # def test_database_creation(self):
    #     '''
    #     Check, if the database file gets created
    #     '''
    #     main.create_database()
    #     self.assertEqual(os.path.exists(os.path.join(main.working_dir, "PyTask.db")), True)
    
    def test_check_test_function(self):
        '''
        Checks the file validation of the given .csv files
        '''
        with self.assertRaises(ValueError) as context:
            main.get_funcs({}, "test.csv", "TestFunction")
        
        self.assertEqual(str(context.exception),
            "The function for the test data is not neccessary.",
            "Fehlermeldung ist nicht korrekt")
    
    def test_check_file_name(self):
        '''
        Checks the file validation of the given .csv files
        '''
        self.assertRaises(Exception, main.get_funcs({}, "some_name.csv", "MyTable"))
        
    def test_check_file_name(self):
        '''
        Checks the file validation of the given .csv files
        '''
        self.assertRaises(Exception, main.get_funcs({}, "", ""))
    
    def test_check_file_existing(self):
        '''
        Checks the file validation of the given .csv files
        '''
        with patch("os.path.exists", return_value=True):
            self.assertRaises(FileNotFoundError, main.get_funcs({}, "nonexistent.csv", "MyTable"))


if __name__ == "__main__":
    unittest.main(exit=False)