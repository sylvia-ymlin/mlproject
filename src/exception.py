import sys # manipulate Python runtime environment
# if we want to use our custom logger, we need to import logging from logger.py
from src.logger import logging 

def error_message_details(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info() # unpack exception info, exc_tb is traceback object
    '''
    from the traceback object, we can get the filename and line number where the exception occurred
    '''
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_messgae = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,
        line_number,
        str(error)
    )

    return error_messgae

def CustomException(Exception): # self define exception class, need to inherit from Exception class
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message) # call the constructor of the parent Exception class to initialize the exception with the provided error message
        # then we use our custom function to get detailed error message (original error message with file name and line number)
        self.error_message = error_message_details(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message

# if __name__ == "__main__":
#     try:
#         a = 1 / 0
#     except Exception as e:
#         logging.info("Divide by zero exception occurred")
#         raise CustomException(e, sys)