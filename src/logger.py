import logging # use to log messages to a file or console
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_dir = os.path.join(os.getcwd(), "logs")

os.makedirs(logs_dir, exist_ok=True) # ensure logs directory exists

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# define the logging configuration
# this will tell the logging module how to log messages, including the log file location, log level, and log message format
logging.basicConfig(
    # file name
    filename=LOG_FILE_PATH,
    # logging level
    level=logging.INFO,
    # log message format
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


# if __name__ == "__main__":
#     logging.info("Logging has started")