import logging
import os
from datetime import datetime

LOG_FILE_NAME=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
#Joins the Current Working Directory, logs and log file name
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE_NAME)

#Create a directory and if already exists then do nothing
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE_NAME)

#Configuring the logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
