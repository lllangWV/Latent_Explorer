import os
import logging

def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir,'package.log'), 
                        # level=logging.INFO, 
                        # level=logging.NOTSET,
                        level=logging.DEBUG,
                        # level=logging.WARNING,
                        # level=logging.ERROR, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - Line: %(lineno)d')

    logger = logging.getLogger('matattnet')  # define globally (used in train.py, val.py, detect.py, etc.)
    return logger