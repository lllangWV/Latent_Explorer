# Important directory paths
from latentexp.utils.config import FILE, PKG_DIR, ROOT, LOG_DIR, DATA_DIR

# Impot config settings
from latentexp.utils.config import CONFIG, MP_API_KEY, N_CORES

# Database paths
from latentexp.utils.config import (MP_DIR, DB_DIR, 
                                    DB_CALC_DIR, GLOBAL_PROP_FILE, N_CORES)

# Other important variables
from latentexp.utils.log_config import setup_logging
from latentexp.utils.timing import Timer, timeit

# Initialize logger
LOGGER = setup_logging(log_dir=LOG_DIR)