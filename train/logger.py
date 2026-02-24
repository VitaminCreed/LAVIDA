import logging
import os
from datetime import datetime

def get_logger(filename, verbosity=1, name=None):
    """Create and configure a logger instance.
    
    Args:
        filename: Path to the log file
        verbosity: Logging level (0: DEBUG, 1: INFO, 2: WARNING)
        name: Name of the logger
        
    Returns:
        Configured logger instance
    """
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "%(asctime)s - %(message)s",  # Log format: timestamp - message
        datefmt="%Y-%m-%d %H:%M"     # Timestamp format: YYYY-MM-DD HH:MM
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    
    # File handler for logging to file
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Stream handler for console output
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

def dict_to_message(data):
    """Convert a dictionary to formatted log message string.
    
    Args:
        data: Dictionary to convert
        
    Returns:
        Formatted string with key-value pairs separated by tabs
    """
    message_parts = []
    for key, value in data.items():
        if isinstance(value, float):  # Format floats to 4 decimal places
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = value
        message_parts.append(f"'{key}': {formatted_value}")
    
    message = "\t".join(message_parts)
    return message

class Logger:
    """Wrapper class for logging training and testing messages."""
    
    def __init__(self, log_path):
        """Initialize logger with specified log file path.
        
        Args:
            log_path: Path to the log file
        """
        self.log_path = log_path
        
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Initialize logger instance
        self.logger = get_logger(filename=log_path, verbosity=1)
    
    def log_train(self, epoch, total_epoch, message):
        """Log training progress message.
        
        Args:
            epoch: Current epoch number
            total_epoch: Total number of epochs
            message: Message to log (can be string or dict)
        """
        if isinstance(message, dict):
            message = dict_to_message(message)
        log_message = f'[Epoch {epoch}/{total_epoch}]{message}'
        self.logger.info(log_message)
    
    def log_test(self, message):
        """Log testing message.
        
        Args:
            message: Message to log (can be string or dict)
        """
        if isinstance(message, dict):
            message = dict_to_message(message)
        log_message = f'{message}'
        self.logger.info(log_message)