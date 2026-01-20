import logging
import os
from dotenv import load_dotenv
from shutil import get_terminal_size
from typing import Literal

def create_logger(
    name: str,
    level: Literal["debug", "info", "warning", "error", "critical"] = "info",
) -> logging.Logger:
    """Create a logging.Logger with nice formatting for terminal output.
    The specified logging level is set for the logger, but the actual level can be overridden
    by the LOG_LEVEL environment variable assigned to the log handler.

    Example:
    ```
        logger1 = create_logger("my_logger1", level="debug")
        logger1.info("This is an info message.") # Visible
    ```
    ```
        logger2 = create_logger("my_logger2", level="info")
        logger2.debug("This is a debug message.") # Not visible
    ```
    ```
        logger3 = create_logger("my_logger3", level="debug")
        logger3.debug("This is a debug message.") # Visible if LOG_LEVEL is set to "debug"
    ```

    Args:
        name (str): The name of the logger.
        level (Literal["debug", "info", "warning", "error", "critical"]): The logging level to set for the logger.

    Returns:
        logging.Logger: The configured logger instance.
    """
    
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logging_level = levels.get(level.lower())
    if logging_level is None:
        raise ValueError(f"Invalid logging level. Choose from: {list(levels.keys())}")    

    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    logger.handlers.clear()

    # compute terminal width for formatting
    columns = get_terminal_size().columns
    class DynamicFormatter(logging.Formatter):
        def format(self, record):
            info = f"({record.filename}:{record.lineno})"
            # 32 = estimated length of level + timestamp + spaces
            msg_width = columns - len(info) - 32
            if msg_width < 0:
                msg_width = 0
            self._style._fmt = "%(levelname)-8s %(asctime)s - %(message)-" + str(msg_width) + "s " + info
            return super().format(record)

    handler = logging.StreamHandler()
    handler.setFormatter(DynamicFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
    
    if not "LOG_LEVEL" in os.environ:
        load_dotenv()
    desired_log_level = os.getenv("LOG_LEVEL", "info").lower()
    handler.setLevel(levels.get(desired_log_level, logging.INFO))

    logger.addHandler(handler)
    logger.propagate = False
    return logger
