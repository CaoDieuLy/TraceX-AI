import sys
from loguru import logger
from .config import config

logger.remove()
logger.add(sys.stdout, level=config.log_level,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> - <level>{message}</level>")
logger.add("logs/app.log", rotation="10 MB", level="DEBUG")

__all__ = ["logger"]
