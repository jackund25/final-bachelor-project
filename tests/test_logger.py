# test_logger.py
from src.utils.logger import setup_logger

logger = setup_logger('test')
logger.info("✓ Logger working!")
logger.warning("This is a warning")
logger.error("This is an error")