import logging
import sys

# Structured logger
logger = logging.getLogger("ml_inference_api")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
handler.setFormatter(formatter)
logger.addHandler(handler)