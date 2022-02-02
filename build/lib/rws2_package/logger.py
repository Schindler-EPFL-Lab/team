import logging

logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(module)s.py-%(funcName)s -> %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

log = logging.getLogger(__name__)
