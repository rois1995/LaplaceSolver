import logging
import sys

def setup_logging(verbose=True):
    # Create loggers
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    # File handler for normal output
    fh_out = logging.FileHandler('output.log', mode='w')
    fh_out.setLevel(logging.INFO)
    fh_out.setFormatter(logging.Formatter('%(message)s'))

    # File handler for errors
    fh_err = logging.FileHandler('error.log', mode='w')
    fh_err.setLevel(logging.ERROR)
    fh_err.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh_out)
    logger.addHandler(fh_err)

    # Console handler (if verbose)
    if verbose:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

        ch_err = logging.StreamHandler(sys.stderr)
        ch_err.setLevel(logging.ERROR)
        ch_err.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch_err)

    return logger