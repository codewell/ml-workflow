from tqdm import tqdm
from .constants import TQDM_OUTFILE


def tqdm_print(string):
    tqdm.write(str(string), file=TQDM_OUTFILE)
