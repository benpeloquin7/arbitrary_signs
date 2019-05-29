"""utils.py"""

import os


def make_safe_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
