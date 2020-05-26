import argparse

# Python 3 script

def parser():
    parse = argparse.ArgumentParser()
    parse.add_argument("--dataroot", required=True)
    parse.add_argument("--seed", required=False)
    parse.add_argument("--makesize", required=False, default=64)

    return parse