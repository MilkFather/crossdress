import argparse

# Python 3 script

class MakeOptions(argparse.ArgumentParser):

    def __init__(self):
        super(MakeOptions, self).__init__()

        self.add_argument("--dataroot", required=True)
        self.add_argument("--seed", required=False)
        self.add_argument("--makesize", required=False, default=64, type=int)
        self.add_argument("--use_existing_guide", action="store_true")
        self.add_argument("--phases", required=True, type=str, nargs='+')

class TestOptions(argparse.ArgumentParser):

    def __init__(self):
        super(TestOptions, self).__init__()

        self.add_argument("--dataroot", required=True)
        self.add_argument("--use_data", required=True, nargs='+')
        #self.add_argument("--use_resnet", action="store_true")
        self.add_argument("--sample_size", required=False, type=int, default=20000)