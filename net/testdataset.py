from opt import TestOptions
import os

if __name__ == "__main__":
    opts = TestOptions().parse_args()
    os.system("./script/testdataset.sh \"{0}\" {1}".format(os.path.join(opts.dataroot, ".."), '+'.join(opts.use_data)))