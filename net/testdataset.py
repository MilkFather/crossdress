from opt import TestOptions
import os

if __name__ == "__main__":
    opts = TestOptions().parse_args()
    if opts.use_network == "abd":
        #if not opts.use_resnet:
        os.system("./script/testdatasetabd.sh \"{0}\" {1} {2}".format(os.path.join("..", opts.dataroot), '+'.join(opts.use_data), str(opts.sample_size)))
        #else:
            #os.system("./script/validatedataset.sh \"{0}\" {1}".format(os.path.join("..", opts.dataroot), '+'.join(opts.use_data)))
    elif opts.use_network == "dg":
        os.system("./script/testdatasetdg.sh")