import opt
import os

def PoseTransferDataPrep(opts):
    # prepare the data for Pose-Transfer module
    if not os.path.isdir(os.path.join(opts.dataroot, "bounding_box_trainK")):
        # run bash script to call virtualenv
        os.system("./script/makeposemap.sh")

def PoseTransferMakeImage(opts):
    os.system("./script/makeposetransfer.sh")


if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    opts = opt.parser().parse_args()
    PoseTransferDataPrep(opts)

    # Step: randomly generate a csv file that lists where the generated images are from