import opt
import os
import random
import pandas as pd
import hashlib
from progress.bar import IncrementalBar

def DatasetGenerateGuide(opts):
    # Randomly generate a csv file that lists where the generated images are from
    if opts.seed is not None:
        print("Using random seed:", opts.seed)
        random.seed(opts.seed)
    
    # get a full list of train images in "market-1501/bounding_box_train/" directory
    # read the name of the images from "market-1501/market-annotation-train.csv"
    anno_file = pd.read_csv(os.path.join(opts.dataroot, "market-annotation-train.csv"), header=0, sep=":")
    file_ls = anno_file["name"]

    dataset_gen = pd.DataFrame(columns=["shape", "pose", "cloth", "phase1", "phase2"])

    print("Generating dataset generation guide...")
    bar = IncrementalBar(max=opts.makesize, message="%(index)d/%(max)d", suffix="%(elapsed_td)s/%(eta_td)s")

    for i in range(opts.makesize):
        shape_file = file_ls[random.randint(0, len(file_ls) - 1)]
        shape_person_id = shape_file.split("_")[0]
        
        # find a pose file.
        pose_file_should_done = False
        while not pose_file_should_done:
            pose_file = file_ls[random.randint(0, len(file_ls) - 1)]
            # ensure that the person in pose_file is different from that in shape_file
            pose_person_id = pose_file.split("_")[0]
            if pose_person_id != shape_person_id:
                pose_file_should_done = True

        # find a cloth file.
        cloth_file_should_done = False
        while not cloth_file_should_done:
            cloth_file = file_ls[random.randint(0, len(file_ls) - 1)]
            # ensure that the person in pose_file is different from that in shape_file
            cloth_person_id = cloth_file.split("_")[0]
            if cloth_person_id != shape_person_id:
                cloth_file_should_done = True

        phase1_filename = shape_person_id + "_gen_" + hashlib.md5((shape_file + pose_file + cloth_file + "phase1").encode()).hexdigest()[:6] + ".jpg"

        phase2_filename = shape_person_id + "_gen_" + hashlib.md5((shape_file + pose_file + cloth_file + "phase2").encode()).hexdigest()[:6] + ".jpg"

        dataset_gen.loc[len(dataset_gen)] = [shape_file, pose_file, cloth_file, phase1_filename, phase2_filename]
        bar.next()

    bar.finish()

    # write the output to a line in a new csv file
    dataset_gen.to_csv(os.path.join(opts.dataroot, "market-gen-guide.csv"), index=False)

def PoseTransferDataPrep(opts):
    # prepare the data for Pose-Transfer module
    if not os.path.isdir(os.path.join(opts.dataroot, "bounding_box_trainK")):
        print("Calling external script to generate Pose Map")
        # run bash script to call virtualenv
        os.system("./script/makeposemap.sh \"{0}\"".format(opts.dataroot))

def PoseTransferMakeImage(opts, phase):
    os.system("./script/makeposetransfer.sh \"{0}\" \"{1}\" {2}".format(opts.dataroot, "Pose-Transfer/checkpoint", phase))

def DGNetMakeImage(opts, phase):
    os.system("./script/makedgnet.sh \"{0}\" {1}".format(opts.dataroot, phase))


if __name__ == "__main__":
    phase = 1
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    opts = opt.parser().parse_args()

    PoseTransferDataPrep(opts)

    if not opts.use_existing_guide:
        DatasetGenerateGuide(opts)
        pass
    else:
        print("--use_existing_guide makes this script use an existing generation guide")

    for phasestr in opts.phases:
        if phase > 2:
            raise Exception("Too many phases. Maximum two phases")
        if phasestr == "pose":
            PoseTransferMakeImage(opts, phase)
        elif phasestr == "cloth":
            DGNetMakeImage(opts, phase)
        else:
            raise Exception("Unknown phase string \"{0}\". Please enter an enumeration of [pose, cloth]!")
        phase += 1
