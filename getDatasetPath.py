import glob
import os

imgPath = {
    "Guo": "{}_recovery.jpg",
    "Gong": "{}.png",
    "ST-CGAN": "{}_fake_C.png",
    "Yang": "{}.png"
}


def getDatasetPath(evalPath, dataset="ISTD"):

    if dataset == "ISTD":
        output_img_path = sorted(glob.glob("results/ISTD/" + evalPath + "/*.png"))
        gt_img_path = sorted(glob.glob("datasets/ISTD/test/test_C/*.png"))
        mask_img_path = sorted(glob.glob("datasets/ISTD/test/test_B/*.png"))

        if evalPath == "Guo" or evalPath == "ST-CGAN" or evalPath == "Gong" or evalPath == "Yang":
            output_img_path = []
            for i in range(len(gt_img_path)):
                output_img_path.append("results/ISTD/" + evalPath + "/" + imgPath[evalPath].format(i + 1))

        elif evalPath == "Input":
            output_img_path = sorted(glob.glob("datasets/ISTD/test/test_A/*.png"))

    elif dataset == "ISTDplus":
        output_img_path = sorted(glob.glob("results/ISTDplus/" + evalPath + "/*.png"))
        gt_img_path = sorted(glob.glob("datasets/ISTDplus/test/*.png"))
        mask_img_path = sorted(glob.glob("datasets/ISTD/test/test_B/*.png"))

        if evalPath == "Input":
            output_img_path = sorted(glob.glob("datasets/ISTD/test/test_A/*.png"))

    elif dataset == "SRD":
        output_img_path = sorted(glob.glob("results/SRD/" + evalPath + "/*.*"))
        gt_img_path = sorted(glob.glob("datasets/SRD/test/shadow_free/*.jpg"))
        mask_img_path = sorted(glob.glob("datasets/SRD/test/mask_new/*.png"))

        if evalPath == "DSC":
            gt_img_path = []
            mask_img_path = []
            for i in range(len(output_img_path)):
                gt_img_path.append("datasets/SRD/test/shadow_free/{}".format(os.path.basename(output_img_path[i]).replace('.jpg', '_free.jpg')))
                mask_img_path.append("datasets/SRD/test/mask/{}".format(os.path.basename(output_img_path[i])))

        elif evalPath == "Input":
            output_img_path = sorted(glob.glob("datasets/SRD/test/shadow/*.jpg"))

    elif dataset == "SRDplus":
        output_img_path = sorted(glob.glob("results/SRDplus/" + evalPath + "/*.*"))
        gt_img_path = sorted(glob.glob("datasets/SRDplus/test/*.jpg"))
        mask_img_path = sorted(glob.glob("datasets/SRD/test/mask_new/*.png"))

        if evalPath == "DSC":
            gt_img_path = []
            mask_img_path = []
            for i in range(len(output_img_path)):
                gt_img_path.append("datasets/SRDplus/test/{}".format(os.path.basename(output_img_path[i])))
                mask_img_path.append("datasets/SRD/test/mask/{}".format(os.path.basename(output_img_path[i])))

        elif evalPath == "Input":
            output_img_path = sorted(glob.glob("datasets/SRD/test/shadow/*.jpg"))

    if (len(output_img_path) == 0):
        print("Cannot find any Image. Provide another output data.")
        return [0, 0, 0]

    if len(output_img_path) != len(gt_img_path):
        print("Number of images is not same. Output: {}, Dataset: {}".format(len(output_img_path), len(gt_img_path)))
        return [0, 0, 0]

    return [output_img_path, gt_img_path, mask_img_path]
