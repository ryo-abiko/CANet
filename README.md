
# CANet
Repo of "Channel Attention GAN Trained with Enhanced Dataset for Single Image Shadow Removal"

The full source code will be available once the paper is accepted.

## Datasets
* [ISTD](https://github.com/DeepInsight-PCALab/ST-CGAN) : ST-CGAN
* [SRD](http://www.shengfenghe.com/publications/) : Deshadow-Net
* ISTD+, SRD+  
Datasets are not available but you can create the dataset by applying the [MATLAB code](https://www3.cs.stonybrook.edu/~cvl/projects/SID/index.html) provided by the authors.

## Evaluation
Put your results under `results/<dataset name>/<method name>`.  
Make sure that test images are located under `datasets/<dataset name>`.

Run the following command.
```
python evaldata.py \
    --method_name <method name> \
    --dataset_name ISTD \
    --resized 1
```

## Directory structure
```
CANet
├── evaldata.py
├── getDatasetPath.py
├── README.md
├── datasets/
│   ├── ISTD/
│       ├── test/
│           ├── test_A/
│               ├── 101.png
│           ├── test_B/
│           ├── test_C/
│       ├── train/
├── results/
│   ├── ISTD/
│       ├── CANet/
│           ├── 101.png
```