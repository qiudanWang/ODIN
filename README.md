# ODIN
## Documents
https://drive.google.com/file/d/1uJb0GoyVtdrc-ETLh1jUgTrDmwNaw6CF/view?usp=share_link

## Usage
For the ODIN - MJT method, just install the jupter and run the ipynb inside the folder.

For the Extended-Validation:

1. PointNet++: python3 test_classification.py --log_dir pointnet2_cls_ssg --num_category 10 --use_cpu

2. LSTM: python3 predict.py I hate this movie

The output of both the PointNet++ and LSTM algorithms includes three columns: 
the first column represents the temperature scaling factor used, 
the second column shows the magnitude of the perturbation applied, 
and the third column indicates the maximum softmax score of the input. 

The file "calMetric.py" is used to calculate metrics for both algorithms.
