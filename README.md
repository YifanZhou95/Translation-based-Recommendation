# Translation-based-Recommendation

## Description
This project intends to reproduce [Translation-based Recommendation](https://arxiv.org/abs/1707.02410) using Python, though, the authors published official C++ code.

## Environment
Python 2.7

## Run
1. Clone the entire project

2. You could of course download raw dataset to root directory from http://jmcauley.ucsd.edu/data/amazon/links.html, then execute Datapreprocessing.py and DataPartition.py in order to get more structured dataset packaged in numpy format.

3. For your convenience, alternatively, you can basically run "src/TransRec.py" and other baselines, e.g. "FPMC.py", since numpy datasets of a few categories already exist.

4. To change different dataset category, e.g. "Automotive", for training and evaluation, put your category name here.
```
dataset_name = 'Automotive'
```
