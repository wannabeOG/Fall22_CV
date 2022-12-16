A PyTorch Implementation of iCaRL
A PyTorch Implementation of iCaRL: Incremental Classifier and Representation Learning.

The code implements experiments on Tiny ImageNet

Notes
This code does not reproduce the results from the paper. This may be due to following reasons:
Different hyper-parameters than the original ones in the paper (e.g. learning rate schedules).
Different loss function; I replaced BCELoss to CrossEntropyLoss since it seemed to produce better results.
Versions
Python 3+
PyTorch v0.1.12


First run data_prep.py to download and assemble the dataset
Then run main.py
Import AlexNet separately if needed.