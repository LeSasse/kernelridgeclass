# kernel ridge classification
A scikit-learn estimator for kernel ridge classification.
It fits a [KernelRidge regression](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html#sklearn.kernel_ridge.KernelRidge)
and then binarises the predictions.

First create a virtual environment using venv or conda. 
Make sure you upgrade pip. For example:
```sh
python3 -m venv .myexamplevenv
source .myexamplevenv/bin/activate
pip install -U pip
```
To use the existing code simply install the package:
```sh
git clone https://github.com/LeSasse/kernelridgeclass.git
cd kernelridgeclass
pip install -e .
```
