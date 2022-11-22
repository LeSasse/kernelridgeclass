# fcmodels
A scikit-learn estimator searching through models commonly successful in the functional connectivity literature

First create a virtual environment using venv or conda. 
Make sure you upgrade pip. For example:
```sh
python3 -m venv .myexamplevenv
source .myexamplevenv/bin/activate
pip install -U pip
```
To use the existing code simply install the package:
```sh
git clone https://github.com/LeSasse/fcmodels.git
cd fcmodels
pip install -e .
```
Also make sure you install the developer version of julearn:
```sh
pip install --index-url https://test.pypi.org/simple/ -U julearn --pre
```
You can also install the dev-requirements.txt which installs some additional packages that are useful for development:
```sh
pip install -r dev-requirements.txt
```