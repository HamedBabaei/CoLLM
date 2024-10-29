# REPRO
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## Directories
- `case-studies`: you need to create your case-study-specific directory here (if it is not there already). Please organize your code to be executable via the same directory as a root.
- `images`: directory contains any general images/diagrams related to the project. e.g. architecture diagram.

## Contributors Guidelines
*Feel free to skip step 2 if it is inconvenient for you to use pre-commit, once you are done and request for merge, I will take care of fixing the pre-commit related issues -- it is not a big deal at the moment*

1. Clone the repository to your local machine:
```bash
 git clone git@github.com:HamedBabaei/REPRO.git
 cd REPRO
```

2. Create a virtual environment with `python=3.9` (or any python distribution), activate it, install the required
   dependencies and **install the pre-commit configuration:**

```bash
conda create -n my_env python=3.9
conda activate my_env
pip install -r requirements.txt
pre-commit install
```

3. Create a branch and commit your changes:
```bash
git switch -c <name-your-branch>
# do your changes
git add .
git commit -m "your commit msg"
git push
```

4. Once you finished your work, please make a merge request to `main` for review. We will check for any minor issue that code may cause - such removal of secret keys, missing files and ...