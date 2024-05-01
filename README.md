# CS336 Spring 2024 Assignment 2: Systems

For a full description of the assignment, see the assignment handout at
[cs336_spring2024_assignment2_systems.pdf](./cs336_spring2024_assignment2_systems.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `setup.py`. This module should contain
  your from-scratch language model from assignment 1.
- [`./cs336-systems`](./cs336-systems): directory containing a module
  `cs336_systems` and its associated `setup.py`. In this module, you will
  implement an optimized Transformer language model---feel free to take your
  code from assignment 1 (in `cs336-basics`) and copy it over as a starting
  point. In addition, you will implement for distributed training and
  optimization in this module.

Visually, it should look something like:

``` sh
.
├── cs336-basics # Files from assignment 1 
│   ├── cs336_basics # A python module named cs336_basics
│   │   ├── __init__.py
│   │   ├── VERSION
│   │   └── ... other files in the cs336_basics module, taken from assignment 1 ...
│   ├── requirements.txt
│   └── setup.py (setup.py to install `cs336_basics`) 
├── cs336-systems # TODO(you):code that you'll write for assignment 2 
│   ├── cs336_systems # A python module named cs336_systems
│   │   ├── __init__.py
│   │   ├── VERSION
│   │   └── ... TODO(you): other python files that you need for assignment 2 ...
│   ├── requirements.txt
│   ├── ... TODO(you): any other files or folders you need for assignment 2 ...
│   └── setup.py (setup.py to install `cs336_systems`)
├── README.md
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

0. Set up a conda environment and install packages. In particular, the
   `cs336-basics` package (located at [`./cs336-basics`](./cs336-basics))
   installs the `cs336_basics` module, and the `cs336-systems` package (located
   at [`./cs336-systems`](./cs336-systems)) installs the `cs336_systems` module.

``` sh
conda create -n cs336_systems python=3.10 --yes
conda activate cs336_systems
pip install -e ./cs336-basics/ -e ./cs336-systems/'[test]'
```

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.
