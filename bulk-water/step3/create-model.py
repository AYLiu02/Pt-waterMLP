#!/usr/bin/env python

from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,'../../aml')
import aml


# Set random number seed for reproducability
np.random.seed(0)

# load training set from the end of QbC
print('Load training set... ', end='', flush=True)
structures = aml.Structures.from_file('../step2/iteration-015/train-000/input.data')
print('done.')
print()

# construct blank C-NNP object
print('Create blank C-NNP model... ', end='', flush=True)
# settings - n2p2 constructor
kwargs_model = dict(
    elements = ('O', 'H'),
    n = 8,
    fn_template = '../step2/input.nn',
    n_tasks = 8,
    n_core_task = 16
)
n2p2 = aml.N2P2(dir_run=Path('./final-training'), **kwargs_model)
print('done.')
print()

# perform training
print('Train all member models... ', end='', flush=True)
n2p2.train(structures, n_epoch=50)
print('done.')
print()

# save final C-NNP model in a format suitable for prediction
print('Save all member models... ', end='', flush=True)
n2p2.save_model()
print('done.')
print()
