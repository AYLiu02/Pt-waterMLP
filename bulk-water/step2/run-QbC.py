#!/usr/bin/env python

from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,'../../aml')

import aml


# Set random number seed for reproducability
np.random.seed(0)

def read_structures():

    # location of data
    dir_trj = Path('../step1')

    # stride through trajectory
    stride_trj = 1

    print('Reading structures')
    print('------------------')
    print()
    print(f'Directory: {dir_trj}')
    print(f'Stride: {stride_trj}')
    print()

    fn_positions = dir_trj / 'bulk_water_AIMD-pos-1-bohr.xyz'
    fn_forces = dir_trj / 'bulk_water_AIMD-frc-1-harbohr.xyz'

    frames = aml.read_frames_cp2k(fn_positions=fn_positions, fn_forces=fn_forces, cell=23.4703962903*np.identity(3))
    print("Frames read")
    structures = aml.Structures.from_frames(frames, stride=stride_trj, probability=1.0)

    print(f'{len(structures):} structures kept')
    print()

    return structures


# settings - n2p2 constructor
kwargs_model = dict(
    elements = ('O', 'H'),
    n = 8,
    fn_template = 'input.nn',
    n_tasks = 8,
    n_core_task = 16,
    remove_output = True
)

# structures to select from
structures = read_structures()

qbc = aml.QbC(
    structures = structures,
    cls_model = aml.N2P2,
    kwargs_model = kwargs_model,
    n_train_initial = 20,
    n_add = 20,
    n_epoch = 15,
    n_iterations = 15,
    n_candidate = 10001,
    fn_results = 'results.shelf',
    fn_restart = None
)

qbc.run()
