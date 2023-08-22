#!/usr/bin/env python

import sys
sys.path.insert(0,'../../aml')
import aml
import aml.score as mlps
import numpy as np

# settings - original AIMD trajectory
dt_ref = 2.5
dir_AIMD = '../step1/'
fn_trj_ref = dir_AIMD + 'bulk_water_AIMD-pos-1.xyz'
fn_trj_ref_au = dir_AIMD + 'bulk_water_AIMD-pos-1-bohr.xyz'
#fn_frc_ref = dir_AIMD + 'bulk_water_AIMD-frc-1.xyz'
fn_frc_ref_au = dir_AIMD + 'bulk_water_AIMD-frc-1-harbohr.xyz'
#fn_vel_ref = dir_AIMD + 'bulk_water_AIMD-vel-1.xyz'
fn_topo_ref = dir_AIMD + 'top.pdb'
fn_topo_au = dir_AIMD + 'top_au.pdb'

# settings - C-NNP model
dir_model = '../step3/final-training/model/'

# settings - C-NNP trajectory
dt_test = 2.5
dir_C_NNP = '../step42/'
fn_trj_test = dir_C_NNP + '64wat-pos-1.xyz'
#fn_vel_test = dir_C_NNP + '64wat-vel-1.xyz'
fn_topo_test = dir_AIMD + 'top.pdb'

# load position trajectory
trj_ref = mlps.load_with_cell(fn_trj_ref, top=fn_topo_ref)
trj_test = mlps.load_with_cell(fn_trj_test, top=fn_topo_test)

# perform RDF scoring
mlps.run_rdf_test(trj_ref, trj_test)

# load velocity trajectory
#vel_ref = mlps.load_with_cell(fn_vel_ref, top=fn_topo_au)
#vel_test = mlps.load_with_cell(fn_vel_test, top=fn_topo_au)

# perform VDOS scoring
#mlps.run_vdos_test(vel_ref, dt_ref, vel_test, dt_test)

# read AIMD trajectory positions and forces
frames = aml.read_frames_cp2k(fn_positions=fn_trj_ref_au, fn_forces=fn_frc_ref_au, cell=23.4703962903*np.identity(3))
structures_ref = aml.Structures.from_frames(frames, stride=1, probability=1.0)

# perform force RMSE scoring
mlps.run_rmse_test(dir_model, fn_topo_au, structures_ref)
