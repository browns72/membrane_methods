#!/usr/bin/env python
# coding: utf-8

# # Description of the system

# Chemical composition: 85% POPC, 15% POPS, 0.15 salt concentration
# System initial size: 400x400x150 nm
# 
# Force field: Martini 2.2
# Barostat: 1 bar, Parrinello-Rahman, semi-isotropic, tau_p = 12 ps, compressibility = 4.5e-5 bar^-1
# Thermostat: 323 K, V-rescale, tau_t = 1 ps
# 
# vdw: potential-shift-verlet, cutoff = 1.1 nm
# coulomb: cutoff = 1.1 nm
# 

# # Imports, helper functions
# 

# In[1]:


import MDAnalysis as mda
import MDAnalysis.transformations
from MDAnalysis.analysis.leaflet import LeafletFinder
from MDAnalysis import units


# In[2]:


import finufft


# In[3]:


import nglview as nv


# In[4]:


from MDAnalysis.analysis.distances import self_distance_array


# In[5]:


import pickle


# In[6]:


from tqdm.notebook import tqdm
from IPython.display import Math, display


# In[7]:


import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
# import griddata
from scipy.interpolate import griddata


# In[8]:


import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# In[9]:


from collections import Counter
import sys
import tracemalloc
import os


# In[10]:


from dask.distributed import Client
import dask


# In[11]:


def print_out_composition(u):
    # print out the breakdown of the universe into segments and residues
    for seg in u.segments:
        #
        print(f'Segment {seg.segid} ({seg.atoms.n_atoms} atoms)')

        # select the atoms in the segment
        sel = u.select_atoms(f'segid {seg.segid}')

        # get the residues
        residues = [residue.resname for residue in sel.residues]

        # print out the residue range
        residue_first = sel.residues[0].resid
        residue_last = sel.residues[-1].resid
        print('residue range %i-%i; ' % (residue_first, residue_last), end='')

        # count the residues
        aa_count = Counter(residues)
        for aa, count in aa_count.items():
            # print out the residue name and the number of residues
            print(f"{count}{aa}, ", end='')
        print()
        print()


# # Temperature

# In[12]:


kB = 1.38064852e-23
T = 310.
kT = kB * T
beta = 1 / kT
coef_k_theta = kT * 1e21


# # Define heads and tails for each lipid

# In[13]:


lipid_ends = {'PAPS':{'head':'PO4','tail':'C5A C5B', 'med':'?1?'},
              'POPS':{'head':'PO4','tail':'C4A C4B', 'med':'C1*'},
              'DFPC':{'head':'PO4','tail':'D4A D4B', 'med':'?1?'},
              'DIPC':{'head':'PO4','tail':'C4A C4B', 'med':'C1*'},
              'DTPC':{'head':'PO4','tail':'C2A C2B', 'med':'?1?'},
              'DVPC':{'head':'PO4','tail':'C4A C4B', 'med':'?1?'},
              'DXPC':{'head':'PO4','tail':'C6A C6B', 'med':'?1?'},
              'DYPC':{'head':'PO4','tail':'C3A C3B', 'med':'?1?'},
              'DGPC':{'head':'PO4','tail':'C5A C5B', 'med':'?1?'},
              'DLPC':{'head':'PO4','tail':'C3A C3B', 'med':'?1?'},
              'DNPC':{'head':'PO4','tail':'C6A C6B', 'med':'?1?'},
              'DPPC':{'head':'PO4','tail':'C4A C4B', 'med':'?1?'},
              'DBPC':{'head':'PO4','tail':'C5A C5B', 'med':'?1?'},
              'PAPC':{'head':'PO4','tail':'C5A C4B', 'med':'?1?'},
              'PEPC':{'head':'PO4','tail':'C5A C4B', 'med':'?1?'},
              'POPC':{'head':'PO4','tail':'C4A C4B', 'med':'C1*'},
              'DOPC':{'head':'PO4','tail':'C4A C4B', 'med':'?1?'},
              'DAPC':{'head':'PO4','tail':'C5A C5B', 'med':'?1?'},
              'LPPC':{'head':'PO4','tail':'C4A C3B', 'med':'?1?'},
              'PGPC':{'head':'PO4','tail':'C5A C4B', 'med':'?1?'},
              'PIPC':{'head':'PO4','tail':'C4A C4B', 'med':'C1*'},
              'DAPE':{'head':'PO4','tail':'C5A C5B', 'med':'?1?'},
              'DBPE':{'head':'PO4','tail':'C5A C5B', 'med':'?1?'},
              'DFPE':{'head':'PO4','tail':'D4A D4B', 'med':'?1?'},
              'DGPE':{'head':'PO4','tail':'C5A C5B', 'med':'?1?'},
              'DIPE':{'head':'PO4','tail':'C4A C4B', 'med':'?1?'},
              'DLPE':{'head':'PO4','tail':'C3A C3B', 'med':'?1?'},
              'DNPE':{'head':'PO4','tail':'C6A C6B', 'med':'?1?'},
              'DOPE':{'head':'PO4','tail':'C4A C4B', 'med':'?1?'},
              'DPPE':{'head':'PO4','tail':'C4A C4B', 'med':'?1?'},
              'DRPE':{'head':'PO4','tail':'D6A D6B', 'med':'?1?'},
              'DTPE':{'head':'PO4','tail':'C2A C2B', 'med':'?1?'},
              'DUPE':{'head':'PO4','tail':'D5A D5B', 'med':'?1?'},
              'DVPE':{'head':'PO4','tail':'C4A C4B', 'med':'?1?'},
              'DXPE':{'head':'PO4','tail':'C6A C6B', 'med':'?1?'},
              'DYPE':{'head':'PO4','tail':'C3A C3B', 'med':'?1?'},
              'LPPE':{'head':'PO4','tail':'C4A C3B', 'med':'?1?'},
              'PAPE':{'head':'PO4','tail':'C5A C4B', 'med':'?1?'},
              'PGPE':{'head':'PO4','tail':'C5A C4B', 'med':'?1?'},
              'PIPE':{'head':'PO4','tail':'C4A C4B', 'med':'?1?'},
              'POPE':{'head':'PO4','tail':'C4A C4B', 'med':'C1*'},
              'PQPE':{'head':'PO4','tail':'C5A C4B', 'med':'?1?'},
              'PRPE':{'head':'PO4','tail':'D6A C4B', 'med':'?1?'},
              'PUPE':{'head':'PO4','tail':'D5A C4B', 'med':'?1?'},
              'DPSM':{'head':'PO4','tail':'C3A C4B', 'med':'?1?'},
              'DXSM':{'head':'PO4','tail':'C5A C6B', 'med':'?1?'},
              'PGSM':{'head':'PO4','tail':'C3A C5B', 'med':'?1?'},
              'PNSM':{'head':'PO4','tail':'C3A C6B', 'med':'?1?'},
              'POSM':{'head':'PO4','tail':'C3A C4B', 'med':'?1?'},
              'XNSM':{'head':'PO4','tail':'C5A C6B', 'med':'?1?'},
              }


# # Load the universe
# 

# In[14]:


# load the universe
u = mda.Universe('20-prod.tpr','20-prod.xtc')
P = u.select_atoms('name PO4')

time_moments = np.linspace(0, u.trajectory.n_frames * u.trajectory.dt, u.trajectory.n_frames+1)


# In[15]:


import MDAnalysis.transformations
# On-the-fly transformations

#lipid = u.select_atoms('resname POPC')
#tr0 = mda.transformations.unwrap(lipid)
#u.trajectory.add_transformations(tr0)


# ## Composition of the universe

# In[16]:


# print out the composition of the universe
print('Box dimensions (A): %.1f %.1f %.1f' % tuple(u.dimensions[:3]))
print('N_frames: ', u.trajectory.n_frames)
print('N_atoms: ', u.atoms.n_atoms)
print('Trajectory time range (ns): %.1f, dt between snapshots (ps): %.1f' % (u.trajectory.n_frames * u.trajectory.dt/1000, u.trajectory.dt))


# In[17]:


# Show the membrane (hide water/salt)
w = nv.show_mdanalysis(u.select_atoms('name PO4'), skip=10)
w.clear()
# draw_box(u, w)

w.add_representation('spacefill', 'all')

w.background = 'black'
#w.display(gui=True)
w._remote_call("setSize", target="Widget", args=["1000px", "600px"])
w


# In[ ]:





# ## Set up the frame sampling

# In[18]:


N_sampled_frames = 900
start_frame = 101
indices = np.linspace(start_frame, u.trajectory.n_frames - 1, N_sampled_frames, dtype=int)
print(f'Number of sampled frames: {N_sampled_frames} out of {u.trajectory.n_frames-start_frame} frames (frame0 = {start_frame})')


# In[19]:


sampled_times = time_moments[indices]
sampled_times_ns = units.convert(sampled_times, 'ps', 'ns')
print(f'Time range of the sampled frames: {sampled_times_ns.min():.2f} ns - {sampled_times_ns.max():.2f} ns')
print(f'Time step between sampled frames: {sampled_times_ns[1]-sampled_times_ns[0]:.2f} ns')


# ## Define the leaflets

# In[20]:


from MDAnalysis.analysis.leaflet import LeafletFinder
L = LeafletFinder(u, 'name PO4',pbc=True)

for i,g in enumerate(L.groups()):
    print('Leaflet %i: %i atoms' % (i,g.n_atoms))


# In[21]:


top, bottom = list(L.groups_iter())


# ## Show the cell dimensions over time

# In[ ]:





# In[22]:


b_load = True
if b_load and os.path.isfile('cell_dimensions.pickle'):
    # Load the cell dimensions from the pickle
    with open('cell_dimensions.pickle', 'rb') as f:
        cell_dimensions = pickle.load(f)
    cell_len = cell_dimensions['cell_len']
    cell_height = cell_dimensions['cell_height']
    thicknesses = cell_dimensions['thicknesses']
else:
    # Collect the cell dimensions change over time
    cell_len = np.zeros(N_sampled_frames)
    cell_height = np.zeros(N_sampled_frames)
    thicknesses = np.zeros(N_sampled_frames)
    i = 0
    for i, ts in enumerate(u.trajectory[indices]):
        cell_len[i], _, cell_height[i] = units.convert(ts.dimensions[:3], 'A', 'nm')
        thicknesses[i] = units.convert(top.positions[:,2].mean() - bottom.positions[:,2].mean(), 'A', 'nm')

    # save the cell dimensions as a pickle
    with open('cell_dimensions.pickle', 'wb') as f:
        pickle.dump({'cell_len': cell_len, 'cell_height': cell_height, 'thicknesses': thicknesses}, f)


# In[23]:


cell_length = cell_len.mean()
area_per_lipid = cell_length**2 / top.n_atoms
cell_area = cell_length**2
thickness = thicknesses.mean()
print(f'Average cell length: {cell_length:.2f} nm +/- {cell_len.std():.2f} nm')
print(f'Average area per lipid: {area_per_lipid:.2f} nm^2')
print(f'Average cell height: {cell_height.mean():.2f} nm +/- {cell_height.std():.2f} nm')
print(f'Average membrane thickness: {thickness:.2f} nm +/- {thicknesses.std():.2f} nm')


# # Perform the Real-Space Fluctuation Analysis of the membrane

# In[24]:


lipid = u.select_atoms('resname POPC or resname POPE')


# In[ ]:





# In[25]:


def get_distance_matrix(P):
    # Get the distance matrix
    self_distance_P = self_distance_array(P.positions, box=P.dimensions)
    # convert to  distance matrix
    self_dist = np.zeros((len(P), len(P)))
    self_dist[np.triu_indices(len(P), k=1)] = self_distance_P
    self_dist += self_dist.T
    return self_dist


# In[26]:


ts = u.trajectory[indices[0]]
lipid.atoms.unwrap(compound='residues')
P = top.select_atoms('name PO4')

dist = get_distance_matrix(P)
print(f'Minimum distance: {dist[dist>0].min():.1f}; Maximum distance: {dist.max():.1f}; Mean distance: {dist.mean():.1f}')
print(f'Expected max distance: {P.dimensions[0] / np.sqrt(2):.1f}')


# In[27]:


def get_neighbors(P, methods):
    """
    Get the neighbor list of each lipid

    :param P: group of atoms representing the membrane surface, usually using the PO4 atoms
    :param methods: list of tuples (method, cutoff)
    :return: list of neighbor lists, each element is a list of indices of the neighbors of the corresponding lipid
    """
    self_dist = get_distance_matrix(P)
    results = []
    for method, cutoff in methods:
        neighbor_list = []
        if method == 'within':
            for i in range(len(P)):
                neighbor_list.append(np.where(self_dist[i] < cutoff)[0])
        elif method == 'nearest':
            for i in range(len(P)):
                neighbor_list.append(np.argsort(self_dist[i])[0:cutoff])
        results.append(neighbor_list)
    return results


# In[28]:


ts = u.trajectory[indices[0]]
lipid.atoms.unwrap(compound='residues')
P = top.select_atoms('name PO4')

neighbor_list = get_neighbors(P, methods=[('within', 6)])[0]
print(neighbor_list[0])

neighbor_list = get_neighbors(P, methods=[('nearest', 3)])[0]
print(neighbor_list[0])


# In[29]:


class StructuralParameters:
    def __init__(self, P, neighbor_list, lipid_ends, method_normals = 'rotation'):
        self.P = P
        N = len(P)
        self.method_normals = method_normals
        self.neighbor_list = neighbor_list
        self.lipid_ends = lipid_ends

        self.computed = np.zeros(N, dtype=bool)

        self.p_head = np.zeros((N, 3))
        self.p_med = np.zeros((N, 3))
        self.p_tail = np.zeros((N, 3))

        self.n = np.zeros((N, 3))
        self.N = np.zeros((N, 3))

    def __repr__(self):
        s = f'StructuralParameters(N={len(self.P)}, method_normals={self.method_normals})'
        s += f'\nComputed: {self.computed.sum()} out of {len(self.P)}'
        return s

    def __getitem__(self, i):
        if not self.computed[i]:
            self.compute(i)
            self.computed[i] = True

        d = {'head': self.p_head[i, :],
             'med': self.p_med[i, :],
             'n': self.n[i, :],
             'N': self.N[i, :]
             }
        return d

    def compute_N_rotation(self, neighbor_pos, neighbor_masses):
        com = np.average(neighbor_pos, axis=0, weights=neighbor_masses)
        pos = neighbor_pos - com

        inertia_tensor = np.zeros((3, 3))
        for coord, mass in zip(pos, neighbor_masses):
            inertia_tensor += mass * (np.dot(coord, coord) * np.eye(3) - np.outer(coord, coord))

        # Get the eigenvectors and eigenvalues of the inertia tensor
        eigvals, eigvecs = np.linalg.eig(inertia_tensor)
        # Sort the eigenvectors and eigenvalues
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # Get the principal axes of rotation
        principal_axes = eigvecs.T
        largest_axis = principal_axes[0]
        # Check if the largest axis has a positive z-component
        if largest_axis[2] < 0:
            largest_axis *= -1
        return largest_axis

    #SVD to return normal to bilayer surface, equal to angle of rotation
    def compute_N_SVD(self, neighbor_pos):
        _, _, vh = np.linalg.svd(neighbor_pos)
        # Plane's normal vector is the last column in Vh
        normal = vh[-1]
        if normal[2] < 0:
            normal *= -1
        return normal

    def compute_N(self, i):

        neighbors = self.neighbor_list[i]
        neighbor_pos = P.positions[neighbors] - P.positions[i]
        neighbor_pos -= np.round(neighbor_pos / P.dimensions[:3]) * P.dimensions[:3]
        neighbor_masses = P.masses[neighbors]

        #Calls to desired function within class
        if self.method_normals == 'rotation':
            return self.compute_N_rotation(neighbor_pos, neighbor_masses)
        elif self.method_normals == 'SVD':
            return self.compute_N_SVD(neighbor_pos)
        else:
            raise ValueError('Invalid compute method')

    def compute(self, i):
        #Select atoms and define lipid parameters
        ri = self.P[i].residue
        ri_tail = ri.atoms.select_atoms('name %s' % self.lipid_ends[ri.resname]['tail'])
        ri_head = ri.atoms.select_atoms('name %s' % self.lipid_ends[ri.resname]['head'])
        ri_med = ri.atoms.select_atoms('name %s' % self.lipid_ends[ri.resname]['med'])

        self.p_tail[i, :] = ri_tail.center_of_mass()
        self.p_head[i, :] = ri_head.center_of_mass()
        self.p_med[i, :] = ri_med.center_of_mass()

        self.n[i, :] = self.p_head[i, :] - self.p_tail[i, :]
        n_length = np.linalg.norm(self.n[i, :])
        assert n_length < 30, f'Director {i} too long: {n_length}'
        self.n[i, :] /= n_length

        #Call to self.compute_N
        self.N[i, :] = self.compute_N(i)

    def get(self, i):
        if not self.computed[i]:
            self.compute(i)
            self.computed[i] = True
        return {'head': self.p_head[i, :], 'med': self.p_med[i, :], 'n': self.n[i, :], 'N': self.N[i, :]}



# In[30]:


# Test the class
ts = u.trajectory[indices[0]]
lipid.atoms.unwrap(compound='residues')
neighbor_list = get_neighbors(P, methods=[('within', 10.0)])[0]
struct_svd = StructuralParameters(P, neighbor_list, lipid_ends, method_normals='SVD')
struct_rot = StructuralParameters(P, neighbor_list, lipid_ends, method_normals='rotation')


# In[31]:


normals_svd = np.array([struct_svd[i]['N'] for i in range(len(P))])
normals_rot = np.array([struct_rot[i]['N'] for i in range(len(P))])


# In[ ]:


Ss = []

for ts in tqdm(u.trajectory[indices]):
    lipid.atoms.unwrap(compound='residues')
    #neighbor_list = get_neighbors(P, method='d_cutoff', d_cutoff=10.)
    neighbors_for_S, neighbors_for_N = get_neighbors(P, methods=[('nearest', 5), ('nearest', 10)])  #Get neighbor list
    struct = StructuralParameters(P, neighbors_for_N, lipid_ends, method_normals='SVD') #Call to structural parameters class


    #For each lipid, compute splay with neighbors
    for i in range(len(P)):
        for j in neighbors_for_S[i]:
            if i >= j:
                continue
            ni, nj = struct[i]['n'], struct[j]['n']
            Ni, Nj = struct[i]['N'], struct[j]['N']
            e = struct[j]['med'] - struct[i]['med']

            # apply minimal image convention
            e = e - np.round(e / u.dimensions[:3]) * u.dimensions[:3]
            e -= np.dot(e, Ni) * Ni
            d = np.linalg.norm(e)

            assert d < 50, f'Neighbors {i}, {j} are too far apart from each other: {d:.2f} A'
            e_unit = e / d

            #S = np.dot(nj - ni, e_unit) / d  # Eq 8 of Doktorova PCCP 2017
            S = np.dot(nj - ni - Nj + Ni, e_unit) / d  # Eq 8 of Doktorova PCCP 2017
            # S = np.dot(Ni - Nj, e_unit) / d  # Eq 8 of Doktorova PCCP 2017

            Ss.append(S)


# In[ ]:


np_Ss = np.array(Ss)
np_Ss.sort()

print(f'Number of pairs used: {len(Ss)}, that is {len(Ss) / len(P) / len(indices):.2f} pairs per lipid per frame')
print(f'Range of S: from {np_Ss.min():.2f} to {np_Ss.max():.2f}')


# In[ ]:

#Plot and bin splays with associated probabilities
Probabilities, bins = np.histogram(np_Ss[20:-20], bins=50, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2


# In[ ]:

# Fit Gaussian
from scipy.optimize import curve_fit
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
popt, pcov = curve_fit(gaussian, bin_centers, Probabilities, p0=[1, 0, 1])


# In[ ]:

#Extract bending modulus from gaussian fit.
sigma2 = popt[2]**2
area_per_lipid_A2 = area_per_lipid * 100
# Calculate the bending modulus
Kc = 1 / (sigma2 * area_per_lipid_A2)
print(f'Bending modulus: {Kc:.2f} kT')


# In[ ]:


plt.plot(bin_centers, Probabilities)
plt.plot(bin_centers, gaussian(bin_centers, *popt))
plt.xlabel('S, 1/A')
plt.ylabel('Probability density, unscaled')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




