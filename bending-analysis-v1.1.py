#!/usr/bin/env python
# coding: utf-8

output_me={}
write_to='results.txt'
lipid_composition=[]
selection_string=str()
# # Imports, helper functions
# 

# In[2]:


import MDAnalysis as mda
import MDAnalysis.transformations
from MDAnalysis.analysis.leaflet import LeafletFinder
from MDAnalysis import units


# In[3]:


import finufft


# In[4]:


import nglview as nv


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



def print_out_composition(u):
    # print out the breakdown of the universe into segments and residues
    total_lipids=0
    composition={}
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

        # Sum up lipid types
        if ('PC' in seg.segid) or ('PE' in seg.segid) or ('SM' in seg.segid) or ('CHOL' in seg.segid):
            lipid_name=seg.segid.split('_')[2]
            if lipid_name not in composition:
                composition[str(lipid_name)] = residue_last-residue_first
            elif lipid_name in composition:
                composition[str(lipid_name)] += (residue_last-residue_first)
            total_lipids += (residue_last-residue_first)
            if (f'resname {lipid_name}') not in lipid_composition:
                lipid_composition.append(f'resname {lipid_name}')


        #Append water count to output dictionary
        if '_W' in seg.segid:
            output_me['water_count']=seg.atoms.n_atoms

    print(composition)

    #Compute percent composition and output into dictionary
    output_me['composition']=[]
    for i in composition.keys():
        output_me['composition'].append(f'{round(100*(composition[i]/total_lipids)):.2f}% {i}')






def square_complex(x, y=None):
    if y is None:
        y = x
    return np.real(np.conj(x) * y)




def expand_cell(group, z_mean, by=0.1):
    """
    Expand the cell by 10% by taking into account the periodicity of the cell

    :param group: atom group to be expanded
    :param by: the fraction of expansion
    :param z_mean: the average z position of the membrane
    :return: xy and z coordinates of the expanded cell
    """
    L_max = units.convert(group.dimensions[0], 'angstrom', 'nm')
    pos = units.convert(group.positions, 'angstrom', 'nm')

    xy, z = pos[:,:2], pos[:,2] - z_mean

    # Expand the cell by 10% by taking into account the periodicity of the cell
    cutoff = by * L_max
    m_left = xy[:,0] < cutoff
    m_right = xy[:,0] > L_max - cutoff
    m_bottom = xy[:,1] < cutoff
    m_top = xy[:,1] > L_max - cutoff

    expanded_xy = np.vstack([
        xy,
        xy[m_left] + np.array([L_max, 0]),
        xy[m_right] + np.array([-L_max, 0]),
        xy[m_bottom] + np.array([0, L_max]),
        xy[m_top] + np.array([0, -L_max]),
        xy[m_left & m_bottom] + np.array([L_max, L_max]),
        xy[m_left & m_top] + np.array([L_max, -L_max]),
        xy[m_right & m_bottom] + np.array([-L_max, L_max]),
        xy[m_right & m_top] + np.array([-L_max, -L_max]),
    ])

    expanded_z = np.hstack([
        z,
        z[m_left],
        z[m_right],
        z[m_bottom],
        z[m_top],
        z[m_left & m_bottom],
        z[m_left & m_top],
        z[m_right & m_bottom],
        z[m_right & m_top],
    ])

    return expanded_xy, expanded_z


# In[14]:


def create_grid(L_max, N_points):
    """
    Create a regular 2D grid of points in the xy plane

    :param L_max: the maximum length of the box
    :param N_points: the number of points in the grid
    :return: x and y coordinates of the grid
    """
    x = np.linspace(0, L_max, N_points+1)
    x = (x[:-1] + x[1:]) / 2

    y = np.linspace(0, L_max, N_points+1)
    y = (y[:-1] + y[1:]) / 2

    xx, yy = np.meshgrid(x, y)

    xy_new = np.vstack([xx.ravel(), yy.ravel()]).T
    return xy_new


# In[15]:


def fit_over_q4(x, y):
    # Returns the log10(A) of the fit y = A/x**4 of the intercept of the linear fit,
    # as well as the intercept with the theoretical slope of -4

    # Fit a line y= -4x + c to the data using least squares
    lg_A = np.log10(y).mean() - (-4) * np.log10(x).mean()
    return 10 ** lg_A


def get_fit(d):
    # Returns the slope and intercept of the linear fit,
    # as well as the intercept with the theoretical slope of -4
    x, y = np.log10(d['abs_q']), np.log10(d['u2_mean'])

    # Fit a line y= mx + c to the data using least squares
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    # Fit a line y= -4x + c to the data using least squares
    c4 = np.mean(y) - (-4) * np.mean(x)

    return m, c, c4


# In[16]:


def bending_constant_from_a_fit(a_fit, cell_area):
    return 1 / cell_area / a_fit


# In[17]:


def cummean_columns(arr):
    cumsum = np.cumsum(arr, axis=1)
    indices = np.arange(1, arr.shape[1] + 1)
    return cumsum / indices


# In[18]:


def cv_series(abs_q, u2s, maxq=None):
    unique_abs_q = np.unique(abs_q)
    if maxq is not None:
        unique_abs_q = unique_abs_q[unique_abs_q < maxq]
    cv = np.zeros((len(unique_abs_q), u2s.shape[1]))

    for i, q in enumerate(unique_abs_q):
        u2 = u2s[abs_q == q]
        # compute cummean of u2 along the time axis
        u2_mean = np.cumsum(u2, axis=1) / np.arange(1, u2.shape[1] + 1)
        cv[i, :] = np.std(u2_mean, axis=0) / np.mean(u2_mean, axis=0)
    return unique_abs_q, cv


# In[19]:


def get_decorrelation(x):
    x = (x - np.mean(x)) / (np.std(x) * np.sqrt(len(x)))
    autocorr_f = np.correlate(x, x, mode='full')
    return autocorr_f[autocorr_f.size // 2:]


# In[20]:


def group_by_q(abs_q, u2s, maxq=None):
    unique_abs_q = np.unique(abs_q)
    if maxq is not None:
        unique_abs_q = unique_abs_q[unique_abs_q < maxq]
    u2s_unique = np.zeros((len(unique_abs_q), u2s.shape[1]))

    for i, q in enumerate(unique_abs_q):
        u2s_unique[i, :] = u2s[abs_q == q, :].mean(axis=0)
    return unique_abs_q, u2s_unique


# In[21]:


def by_q(d, cols=None, maxq=None):
    if cols is None:
        d_by_q = d.copy().groupby('abs_q').mean().reset_index()
    else:
        if type(cols) == str:
            cols = [cols]
        if not 'q_abs' in cols:
            cols = cols + ['abs_q']
        d_by_q = d[cols].copy().groupby('abs_q').mean().reset_index()
    if maxq is not None:
        d_by_q = d_by_q[d_by_q['abs_q'] < maxq]
    return d_by_q


# # Temperature

# In[22]:


kB = 1.38064852e-23
T = 310.
kT = kB * T
beta = 1 / kT
coef_k_theta = kT * 1e21


# # Load the universe
# 

# In[23]:


# load the universe
u = mda.Universe('20-prod.tpr','20-prod.xtc')
time_moments = np.linspace(0, u.trajectory.n_frames * u.trajectory.dt, u.trajectory.n_frames+1)


# ## Composition of the universe

# In[24]:


# print out the composition of the universe
print('Box dimensions (A): %.1f %.1f %.1f' % tuple(u.dimensions[:3]))
print('N_frames: ', u.trajectory.n_frames)
print('N_atoms: ', u.atoms.n_atoms)
print('Trajectory time range (ns): %.1f, dt between snapshots (ps): %.1f' % (u.trajectory.n_frames * u.trajectory.dt/1000, u.trajectory.dt))


# In[25]:


print_out_composition(u)



# In[26]:


# Show the membrane (hide water/salt)
#w = nv.show_mdanalysis(u.select_atoms('name PO4'))
#w.clear()
# draw_box(u, w)

#w.add_representation('spacefill', 'all')

#w.background = 'black'
#w.display(gui=True)
#w._remote_call("setSize", target="Widget", args=["1000px", "600px"])
#w

#from nglview.contrib.movie import MovieMaker
#movie = MovieMaker(w, output='movie.gif',in_memory=False,moviepy_params='write_gif')
#movie.make()


# In[ ]:





# ## Set up the frame sampling

# In[27]:


N_sampled_frames = 900
start_frame = 101
indices = np.linspace(start_frame, u.trajectory.n_frames - 1, N_sampled_frames, dtype=int)
print(f'Number of sampled frames: {N_sampled_frames} out of {u.trajectory.n_frames-start_frame} frames (frame0 = {start_frame})')


# In[28]:


sampled_times = time_moments[indices]
sampled_times_ns = units.convert(sampled_times, 'ps', 'ns')
print(f'Time range of the sampled frames: {sampled_times_ns.min():.2f} ns - {sampled_times_ns.max():.2f} ns')
print(f'Time step between sampled frames: {sampled_times_ns[1]-sampled_times_ns[0]:.2f} ns')


# ## Define the leaflets

# In[29]:


from MDAnalysis.analysis.leaflet import LeafletFinder
u_leaf=u
u_leaf.trajectory[100]
L = LeafletFinder(u_leaf, 'name PO4',pbc=True)

for i,g in enumerate(L.groups()):
    print('Leaflet %i: %i atoms' % (i,g.n_atoms))

output_me['# Atoms/Leaflet']=g.n_atoms


# In[30]:


top, bottom = list(L.groups_iter())
P = u.select_atoms('name PO4')


# ## Show the cell dimensions over time

# In[ ]:





# In[31]:


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


# In[32]:


cell_length = cell_len.mean()
area_per_lipid = cell_length**2 / top.n_atoms
cell_area = cell_length**2
thickness = thicknesses.mean()
print(f'Average cell length: {cell_length:.2f} nm +/- {cell_len.std():.2f} nm')
print(f'Average area per lipid: {area_per_lipid:.2f} nm^2')
print(f'Average cell height: {cell_height.mean():.2f} nm +/- {cell_height.std():.2f} nm')
print(f'Average membrane thickness: {thickness:.2f} nm +/- {thicknesses.std():.2f} nm')

output_me['Avg Cell Length']=str(f'{cell_length:.2f}')
output_me['Avg Area/Lipid']=str(f'{area_per_lipid:.2f}')
output_me['Average cell height']=str(f'{cell_height.mean():.2f}')
output_me['Average Membrane thickness']=str(f'{thickness:.2f}')



# In[33]:


# Plot statistics of the cell dimensions
fig, ax = plt.subplots(ncols=2,nrows=3)

ax[0,0].plot(cell_len)
ax[0,1].hist(cell_len,bins=50)

ax[0,0].axhline(y=cell_length,c='red')
ax[0,0].axhline(y=cell_length-cell_len.std(),c='red',dashes=(2,2))
ax[0,0].axhline(y=cell_length+cell_len.std(),c='red',dashes=(2,2))

ax[0,1].axvline(x=cell_length,c='red')
ax[0,1].axvline(x=cell_length-cell_len.std(),c='red',dashes=(2,2))
ax[0,1].axvline(x=cell_length+cell_len.std(),c='red',dashes=(2,2))

ax[1,0].plot(cell_height)
ax[1,1].hist(cell_height,bins=50)

ax[1,0].axhline(y=cell_height.mean(),c='red')
ax[1,0].axhline(y=cell_height.mean()-cell_height.std(),c='red',dashes=(2,2))
ax[1,0].axhline(y=cell_height.mean()+cell_height.std(),c='red',dashes=(2,2))

ax[1,1].axvline(x=cell_height.mean(),c='red')
ax[1,1].axvline(x=cell_height.mean()-cell_height.std(),c='red',dashes=(2,2))
ax[1,1].axvline(x=cell_height.mean()+cell_height.std(),c='red',dashes=(2,2))

ax[2,0].plot(thicknesses)
ax[2,1].hist(thicknesses,bins=50)

fig.set_figwidth(8)


fig.savefig('Images/cell_statistics.png')

# In[ ]:





# # Fourier transform of non-uniform z

# ## Define the q-vectors

# In[34]:


cf =  2 * np.pi / cell_length

q_max = 1.0
qi_max = int(q_max / cf)

qi_max
print(f'q-vectors are truncated at {q_max} nm^-1 with maximum index {qi_max}')
print(f'Lowest q-vector: {cf:.3f} nm^-1')

output_me['Lowest q-vector']=str(f'{cf:.3f}')

# In[35]:


length_per_lipid = np.sqrt(area_per_lipid)
print(f'Length per lipid: {length_per_lipid:.2f} nm')

# Largest wavevector corresponds to the length per lipid
largest_q = 2 * np.pi / length_per_lipid
print(f'Largest q-vector with physical meaning: {largest_q:.2f} nm^-1')
print(f'Largest index of q-vector with physical meaning: {int(largest_q / cf)}')

output_me['Length per lipid']=str(f'{length_per_lipid:.2f}')

# In[36]:


# Generate the q-grid
q_range = np.arange(-qi_max, qi_max+1)
d_qx = np.tile(q_range, len(q_range))
d_qy = np.repeat(q_range, len(q_range))


# In[37]:


# Use pandas to store the results
d = pd.DataFrame(data={'i':d_qx,'j':d_qy})
d['qx'], d['qy'] = d['i'] * cf, d['j'] * cf
d['q2'] = d['qx']**2 + d['qy']**2
d['abs_q'] = np.sqrt(d['q2'])


# In[38]:


# Truncate the q-grid

# remove the q=0 point and the point exceeding q_max
d = d[(d['abs_q'] > 0) & (d['abs_q'] < q_max)]

# remove half of the unit circle because u(qi,qj) =
d = d[d['i'] >= 0]
d = d[~((d['j'] <= 0) & (d['i']==0))]
print(f'{d.shape[0]} q-vectors selected with q_max = {q_max:.2f} nm^-1 and maximum index {qi_max}')


# In[39]:


q_ij = d[['i','j']].values


# In[40]:


# Choose only q-vectors with one of the components equal to zero
# d = d[(d['i'] == 0) | (d['j'] == 0)]
# print(f'Number of q-vectors with one component equal to zero: {d.shape[0]}')


# In[41]:


# Generate the q-vectors
qs = d[['qx', 'qy']].values


# In[42]:


np.allclose(q_ij * 2 * np.pi / cell_length, qs )


# ## Compute fluctuations

# In[43]:


def fourier_transform5(pos, mean_z, N_at_layer, qs, density=False):
    # Take a snapshot of the membrane and perform a Fourier transform for a group

    # Identical but slower code
    # Follow eq. 10 from Brandt et al. 2011 (doi: 10.1016/j.bpj.2011.03.010)
    # xy, z = pos[:,0:2], pos[:,2]-mean_z
    # m = np.dot(qs, xy.T)
    # u = np.dot(np.exp(-1j * m) , z) / (N_at_layer)

    xy = np.array(pos[:,0:2], dtype=np.float32)
    if density:
        z = np.ones_like(pos[:,2], dtype=np.complex64)
    else:
        z = np.array(pos[:,2]-mean_z, dtype=np.complex64)
    qs2 = np.array(qs, dtype=np.float32)

    return finufft.nufft2d3(*xy.T, z, *qs2.T, isign=-1) / N_at_layer


# In[44]:


# 's' at the end of the variable name means 'sampled over time'
flucs = {}
for name in ['top', 'bottom', 'monolayer', 'cross', 'undulation', 'peristaltic', 'density']:
    flucs[name] = np.zeros((len(qs), N_sampled_frames))

h_bars = np.zeros(N_sampled_frames)
h2_u2_bars = np.zeros(N_sampled_frames)

print(f'Number of q-vectors: {qs.shape[0]}')
print(f'Number of frames: {N_sampled_frames}')
print(f'Number of atoms per layer: {top.n_atoms} and {bottom.n_atoms}')
print(f'Fluctuations to be computed: {list(flucs.keys())}')
print('All Fourier amplitudes will be divided by the number of atoms in the layer')

print(f'Starting Fourier transform...')
progress_bar = tqdm(total=N_sampled_frames)
for i, ts in enumerate(u.trajectory[indices]):

    # L_max = units.convert(ts.dimensions[0], 'angstrom', 'nm')
    # qs = q_ij * (2 * np.pi / L_max)

    pos_all = units.convert(P.positions, 'angstrom', 'nm')
    mean_z = pos_all[:,2].mean()

    pos_top = units.convert(top.positions, 'angstrom', 'nm')
    pos_bottom = units.convert(bottom.positions, 'angstrom', 'nm')

    h_bars[i] = 0.5 * (pos_top[:,2].mean() - pos_bottom[:,2].mean())
    h2_u2_bars[i] = np.mean((pos_all[:,2] - mean_z)**2)

    ft_top = fourier_transform5(pos_top, mean_z, top.n_atoms, qs)
    ft_bottom = fourier_transform5(pos_bottom, mean_z, bottom.n_atoms, qs)
    ft_undulations = 0.5 * (ft_top + ft_bottom)
    ft_peristaltics = 0.5 * (ft_top - ft_bottom)

    flucs['top'][:,i] = square_complex(ft_top)
    flucs['bottom'][:,i] = square_complex(ft_bottom)

    flucs['monolayer'][:,i] = 0.5 * (flucs['top'][:,i] + flucs['bottom'][:,i])
    flucs['cross'][:,i] = square_complex(ft_top, ft_bottom)
    flucs['undulation'][:,i] = square_complex(ft_undulations)
    flucs['peristaltic'][:,i] = square_complex(ft_peristaltics)

    dens_top = fourier_transform5(pos_top, mean_z, top.n_atoms, qs, density=True)
    dens_bottom = fourier_transform5(pos_bottom, mean_z, bottom.n_atoms, qs, density=True)
    dens_mean = 0.5 * (dens_top + dens_bottom) / area_per_lipid

    flucs['density'][:,i] = square_complex(dens_mean)

    progress_bar.update(1)

print('Done!')


# In[45]:


h_bar = h_bars.mean()
print(f'h_bar = {h_bar:.2f} nm +/- {h_bars.std():.2f} nm')

h2_u2_bar = h2_u2_bars.mean()
print(f'h2_u2_bar = {h2_u2_bar:.2f} nm^2 +/- {h2_u2_bars.std():.2f} nm^2')


# In[46]:


print('Density will be rescaled by (h_bar^2 + h2_u2_bar) * area_per_lipid**2 = {:.2f} nm^2'.format( h2_u2_bar * area_per_lipid**2))
flucs['density'] *= h2_u2_bar * area_per_lipid**2


# In[47]:


flucs_by_q = {}
for name, vs in flucs.items():
    unique_q, unique_q_vs = group_by_q(d['abs_q'], vs)
    flucs_by_q[name] = unique_q_vs


# In[48]:


# Get the time average of the fluctuations
fluc, fluc_std = {}, {}
fluc_by_q, fluc_std_by_q = {}, {}
for name, vs in flucs.items():
    fluc[name] = vs.mean(axis=1)
    fluc_std[name] = vs.std(axis=1)

    fluc_by_q[name] = flucs_by_q[name].mean(axis=1)
    fluc_std_by_q[name] = flucs_by_q[name].std(axis=1)


# In[49]:


# Confirm that the cross term is the difference of the undulation and peristaltic terms
b = np.allclose(fluc['cross'], fluc['undulation'] - fluc['peristaltic'])
print(f'Cross term is the difference of the undulation and peristaltic terms: {b}')


# ## Convergence of individual amplitudes

# In[50]:


from scipy import stats

data = flucs_by_q['undulation']
t_value = stats.t.ppf((1 + 0.90) / 2., data.shape[1]-1)
margin_of_error = t_value * data.std(ddof=1, axis=1) / np.sqrt(data.shape[1])


# In[ ]:





# In[51]:


27 // 10


# In[ ]:





# In[52]:


corr_width = 0.05

t_values = [stats.t.ppf((1 + 0.90) / 2., i-1) for i in range(len(indices))]

fig, ax = plt.subplots(1,4, figsize=(12, 3))
for i_q in range(4):
    x = flucs_by_q['undulation'][i_q]
    cmean = x.cumsum() / np.arange(1, len(x) + 1)

    i_time = np.arange(1, len(indices) + 1)
    t_values = stats.t.ppf((1 + 0.90) / 2., i_time-1)
    cumstd = np.zeros(len(indices))
    for i in range(len(x)):
        cumstd[i] = np.std(data[:i+1])
    margin_of_error = t_values * cumstd / np.sqrt(i_time)

   
    # plot cummean
    ax[i_q].plot(sampled_times_ns, margin_of_error / cmean)
 
    ax[i_q].set_xlabel('Time (ns)')

    ax[i_q].set_title(f'{unique_q[i_q]:.2f} nm$^{{-1}}$')
    ax[i_q].set_xscale('log')
    ax[i_q].set_yscale('log')
    # set

plt.tight_layout()
fig.savefig('Images/Figure1.png')


# In[53]:


corr_width = 0.05

t_values = [stats.t.ppf((1 + 0.90) / 2., i-1) for i in range(len(indices))]

fig, ax = plt.subplots(1,4, figsize=(12, 3))
for i_q in range(4):
    x = flucs_by_q['undulation'][i_q]
    cmean = x.cumsum() / np.arange(1, len(x) + 1)

   
    # plot cummean
    ax[i_q].plot(sampled_times_ns, cmean)
    # Add corridor for the +/- 1% of the final value
    ax[i_q].axhline(cmean[-1], color='k', linestyle='--')
    ax[i_q].axhspan(cmean[-1] * (1-corr_width), cmean[-1] * (1+corr_width), alpha=0.2, color='k', zorder=0)

    ax[i_q].set_xlabel('Time (ns)')
    #ax[i_q].set_ylabel('Cumulative mean')

    ax[i_q].set_title(f'{unique_q[i_q]:.2f} nm$^{{-1}}$')
    ax[i_q].set_xscale('log')
    # set

plt.tight_layout()
fig.savefig('Images/Figure2.png')



# ## Check the convergence using isotropy

# In[54]:


unique_abs_q, cv = cv_series(d['abs_q'], flucs['undulation'], maxq=1.0)


# In[55]:


print(f'Number of unique |q| values: {len(unique_abs_q)}')
print(f'Average coefficient of variation: {cv[:, -1].mean():.2f}')
print(f'Maximum coefficient of variation: {cv[:, -1].max():.2f}')
print('There is no correlation between the vector length and CV')


# In[56]:


plt.plot(sampled_times_ns, cv.max(axis=0), label='max')

plt.plot(sampled_times_ns, cv.mean(axis=0), label='mean')
# fit to the y=ax**b
popt, pcov = curve_fit(lambda x, a, b: a * x**b, sampled_times_ns[100:], cv.mean(axis=0)[100:])
plt.plot(sampled_times_ns, popt[0] * sampled_times_ns**popt[1], label='fit: $%.2f x^{%.2f}$' % tuple(popt))


#plt.plot(sampled_times_ns, cv[unique_abs_q.ravel()<.4,:].mean(axis=0), label='mean (|q| < 0.4 nm$^{-1}$)')

#plt.plot(sampled_times_ns, cv[0], label=f'lowest |q| = , {unique_abs_q[0]:.2f}')
#plt.plot(sampled_times_ns, cv[1], label=f'|q| = {unique_abs_q[1]:.2f}')
#plt.plot(sampled_times_ns, cv[2], label=f'|q| = {unique_abs_q[2]:.2f}')

plt.xlabel('Time (ns)')
plt.ylabel('Coefficient of variation')
plt.title('CV of $<u^2>$ for interval [0, t]')
plt.xscale('log')
plt.yscale('log')
plt.legend()


plt.savefig('Images/q-CV.png')


# ## $<u^2>$ vs $1/q^4$ trend

# ### All sampled frames

# In[57]:


fit_maxq = 0.8

bm = (unique_q <= fit_maxq)

x = unique_q[bm]
y = fluc_by_q['undulation'][bm]
a_fit = fit_over_q4(x, y)
print(f'<u^2> = a_fit/q^4, where a_fit = {a_fit :.3e}')

kappa = bending_constant_from_a_fit(a_fit, cell_area)
print(f'Bending constant: {kappa:.2f} kT')
output_me['kappa(q^-4)']=str(f'{kappa:.2f}')


# In[58]:


# fit y=a_fit/q^4 + b_fit/q^2
popt, pcov = curve_fit(lambda x, a, b: a/x**4 + b, x, y, p0=[1, 1], bounds=([0, 0], [np.inf, np.inf]))
print('Fit y=a_fit/q^4 + b_fit')
print(f'a_fit = {popt[0]:.3e}')
print(f'b_fit = {popt[1]:.3e}')
print(f'kappa = {bending_constant_from_a_fit(popt[0], cell_area):.2f} kT')


# In[59]:


from scipy import stats

data = flucs_by_q['undulation'][bm]
t_value = stats.t.ppf((1 + 0.90) / 2., data.shape[1]-1)
margin_of_error = t_value * data.std(ddof=1, axis=1) / np.sqrt(data.shape[1])
# Margin of error is ~3-4% of the amplitude value and seems independent of q-value
# plt.plot(unique_q[bm], margin_of_error / fluc_by_q['undulation'][bm], 'o')


# In[ ]:





# In[ ]:





# In[ ]:





# In[60]:


fig, ax = plt.subplots(figsize=(8,6))

ax.errorbar(unique_q[bm], fluc_by_q['undulation'][bm], yerr=margin_of_error, label='Undulation Mean 90% CI', fmt='o', capsize=3, capthick=1, markersize=0, elinewidth=1, color='C0')

ax.scatter(unique_q, fluc_by_q['undulation'], label='Undulation')
ax.scatter(unique_q, fluc_by_q['density'] * area_per_lipid**2 * h2_u2_bar, label='Density')
ax.scatter(unique_q, fluc_by_q['peristaltic'], label='Peristaltic')

ax.plot(unique_q[bm], a_fit / unique_q[bm]**4, 'k:', label='Fit $A_0 \\kappa q^{-4}$ (q < %.2f)' % fit_maxq)
ax.plot(unique_q[bm], popt[0] / unique_q[bm]**4 + popt[1], 'k--', label='Fit $A_0 \\kappa q^{-4} + B$ (q < %.2f)' % fit_maxq)

ax.axhline(h2_u2_bar/2 / top.n_atoms, color='k', linestyle='--')

ax.legend()

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel('q (nm$^{-1}$)')
ax.set_ylabel('$<u^2(q)>$ (nm$^2$)')

ax.set_title('q-vector $<u^2>$ contribution')

# Add grid lines
ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.5, axis='both', zorder=1, )
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray' )

# add minor tick labels
# from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
# ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))


fig.savefig('Images/avg-q-fit.png')

# ### Frames up to $t$

# In[61]:

##SAM START HERE AGAIN
def get_k_from_u2(u2, abs_q, cell_area, fit_maxq=0.8):

    d_by_q = pd.DataFrame({'abs_q': abs_q, 'u2_mean': u2}).groupby('abs_q').mean().reset_index()
    d_fit = d_by_q[d_by_q['abs_q'] < fit_maxq]

    a_fit = fit_over_q4(d_fit['abs_q'], d_fit['u2_mean'])
    kappa = bending_constant_from_a_fit(a_fit, cell_area)
    return kappa


# In[62]:


u2_cummean = cummean_columns(flucs['undulation'])


# In[63]:


ks = np.array([get_k_from_u2(u2_cummean[:,i], d['abs_q'], cell_area) for i in range(u2_cummean.shape[1])])


print(f'h_bar = {h_bar:.2f} nm +/- {h_bars.std():.2f} nm')

h2_u2_bar = h2_u2_bars.mean()
print(f'h2_u2_bar = {h2_u2_bar:.2f} nm^2 +/- {h2_u2_bars.std():.2f} nm^2')
# In[64]:


# Plot the bending constant as a function of the number of frames
fig, ax = plt.subplots(figsize=(8,6))

ax.plot(sampled_times_ns, ks, 'k:', label='Fit line')

# Add horizontal corridor for +/- 1 kT from the last value
last_k = ks[-1]
ax.axhline(last_k, color='k', linestyle='--', alpha=0.5, zorder=0)
ax.axhspan(last_k-1, last_k+1, alpha=0.2, color='k', zorder=0)
# add the description of the corridor, place text close to the bottom right corner
ax.text(0.95, 0.05, f'{last_k:.1f} $\\pm$ 1 kT', transform=ax.transAxes, va='bottom', ha='right', color='k', fontsize=14)

# add grid lines
ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.5, axis='both', zorder=1, )
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray' )

# increase the density of horizontal grid
ax.yaxis.set_minor_locator(AutoMinorLocator())

ax.legend()

ax.set_xlabel('Time (ns)')
ax.set_ylabel('Bending constant (kT)')
ax.set_title('Kappa at time',fontsize=20)
plt.xscale('log')
plt.savefig('Images/KappaVsTime.png')


# In[ ]:





# ## gamma-CU trend

# Following Tarazona2013, 10.1063/1.4818421
# Specifically:
# * Eqs 4-7 for definitions of the fluctuation amplitudes
# * Eqs 8, 10 for definition of the gamma function
# * Eq. 11 for fitting
# 
# Difference:
# * we use the Fourier transform over the atom positions, instead of the interpolated surface

# ### Fit u2-CU

# In[65]:


d['cross'] = fluc['cross']
d_fit = by_q(d, cols='cross', maxq=0.4)

d_fit = d_fit[d_fit['cross'] > 0]

a_fit = fit_over_q4(d_fit['abs_q'], d_fit['cross'])
kappa_cu = bending_constant_from_a_fit(a_fit, cell_area)
print(f'Bending constant from CU: {kappa_cu:.3f} kT')
print('Fitting line is shown in the next section')


# ### Plot u2 and gamma

# In[66]:


def gamma(f, q, cell_area, kT=1):
    return kT / (f * q**2 * cell_area)


# In[67]:


# Make side-by-side plots for fluc and gamma
fig, ax = plt.subplots(1, 2, figsize=(12,6))
ax[0].plot(d['abs_q'], fluc['undulation'], 'o', label='$<u^2_{U}>$', color='blue')
ax[0].plot(d['abs_q'], fluc['peristaltic'], 'o', label='$<u^2_{p}>$', color='red')
ax[0].plot(d['abs_q'], fluc['cross'], 'o', label='$<u^2_{CU}>$', color='green')
ax[0].plot(d['abs_q'], fluc['monolayer'], 'o', label='$<u^2_{m}>$', color='black')
ax[0].plot(d_fit['abs_q'], a_fit / d_fit['abs_q']**4, '--', label='Fit $(A_0 \\kappa q^4)^{-1}$ (q < %.2f)' % fit_maxq, color='green')

ax[0].set_xlabel('q (nm$^{-1}$)')
ax[0].set_ylabel('$<u^2(q)>$ (nm$^2$)')
ax[0].legend()
ax[0].set_xscale('log')
ax[0].set_yscale('log')

ax[1].plot(d['abs_q'], gamma(fluc['undulation'], d['abs_q'], cell_area), 'o', label='$\\gamma^{U}$', color='blue')
ax[1].plot(d['abs_q'], gamma(fluc['peristaltic'], d['abs_q'], cell_area), 'o', label='$\\gamma^{p}$', color='red')
ax[1].plot(d['abs_q'], gamma(fluc['cross'], d['abs_q'], cell_area), 'o', label='$\\gamma^{CU}$', color='green')
ax[1].plot(d['abs_q'], gamma(fluc['monolayer'], d['abs_q'], cell_area), 'o', label='$\\gamma^{m}$', color='black')

ax[1].set_xlabel('q (nm$^{-1}$)')
ax[1].set_ylabel('$\\gamma$ (kT)')
ax[1].legend()
ax[1].set_xscale('log')
ax[1].set_yscale('log')




# In[ ]:





# ### Fit gamma_CU

# In[68]:


# import curve_fit
from scipy.optimize import curve_fit

# fit gamma_CU(q) = gamma_0 + kappa * q**2 * (1 + (q/q_u)**alpha)
def fit_gamma_CU(q, gamma_0, kappa, q_u, alpha):
    #gamma_0 = 0.
    return gamma_0 + kappa * q**2 * (1 + (q/q_u)**alpha)

# choose points with q < 0.6

#d['gamma_CU'] = gamma(fluc['cross'], d['abs_q'], cell_area, kT)
#d_fit = by_q(d, cols='gamma_CU', maxq=0.58)

#popt, pcov = curve_fit(fit_gamma_CU, d_fit['abs_q'], d_fit['gamma_CU'], p0=[0,20*kT, 1., 4])


# In[69]:


#print(f'gamma_0 = {popt[0] / kT:.2f} kT/nm^2')
#print(f'kappa = {popt[1] / kT:.2f} kT')
#print(f'q_u = {popt[2]:.2f} nm^-1')
#print(f'alpha = {popt[3]:.2f}')


# In[70]:


#plt.plot(d_fit['abs_q'], d_fit['gamma_CU'] * beta, 'o')
#plt.plot(d_fit['abs_q'], fit_gamma_CU(d_fit['abs_q'], *popt) * beta)
# limit x range to [0, 0.5)
#plt.xlim(0.1, 0.58)
#plt.ylim(0,50)


# ## BW-DCF

# 10.1021/acs.jctc.2c00099

# ### Parameters

# In[71]:


bw_loaded = np.load('bw-dcf.npz')

# ## C matrix loaded from npz file produced by bw-dcf.py

# In[88]:


C = bw_loaded['C']


# In[91]:


d['C11'] = C[:,-1]


# In[92]:


d.shape


# In[ ]:





# ### Plot the results

# In[93]:


d['bg'] = 1 / (d['q2'] * d['C11'])
d['q4C11'] = d['q2']**2 * d['C11']


# In[94]:


# group by the absolute value of q, collect mean and std
d_abs_q = d.groupby('abs_q').agg({'bg': ['mean', 'std'], 'q4C11': ['mean','std']}).reset_index()


# In[95]:


# plot the means and std
fig, ax = plt.subplots(ncols=2, figsize=(12,4))

ax[0].errorbar(d_abs_q['abs_q'], d_abs_q['bg']['mean'], yerr=d_abs_q['bg']['std'], fmt='o')
#ax[0].scatter(d['abs_q'], 1/(d['q2']*d['C11']), c=d['qx'], cmap='viridis')

ax[0].set_xlabel('q, 1/nm')
ax[0].set_ylabel('$\\beta\\gamma^{CU}(q)$ = $1/(q^2*C_{11})$')

ax[1].errorbar(d_abs_q['abs_q'], d_abs_q['q4C11']['mean'], yerr=d_abs_q['q4C11']['std'], fmt='o')
#ax[1].scatter(d['abs_q'], (d['q2']**2 * d['C11']), c=d['qx'], cmap='viridis')

ax[1].set_xlabel('q, 1/nm')
ax[1].set_ylabel('$q^2 / \\beta \\gamma^{CU}(q)$ = $q^4 C_{11}$')




# ### Fit the q4C11 curve

# In[96]:


d_fit = d_abs_q[d_abs_q['abs_q'] < 1.0]


# In[97]:


# do the fitting of d_abs_q['q4C11']['mean'] vs d_abs_q['abs_q'] according to eq. 19 in the paper
def fit_func(q, kappa, kappa_th, qd):
    return (1/kappa + q**2/kappa_th) * (1 - (q / qd)**4)


# In[98]:


model = curve_fit(fit_func, d_fit['abs_q'], d_fit['q4C11']['mean'], p0=[40, 100, 2], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))


# In[99]:


plt.errorbar(d_fit['abs_q'], d_fit['q4C11']['mean'], yerr=d_fit['q4C11']['std'], fmt='o', label='MD')
#plt.plot(d_abs_q['abs_q'], d_abs_q['q4C11']['mean'], 'o')
plt.plot(d_fit['abs_q'], fit_func(d_fit['abs_q'], *model[0]))
plt.xlabel('q, 1/nm')
plt.ylabel('$q^2 / \\beta \\gamma^{CU}(q)$ = $q^4 C_{11}$')
plt.title('BW-DCF Fit',fontsize=20)
plt.savefig('Images/BW-DCF-fit.png')



# ### Bending modulus

# In[100]:


k, k_theta, qd = model[0]
k_theta *= coef_k_theta

print(f'k = {k:.2f} kT')
print(f'k_theta = {k_theta:.2f} mN/m^2')
print(f'qd = {qd:.2f} nm^-1')

output_me['Kappa-BWDCF']=(f'{k:.2f}')


# In[ ]:





# In[ ]:





# ### Notes/ideas

# # Auto-correlation function

# In[72]:


def compute_autocorr(x, decorr_threshold=0.):
    x = (x - np.mean(x)) / (np.std(x) * np.sqrt(len(x)))
    autocorr_f = np.correlate(x, x, mode='full')
    autocorr_f = autocorr_f[autocorr_f.size // 2:]

    return autocorr_f, np.argmax(autocorr_f < decorr_threshold)


# In[ ]:





# In[73]:


def extract_decorrelation_time(sampled_times_ns, data):
    decors = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        _, decorr_i = compute_autocorr(data[i,:])
        # find the index of the max
        decors[i] = sampled_times_ns[decorr_i] - sampled_times_ns[0]

    return decors


# In[74]:


dc = extract_decorrelation_time(sampled_times_ns, flucs_by_q['undulation'])

plt.plot(unique_abs_q, dc, 'o-')

plt.xlabel('q, 1/nm')
plt.ylabel('Decorrelation time, ns')
plt.title('Autocorrelation Time',fontsize=20)
plt.savefig('Images/Autocorrelation.png')



# # FFT of interpolated z

# ### Define the q-grid

# In[75]:


N_points = 50

largest_q_to_keep = 2.0 # nm^-1


# In[76]:


qx = 2 * np.pi * np.fft.fftfreq(N_points, d=cell_length/N_points)
qy = 2 * np.pi * np.fft.fftfreq(N_points, d=cell_length/N_points)
QX, QY = np.meshgrid(qx, qy, indexing='ij')


# In[77]:


print(f'Largest q-vector: {np.sqrt(QX.max()**2 + QY.max()**2):.2f} nm^-1')


# In[78]:


data = {
    "qx": QX.flatten(),
    "qy": QY.flatten(),
}

d_grid = pd.DataFrame(data)
d_grid['abs_q'] = np.sqrt(d_grid['qx']**2 + d_grid['qy']**2)


# In[79]:


q_mask = (d_grid['abs_q'] <= largest_q_to_keep) & (d_grid['abs_q'] > 0)

# remove half of the unit circle because u(qx,qy) = u(-qx,-qy)
q_mask = q_mask & (d_grid['qx'] >= 0)
q_mask = q_mask & ~((d_grid['qy'] <= 0) & (d_grid['qx']==0))

print(f'Keeping q-vectors up to {largest_q_to_keep:.2f} nm^-1 and taking into account symmetry')
print(f'Number of q-vectors to keep: {q_mask.sum()} out of {len(d_grid)}')

d_grid = d_grid[q_mask]


# ### Fast Fourier Transform

# In[80]:


b_Load = False
if b_Load and os.path.isfile('fft_result.npy'):
    fft_result = np.load('fft_result.npy')


# In[81]:


grid_flucs = {}
for name in ['top', 'bottom', 'monolayer', 'cross', 'undulation', 'peristaltic']:
    grid_flucs[name] = np.zeros((sum(q_mask), N_sampled_frames))

print(f'Number of q-vectors to compute: {N_points} x {N_points} = {N_points**2}')
print(f'Number of q-vectors to keep: {q_mask.sum()}')
print(f'Number of frames: {N_sampled_frames}')
print(f'Fluctuations to be computed: {list(grid_flucs.keys())}')
print('All Fourier amplitudes will be divided by the number of grid points')


# In[82]:


progress_bar = tqdm(total=len(indices))

for i,ts in enumerate(u.trajectory[indices]):

    # Get the cell length and the average z position of the membrane
    L_max = units.convert(ts.dimensions[0], 'angstrom', 'nm')
    z_mean = units.convert(P.positions[:,2].mean(), 'angstrom', 'nm')

    # Generate a grid of points in the xy plane
    xy_new = create_grid(L_max, N_points)

    # Top layer
    interpolated_z = griddata(*expand_cell(top, z_mean, by=0.1), xy_new, method='nearest').reshape((N_points, N_points))
    ft_top = np.fft.fft2(interpolated_z).ravel()[q_mask] / N_points**2

    # Bottom layer
    interpolated_z = griddata(*expand_cell(bottom, z_mean, by=0.1), xy_new, method='nearest').reshape((N_points, N_points))
    ft_bottom = np.fft.fft2(interpolated_z).ravel()[q_mask] / N_points**2

    ft_undulations = 0.5 * (ft_top + ft_bottom)
    ft_peristaltics = 0.5 * (ft_top - ft_bottom)

    grid_flucs['top'][:,i] = square_complex(ft_top)
    grid_flucs['bottom'][:,i] = square_complex(ft_bottom)

    grid_flucs['monolayer'][:,i] = 0.5 * (grid_flucs['top'][:,i] + grid_flucs['bottom'][:,i])
    grid_flucs['cross'][:,i] = square_complex(ft_top, ft_bottom)
    grid_flucs['undulation'][:,i] = square_complex(ft_undulations)
    grid_flucs['peristaltic'][:,i] = square_complex(ft_peristaltics)

    progress_bar.update(1)

print('Done!')


# In[83]:


grid_flucs_by_q = {}
for name, vs in grid_flucs.items():
    grid_abs_q, grid_flucs_by_q[name] = group_by_q(d_grid['abs_q'], vs)


# In[84]:


# Get the time average of the fluctuations
grid_fluc = {}
grid_fluc_by_q = {}
for name, vs in grid_flucs.items():
    grid_fluc[name] = vs.mean(axis=1)
    grid_fluc_by_q[name] = grid_flucs_by_q[name].mean(axis=1)


# In[ ]:





# In[85]:


d_grid['u2'] = grid_fluc['undulation']


# In[86]:


# bin by q
n_bins = 1000

d_grid['q_bin'] = pd.cut(d_grid['abs_q'], bins=n_bins)
# get the average in each bin


# In[87]:


plot_d = by_q(d_grid, 'u2', maxq=2)


# In[88]:


# plot d_by_q and df_by_q
fig, ax = plt.subplots()
ax.scatter(plot_d['abs_q'], plot_d['u2'])

ax.scatter(unique_q, fluc_by_q['undulation'])

# Plot major and minor grid lines
ax.grid(visible=True, which='major', color='k', linestyle='-', alpha=0.5)
ax.grid(visible=True, which='minor', color='k', linestyle='-', alpha=0.1)

ax.set_xlabel('q')
ax.set_ylabel('u2')

ax.set_xscale('log')
ax.set_yscale('log')





# In[89]:


# Confirm that the cross term is the difference of the undulation and peristaltic terms
b = np.allclose(fluc['cross'], fluc['undulation'] - fluc['peristaltic'])
print(f'Cross term is the difference of the undulation and peristaltic terms: {b}')


# In[90]:


from sklearn.linear_model import LinearRegression

fig, ax = plt.subplots(1,4, figsize=(12, 3))
for i in range(4):

    x = flucs_by_q['undulation'][i]
    y = grid_flucs_by_q['undulation'][i]

    ax[i].scatter(x, y, s=1)
    ax[i].set_xlabel('u2 (non-uniform)')
    ax[i].set_ylabel('u2 (uniform)')
    ax[i].set_title(f'q = {grid_abs_q[i]:.2f}')



    lm = LinearRegression()
    lm.fit(x.reshape(-1,1), y)

    # plot fit
    x_fit = np.linspace(x.min(), x.max(), 2)
    y_fit = lm.predict(x_fit.reshape(-1,1))
    ax[i].plot(x_fit, y_fit, color='k', linestyle='--', label=f'Slope: {lm.coef_[0]:.3f}')

    ax[i].legend()

plt.tight_layout()
plt.savefig('Images/q-vector-fitting.png')



# ## Convergence of individual amplitudes

# In[91]:


corr_width = 0.05

fig, ax = plt.subplots(1,4, figsize=(12, 3))
for i_q in range(4):
    x = grid_flucs_by_q['undulation'][i_q]
    cmean = x.cumsum() / np.arange(1, len(x) + 1)

    # plot cummean
    ax[i_q].plot(sampled_times_ns, cmean)
    # Add corridor for the +/- 1% of the final value
    ax[i_q].axhline(cmean[-1], color='k', linestyle='--')
    ax[i_q].axhspan(cmean[-1] * (1-corr_width), cmean[-1] * (1+corr_width), alpha=0.2, color='k', zorder=0)

    ax[i_q].set_xlabel('Time (ns)')
    #ax[i_q].set_ylabel('Cumulative mean')

    ax[i_q].set_title(f'{unique_q[i_q]:.2f} nm$^{{-1}}$')
    ax[i_q].set_xscale('log')
    x_fit = np.linspace(x.min(), x.max(), 2)
    # set

plt.tight_layout()
plt.savefig('Images/q-vector-convergence.png')



# ## Check the convergence using isotropy

# In[92]:


unique_abs_q, cv = cv_series(d_grid['abs_q'], grid_flucs['undulation'], maxq=1.0)


# In[93]:


print(f'Number of unique |q| values: {len(unique_abs_q)}')
print(f'Average coefficient of variation: {cv[:, -1].mean():.2f}')
print(f'Maximum coefficient of variation: {cv[:, -1].max():.2f}')
print('There is no correlation between the vector length and CV')


# In[94]:


plt.plot(sampled_times_ns, cv.max(axis=0), label='max')
plt.plot(sampled_times_ns, cv.mean(axis=0), label='mean')
plt.xlabel('Time (ns)')
plt.ylabel('Coefficient of variation')
plt.title('CV of $<u^2>$ for interval [0, t]')
plt.yscale('log')
plt.xscale('log')
plt.legend()



# ## Fit u2-CU

# In[95]:


u2_maxq_fit = 0.6

d_grid['undulation'] = grid_fluc['undulation']

d_fit_u2 = by_q(d_grid, cols=['undulation'], maxq=u2_maxq_fit)
a_fit_u2 = fit_over_q4(d_fit_u2['abs_q'], d_fit_u2['undulation'])
kappa_u2 = bending_constant_from_a_fit(a_fit_u2, cell_area)
print(f'Bending constant from u2: {kappa_u2:.1f} kT')
print('Fitting line is shown in the next section')
output_me['Kappa_u2-cu']=(f'{kappa_u2:.1f}')


# In[96]:


cross_maxq_fit = 1.

d_grid['cross'] = grid_fluc['cross']

d_fit_cu = by_q(d_grid, cols=['cross'], maxq=cross_maxq_fit)
a_fit_cu = fit_over_q4(d_fit_cu['abs_q'], d_fit_cu['cross'])
kappa_cu = bending_constant_from_a_fit(a_fit_cu, cell_area)
print(f'Bending constant from CU: {kappa_cu:.1f} kT')


# ## Plot u2 and gamma

# In[97]:


data = {'abs_q': d_grid['abs_q'],
        'undulation': grid_fluc['undulation'],
        'peristaltic': grid_fluc['peristaltic'],
        'cross': grid_fluc['cross'],
        'monolayer': grid_fluc['monolayer'],
        'gamma_undulation': gamma(grid_fluc['undulation'], d_grid['abs_q'], cell_area),
        'gamma_peristaltic': gamma(grid_fluc['peristaltic'], d_grid['abs_q'], cell_area),
        'gamma_cross': gamma(grid_fluc['cross'], d_grid['abs_q'], cell_area),
        'gamma_monolayer': gamma(grid_fluc['monolayer'], d_grid['abs_q'], cell_area),
        }

plot_d = pd.DataFrame(data)


# In[98]:


plot_d['bin'] = pd.cut(plot_d['abs_q'], bins=100)
plot_d = plot_d.groupby('bin').mean().reset_index()
plot_d = plot_d[~plot_d['undulation'].isna()]


# In[99]:


# Make side-by-side plots for grid_fluc and gamma
fig, ax = plt.subplots(1, 2, figsize=(12,6))
ax[0].plot(plot_d['abs_q'], plot_d['undulation'], 'o', label='$<u^2_{U}>$', color='blue')
ax[0].plot(plot_d['abs_q'], plot_d['peristaltic'], 'o', label='$<u^2_{p}>$', color='red')
ax[0].plot(plot_d['abs_q'], plot_d['cross'], 'o', label='$<u^2_{CU}>$', color='green')
ax[0].plot(plot_d['abs_q'], plot_d['monolayer'], 'o', label='$<u^2_{m}>$', color='black')

ax[0].plot(d_fit_cu['abs_q'], a_fit_cu / (d_fit_cu['abs_q']**4), label='$q^{-4}$ (qmax = %.2f)' % cross_maxq_fit, color='green', linestyle='--')
ax[0].plot(d_fit_u2['abs_q'], a_fit_u2 / (d_fit_u2['abs_q']**4), label='$q^{-4}$ (qmax = %.2f)' % u2_maxq_fit, color='blue', linestyle='--')

ax[1].plot(plot_d['abs_q'], plot_d['gamma_undulation'], 'o', label='$\\gamma_{U}$', color='blue')
ax[1].plot(plot_d['abs_q'], plot_d['gamma_peristaltic'], 'o', label='$\\gamma_{p}$', color='red')
ax[1].plot(plot_d['abs_q'], plot_d['gamma_cross'], 'o', label='$\\gamma_{CU}$', color='green')
ax[1].plot(plot_d['abs_q'], plot_d['gamma_monolayer'], 'o', label='$\\gamma_{m}$', color='black')

ax[0].set_xlabel('q')
ax[0].set_ylabel('$<u^2>$')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].legend()

ax[1].set_xlabel('q')
ax[1].set_ylabel('$\\gamma$')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
#ax[1].set_xlim(0, 3)
#ax[1].set_ylim(0, 50)
ax[1].legend()




# ## Fit gamma_CU

# In[100]:


def fit_gamma_CU(q, gamma_0, kappa, q_u, alpha):
    gamma_0 = 0.
    return gamma_0 + kappa * q**2 * (1 + (q/q_u)**alpha)


# In[101]:


gamma_cu_qmax = 1.0

d_grid['gamma_CU'] = gamma(grid_fluc['cross'], d_grid['abs_q'], cell_area, kT)

d_fit = by_q(d_grid, cols='gamma_CU', maxq=gamma_cu_qmax)

popt, pcov = curve_fit(fit_gamma_CU, d_fit['abs_q'], d_fit['gamma_CU'], p0=[0,20*kT, .4, 4])


# In[102]:


print(f'gamma_0 = {popt[0] / kT:.2f} kT/nm^2')
print(f'kappa = {popt[1] / kT:.2f} kT')
print(f'q_u = {popt[2]:.2f} nm^-1')
print(f'alpha = {popt[3]:.2f}')

output_me['Kappa_gamma_cu']=(f'{popt[1] / kT:.2f}')


# In[103]:


plt.plot(d_fit['abs_q'], d_fit['gamma_CU'], 'o')
plt.plot(d_fit['abs_q'], fit_gamma_CU(d_fit['abs_q'], *popt))
# limit x range to [0, 0.5)
#plt.xlim(0.1, 0.58)
#plt.ylim(0,50)
plt.savefig('Images/gamma_cu.png')



#END OF MODIFIED BENDING ANALYSIS SCRIPT; START OF RSF ANALYSIS



# In[104]

## RSF Analysis

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
              'DRPC':{'head':'PO4','tail':'D6A D6B', 'med':'?1?'},
              'DBPC':{'head':'PO4','tail':'C5A C5B', 'med':'?1?'},
              'PAPC':{'head':'PO4','tail':'C5A C4B', 'med':'?1?'},
              'PEPC':{'head':'PO4','tail':'C5A C4B', 'med':'?1?'},
              'POPC':{'head':'PO4','tail':'C4A C4B', 'med':'C1*'},
              'PUPC':{'head':'PO4','tail':'D5A C4B', 'med':'?1?'},
              'DOPC':{'head':'PO4','tail':'C4A C4B', 'med':'?1?'},
              'DAPC':{'head':'PO4','tail':'C5A C5B', 'med':'?1?'},
              'LPPC':{'head':'PO4','tail':'C4A C3B', 'med':'?1?'},
              'PGPC':{'head':'PO4','tail':'C5A C4B', 'med':'?1?'},
              'PRPC':{'head':'PO4','tail':'D6A C4B', 'med':'?1?'},
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
              'BNSM':{'head':'PO4','tail':'C4A C6B', 'med':'?1?'},
              'DBSM':{'head':'PO4','tail':'C4A C5B', 'med':'?1?'},
              'DPSM':{'head':'PO4','tail':'C3A C4B', 'med':'?1?'},
              'DXSM':{'head':'PO4','tail':'C5A C6B', 'med':'?1?'},
              'PGSM':{'head':'PO4','tail':'C3A C5B', 'med':'?1?'},
              'PNSM':{'head':'PO4','tail':'C3A C6B', 'med':'?1?'},
              'POSM':{'head':'PO4','tail':'C3A C4B', 'med':'?1?'},
              'PVSM':{'head':'PO4','tail':'C3A C4B', 'med':'?1?'},
              'XNSM':{'head':'PO4','tail':'C5A C6B', 'med':'?1?'},
              'CHOL':{'head':'ROH','tail':'C2','med':'?1?'},
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
u_leaf=u
u_leaf.trajectory[100]
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

selection_string=' or '.join(lipid_composition)
lipid = u.select_atoms(selection_string)


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

        if self.method_normals == 'rotation':
            return self.compute_N_rotation(neighbor_pos, neighbor_masses)
        elif self.method_normals == 'SVD':
            return self.compute_N_SVD(neighbor_pos)
        else:
            raise ValueError('Invalid compute method')

    def compute(self, i):
        ri = self.P[i].residue
        ri_tail = ri.atoms.select_atoms('name %s' % self.lipid_ends[ri.resname]['tail'])
        ri_head = ri.atoms.select_atoms('name %s' % self.lipid_ends[ri.resname]['head'])
        ri_med = ri.atoms.select_atoms('name %s' % self.lipid_ends[ri.resname]['med'])

        self.p_tail[i, :] = ri_tail.center_of_mass()
        self.p_head[i, :] = ri_head.center_of_mass()
        self.p_med[i, :] = ri_med.center_of_mass()

        self.n[i, :] = self.p_head[i, :] - self.p_tail[i, :]
        n_length = np.linalg.norm(self.n[i, :])
        assert n_length < 40, f'Director {i} too long: {n_length}'
        self.n[i, :] /= n_length

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
    neighbors_for_S, neighbors_for_N = get_neighbors(P, methods=[('nearest', 5), ('nearest', 10)])
    struct = StructuralParameters(P, neighbors_for_N, lipid_ends, method_normals='SVD')

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


Probabilities, bins = np.histogram(np_Ss[20:-20], bins=50, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2


# In[ ]:


# Fit Gaussian
from scipy.optimize import curve_fit
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
popt, pcov = curve_fit(gaussian, bin_centers, Probabilities, p0=[1, 0, 1])


# In[ ]:


sigma2 = popt[2]**2
area_per_lipid_A2 = area_per_lipid * 100
# Calculate the bending modulus
Kc = 1 / (sigma2 * area_per_lipid_A2)
Kc=2*Kc
print(f'Bending modulus: {Kc:.2f} kT')

output_me['kappa-rsf']=(f'{Kc:.2f}')


# In[ ]:


plt.plot(bin_centers, Probabilities)
plt.plot(bin_centers, gaussian(bin_centers, *popt))
plt.xlabel('S, 1/A')
plt.ylabel('Probability density, unscaled')
plt.savefig('Images/RSF.png')


# In[ ]:
#Write output

output_keys=output_me.keys()
output_vals=output_me.values()

with open (write_to,'w') as f:
    f.write(f'\n')
    for i in output_vals:
        f.write(f'{i}\t')

