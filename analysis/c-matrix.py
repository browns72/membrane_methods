import MDAnalysis as mda
from MDAnalysis import units
from MDAnalysis.analysis.leaflet import LeafletFinder

import dask
from dask.distributed import Client

from scipy.optimize import curve_fit

from time import time

import pandas as pd
import os, sys

import numpy as np
import argparse
import logging

import pickle

def load_command_line_args():
    # Load the command line parameters
    parser = argparse.ArgumentParser(description='Run the BW-DCF algorithm on a given file')
    # Load tpr file and one or more xtr files
    parser.add_argument('-t', '--tpr', type=str, required=True, help='The tpr file')
    parser.add_argument('-x', '--xtc', type=str, required=True, nargs='+', help='The xtc files')

    # Define the output file name
    parser.add_argument('-o', '--output', type=str, required=False, default='bw-dcf', help='The output file name')

    # Load the parameters for the algorithm
    parser.add_argument('-s', '--start', type=int, required=False, help='The start frame')
    # end frame, default is the last frame
    parser.add_argument('-e', '--end', type=int, required=False, help='The end frame')

    # numer of frames
    parser.add_argument('-n', '--nframes', type=int, required=False, help='The number of frames')

    # number of bins
    parser.add_argument('-b', '--bins', type=int, required=False, default=50,
                        help='The number of bins')

    # max value for q-vector
    parser.add_argument('-qmax', '--qmax', type=float, required=False, default=2.0,
                        help='The maximum value for the q-vector in nm^-1')

    # number of derivatives to use
    parser.add_argument('-d', '--derivatives', type=int, required=False, default=25,
                        help='The number of derivatives to use')

    # number of CPUs to use
    parser.add_argument('-c', '--cpus', type=int, required=False, default=1,
                        help='The number of CPUs to use')


    return parser.parse_args()

def log_args(args):
    logger.info("Command line arguments:")
    logger.info("tpr file: {}".format(args.tpr))
    logger.info("xtc files: {}".format(args.xtc))
    logger.info("output file: {}.npz".format(args.output))
    logger.info("start frame: {}".format(args.start))
    logger.info("end frame: {}".format(args.end))
    logger.info("number of frames: {}".format(args.nframes))
    logger.info("number of derivatives: {}".format(args.derivatives))
    logger.info("number of CPUs: {}".format(args.cpus))
    logger.info("number of bins: {}".format(args.bins))
    logger.info("maximum value for q-vector: {}".format(args.qmax))
    logger.info("========================================")

def setup_logger():
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Create a file handler
    if os.path.exists('bw-dcf.log.bak'):
        os.remove('bw-dcf.log.bak')
    if os.path.exists('bw-dcf.log'):
        os.rename('bw-dcf.log', 'bw-dcf.log.bak')

    handler = logging.FileHandler('bw-dcf.log')
    handler.setLevel(logging.INFO)
    # Create a logging format
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(handler)

    # log uncaught errors
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    return logger

def load_universe(args):
    # Load the universe

    logger.info("Loading universe")

    u = mda.Universe(args.tpr, args.xtc)
    time_moments = np.linspace(0, u.trajectory.n_frames * u.trajectory.dt, u.trajectory.n_frames + 1)

    # Log the number of frames
    logger.info("Number of frames: {}".format(u.trajectory.n_frames))
    # Log the number of atoms
    logger.info("Number of atoms: {}".format(u.atoms.n_atoms))
    # Log the cell dimensions
    logger.info("Cell dimensions: {}".format(u.dimensions[:3]))
    # Log the time information, dt
    logger.info("Time step between snapshot: {:.2f} ps".format(u.trajectory.dt))
    logger.info("Trajectory time range: {:.2f} ps to {:.2f} ps".format(time_moments[0], time_moments[-1]))

    logger.info("========================================")

    return u, time_moments

def sample_frames(u, args):
    # Set up frame indices for sampling

    logger.info("Sampling frames")

    if args.start is None:
        start = 0
    else:
        start = args.start
    if args.end is None:
        end = u.trajectory.n_frames - 1
    else:
        end = args.end
    if args.nframes is None:
        nframes = end - start + 1
    else:
        nframes = args.nframes

    logger.info("Sampling frames from {} to {}".format(start, end))
    logger.info('Sampling {} frames'.format(nframes))

    indices = np.linspace(start, end, nframes, dtype=int)
    sampled_time_ns = units.convert(time_moments[indices], 'ps', 'ns')
    logger.info("Sampled time range: {:.2f} ns to {:.2f} ns".format(sampled_time_ns[0], sampled_time_ns[-1]))
    logger.info("Time step between sampled snapshots: {} ns".format(sampled_time_ns[1] - sampled_time_ns[0]))

    logger.info("========================================")

    return indices, sampled_time_ns

def cell_properties(u, indices, n_atoms):
    """
    Compute the properties of the cell
    :param u: universe
    :param indices: frame indices for sampling
    :param n_atoms: number of lipids per leaflet
    :return: dict of properties
    """

    logger.info("Computing cell properties")

    props = {'n_atoms': n_atoms}

    N_sampled_frames = len(indices)
    cell_lens = np.zeros((N_sampled_frames))
    for i, ts in enumerate(u.trajectory[indices]):
        cell_lens[i] = units.convert(u.dimensions[0], 'A', 'nm')

    props['cell_length'] = cell_lens.mean()
    logger.info("Cell length: {:.2f} nm".format(props['cell_length']))

    props['cell_area'] = props['cell_length'] ** 2
    logger.info("Cell area: {:.2f} nm^2".format(props['cell_area']))

    props['area_per_lipid'] = props['cell_area'] / n_atoms
    logger.info("Area per lipid: {:.2f} nm^2".format(props['area_per_lipid']))

    props['rho_average'] = n_atoms / props['cell_area']
    logger.info("Average density: {:.2f} nm^-2".format(props['rho_average']))

    props['length_per_lipid'] = np.sqrt(props['area_per_lipid'])
    logger.info("Length per lipid: {:.2f} nm".format(props['length_per_lipid']))

    logger.info("========================================")

    return props

def compute_q_vectors(args, cell_props):
    # Define the q-space

    logger.info("Computing q-space")

    cf = 2 * np.pi / cell_props['cell_length']
    logger.info("Smallest q-vector: {:.2f} nm^-1".format(cf))

    q_max = args.qmax
    logger.info("Maximum q-vector: {:.2f} nm^-1".format(q_max))

    qi_max = int(np.ceil(q_max / cf))
    logger.info("Maximum q-vector index: {}".format(qi_max))

    # Generate the q-grid
    q_range = np.arange(-qi_max, qi_max + 1)
    d_qx = np.tile(q_range, len(q_range))
    d_qy = np.repeat(q_range, len(q_range))

    d = pd.DataFrame(data={'i': d_qx, 'j': d_qy})
    d['qx'], d['qy'] = d['i'] * cf, d['j'] * cf
    d['q2'] = d['qx'] ** 2 + d['qy'] ** 2
    d['abs_q'] = np.sqrt(d['q2'])

    # remove the q=0 point and the point exceeding q_max
    d = d[(d['abs_q'] > 0) & (d['abs_q'] < q_max)]

    # remove half of the unit circle because u(qi,qj) =
    d = d[d['i'] >= 0]
    d = d[~((d['j'] <= 0) & (d['i'] == 0))]

    logger.info(f'{d.shape[0]} q-vectors selected with q_max = {q_max:.2f} nm^-1 and maximum index {qi_max}')

    q_ij = d[['i', 'j']].values
    qs = d[['qx', 'qy']].values

    logger.info("========================================")

    return qs

def collect_z_positions(u, indices, top, bottom, P):
    # Collect the z-positions of the top and bottom leaflets

    logger.info("Collecting z-positions of the top and bottom leaflets")

    N_sampled_frames = len(indices)
    traj_zp = np.zeros((N_sampled_frames, top.n_atoms))
    traj_zm = np.zeros((N_sampled_frames, bottom.n_atoms))
    for i, ts in enumerate(u.trajectory[indices]):
        traj_zp[i,:] = units.convert(top.positions[:,2] - P.positions[:,2].mean(), 'A', 'nm')
        traj_zm[i,:] = units.convert(bottom.positions[:,2] - P.positions[:,2].mean(), 'A', 'nm')

    logger.info("z_p.mean = {:.2f} nm +/- {:.2f} nm".format(traj_zp.mean(), traj_zp.std()))
    logger.info("z_m.mean = {:.2f} nm +/- {:.2f} nm".format(traj_zm.mean(), traj_zm.std()))

    logger.info("========================================")

    return traj_zp, traj_zm

def gau_plus(x, rho0, d, alpha):
    return rho0 / np.sqrt(2*np.pi * alpha) * np.exp(-(x - d/2)**2 / (2 * alpha))

def gau_minus(x, rho0, d, alpha):
    return rho0 / np.sqrt(2*np.pi * alpha) * np.exp(-(x + d/2)**2 / (2 * alpha))

def calculate_rho_deriv(z, gauss, popt, N_deriv=10, sign=1.):
    # Calculate the derivatives of the gaussian fit to the density profile
    # sign = 1 for the top leaflet, -1 for the bottom leaflet
    # N_deriv = number of derivatives to calculate
    # z = grid of z values
    # gauss = function to calculate the gaussian
    # popt = parameters of the gaussian fit
    # returns a grid of derivatives of the gaussian fit
    # deriv_grid[0,:] = rho(z)
    # deriv_grid[1,:] = d rho(z) / dz
    # deriv_grid[2,:] = d^2 rho(z) / dz^2
    # etc.

    deriv_grid = np.zeros((N_deriv+1, len(z)))
    rho0, d, alpha = popt
    deriv_grid[0,:] = gauss(z, *popt)
    deriv_grid[1,:] = -(z - sign * d/2) / alpha * deriv_grid[0,:]
    for i in range(2, N_deriv+1):
        deriv_grid[i,:] =  - (z - sign * d/2) / alpha * deriv_grid[i-1,:] - (i-1) / alpha * deriv_grid[i-2,:]

    return deriv_grid

def compute_histogram(traj_z, n_bins, cell_props, N_sampled_frames):
    rho, z_edges = np.histogram(traj_z, bins=n_bins, density=False)
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    delta_z = z_edges[1] - z_edges[0]
    rho = rho / (cell_props['cell_area'] * N_sampled_frames * delta_z)
    return z_centers, rho

# calculate the A matrix using insights from symbolic algebra
def calculate_A_sym(alpha, rho0, N_deriv=10):
    A = np.zeros((N_deriv, N_deriv))
    A[0,0] = 1 / 4 / np.sqrt(np.pi * alpha**3) * rho0**2
    for i in range(1,N_deriv):
        A[i,i] = A[i-1,i-1] * (2*(i+1) - 1) / (2*alpha)
        for j in range(1,N_deriv):
            if (i+j < N_deriv) and (i-j >= 0):
                A[i+j,i-j] = (-1)**(j%2) * A[i,i]
                A[i-j,i+j] = A[i+j,i-j]
    return A

def calculate_B(i_frame, u, top, bottom, P, qs, gau_plus, gau_minus, popt_p, popt_m, N_deriv, cell_area):
    # Calculate the B matrix as in eq. 14 in the paper
    u.trajectory[i_frame]
    # Get the positions of the lipids
    l_p, l_m, l_pm = top.positions / 10., bottom.positions / 10., P.positions / 10.
    l_p = l_p - l_pm.mean(axis=0)
    l_m = l_m - l_pm.mean(axis=0)

    z_p, z_m = l_p[:,2], l_m[:,2]

    # Calculate the derivatives of the gaussians at the positions of the lipids
    deriv_p = calculate_rho_deriv(z_p, gau_plus, popt_p, N_deriv=N_deriv, sign=1.)
    deriv_m = calculate_rho_deriv(z_m, gau_minus, popt_m, N_deriv=N_deriv, sign=-1.)

    # Calculate the cosines of the dot products between the lipids and the q-vectors
    # cosdot = np.cos(np.einsum('ijk,mk->ijm', l_p[:,np.newaxis,:2] - l_m[np.newaxis,:,:2], qs, optimize=True))
    # B = np.einsum('ni,ijk,mj->nmk', deriv_p[1:,:], cosdot, deriv_m[1:,], optimize=True) / cell_area

    lpm = l_p[:,np.newaxis,:2] - l_m[np.newaxis,:,:2]

    B = np.zeros((N_deriv, N_deriv, qs.shape[0]))
    path = True
    for i in range(qs.shape[0]):
        cosdot = np.cos(np.einsum('ijk,k->ij', lpm, qs[i,:], optimize=False))
        if i==0:
            path = np.einsum_path('ni,ij,mj->nm', deriv_p[1:,:], cosdot, deriv_m[1:,], optimize='optimal')[0]
        B[:,:,i] = np.einsum('ni,ij,mj->nm', deriv_p[1:,:], cosdot, deriv_m[1:,], optimize=path) / cell_area

    print('Frame %d: Calculated B matrix' % i_frame)
    return B


@dask.delayed
def analyze_block(framelist, func, *args, **kwargs):
    result = []
    for i_frame in framelist:
        result.append(func(i_frame, *args, **kwargs))
    return result

if __name__ == '__main__':

    times = []
    times.append(time())
    # Create a logger
    logger = setup_logger()

    # Load the command line arguments
    args = load_command_line_args()
    log_args(args)

    u, time_moments = load_universe(args)

    indices, sampled_time_ns = sample_frames(u, args)
    N_sampled_frames = len(indices)

    # Define the lipid headgroups and leaflets
    P = u.select_atoms("name PO4")
    u_leaf=u
    u_leaf.trajectory[100]
    L = LeafletFinder(u_leaf, 'name PO4', pbc=True)
    for i, g in enumerate(L.groups()):
        logger.info("Leaflet {}: {} atoms".format(i, len(g)))
    top, bottom = L.groups()

    times.append(time())
    logger.info("Time to load and sample the trajectory: {} s".format(times[-1] - times[-2]))

    cell_props = cell_properties(u, indices, top.n_atoms)

    times.append(time())
    logger.info("Time to compute the cell properties: {} s".format(times[-1] - times[-2]))

    qs = compute_q_vectors(args, cell_props)

    # BW-DCF
    n_bins = args.bins
    logger.info("Number of bins: {}".format(n_bins))

    N_deriv = args.derivatives
    logger.info("Number of derivatives: {}".format(N_deriv))

    traj_zp, traj_zm = collect_z_positions(u, indices, top, bottom, P)

    # calculate the histogram of the z positions of the top layer
    z_p_centers, rho_p = compute_histogram(traj_zp, n_bins, cell_props, N_sampled_frames)
    z_m_centers, rho_m = compute_histogram(traj_zm, n_bins, cell_props, N_sampled_frames)

    # Do the fitting for getting the parameters of the gaussians
    # They will be used to compute the A matrix
    p0 = [0.1, 4., 1.]
    popt_p, pcov_p = curve_fit(gau_plus, z_p_centers, rho_p, p0=p0, bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]))
    popt_m, pcov_m = curve_fit(gau_minus, z_m_centers, rho_m, p0=p0, bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]))

    logger.info(f'   Top layer: rho0 = {popt_p[0]:.3f} 1/nm^2, d = {popt_p[1]:.2f} nm, alpha = {popt_p[2]:.4f} nm^2')
    logger.info(f'Bottom layer: rho0 = {popt_m[0]:.3f} 1/nm^2, d = {popt_m[1]:.2f} nm, alpha = {popt_m[2]:.4f} nm^2')

    times.append(time())
    logger.info(f'Finished preps in {times[-1] - times[-2]:.2f} seconds')

    logger.info("===========================================")

    # Calculate the A-matrix
    A_p = calculate_A_sym(popt_p[2], popt_p[0], N_deriv=N_deriv)
    A_p_inv = np.linalg.inv(A_p)

    times.append(time())
    logger.info(f'Finished A-matrix in {times[-1] - times[-2]:.2f} seconds')

    # Calculate the B-matrix

    logger.info("Calculating the B-matrix (this may take a while)...")
    n_jobs = args.cpus
    logger.info("Number of jobs: {}".format(n_jobs))

    index_blocks = np.array_split(indices, n_jobs)

    client = Client(n_workers=n_jobs)
    print(client)


    # Scatter large objects

    from pympler.asizeof import asizeof

    logger.info(f'Size of u: {asizeof(u)/2**20:.2f} MB')


    jobs = []
    for framelist in index_blocks:
        job = analyze_block(framelist, calculate_B, u, top, bottom, P, qs, gau_plus, gau_minus, popt_p, popt_m, N_deriv, cell_props['cell_area'])
        jobs.append(job)
    jobs = dask.delayed(jobs)

    result = np.concatenate(jobs.compute())

    # calculate the cummean of result by axis=0
    result_cummean = np.cumsum(result, axis=0) / np.arange(1, result.shape[0]+1)[:,np.newaxis,np.newaxis, np.newaxis]

    B = result.mean(axis=0)

    b_check = np.allclose(B, result_cummean[-1,:,:])
    logger.info(f'B-matrix check: {b_check}')

    logger.info("B-matrix computed")

    times.append(time())
    logger.info(f'Finished B-matrix in {times[-1] - times[-2]:.2f} seconds')
    logger.info(f'=====================')


    # Calculate the C-matrix
    C_B = np.einsum('i,ijk,j->k', A_p_inv[0, :], B, A_p_inv[0, :])
    C = np.einsum('i,mijk,j->km', A_p_inv[0, :], result_cummean, A_p_inv[0, :])

    logger.info("C-matrix computed")
    logger.info(f'C-matrix shape: {C.shape}')
    logger.info("rows of C-matrix correspond to the q-vectors")
    logger.info("columns of C-matrix correspond to average over time up to time t")

    b_check = np.allclose(C_B, C[:, -1])
    logger.info(f'C-matrix check passed: {b_check}')

    times.append(time())
    logger.info(f'Finished C-matrix in {times[-1] - times[-2]:.2f} seconds')

    # save q-vector and C-matrix
    data = {'q': qs, 'C': C}
    np.savez_compressed(args.output, **data)
    logger.info(f'q-vector and C-matrix saved to {args.output}')

    logger.info('Total execution time: {:.2f} seconds'.format(times[-1] - times[0]))



def end_code():
    pass