
import numpy as np
import scipy.stats as sts
import itertools as it
import general.neural_analysis as na
import general.utility as gu
import math
import scipy.special as sps
import re
import os
import pickle as p

def stitch_pkls(pattern, folder, stitch_fields=('perf','snrs'), dims=(1, 0),
                sort_by='snrs', sort_ax=0):
    fls = os.listdir(folder)
    expr = re.compile(pattern)
    match_fls = list(filter(expr.search, fls))
    out = {}
    for i, fl_name in enumerate(match_fls):
        with open(os.path.join(folder, fl_name), 'rb') as fl:
            d = p.load(fl)
        for j, field in enumerate(stitch_fields):
            if i == 0:
                out[field] = d[field]
            else:
                out[field] = np.concatenate((out[field], d[field]),
                                            axis=dims[j])
    inds = np.argsort(out[sort_by], axis=sort_ax)
    for j, field in enumerate(stitch_fields):
        swap_out = np.swapaxes(out[field], 0, dims[j])
        sorted_swap_out = swap_out[inds]
        sorted_out = np.swapaxes(sorted_swap_out, 0, dims[j])
        out[field] = sorted_out
    return out, d

def organize_types(option_list, order=None, excl=False,
                    reses=None, pure=False):
    combos = generate_combos(len(option_list), order, replace=False, excl=excl)
    types = np.array(list(it.product(*[range(x) for x in option_list])))
    types = types + reses/2
    return combos, types

def hamming_distortion(word1, word2):
    dist = np.logical_not(np.all(word1 == word2))
    return dist

def mse_distortion(word1, word2, axis=None):
    dist = np.sum((word1 - word2)**2, axis=axis)
    return dist

def volume_nball(r, n, pos=False):
    v = (np.pi**(n/2))*(r**n)/sps.gamma((n/2) + 1)
    if pos:
        v = v/(2**n)
    return v

def surface_nball(r, n, pos=False):
    g = sps.gamma((n + 1)/2)
    p = np.pi**((n + 1)/2)
    s = 2*(p*(r**n))/g
    if pos:
        s = s/(2**n)
    return s

def sample_nball(r, d, n_samps, pos=True):
    s = sts.norm.rvs(0, 1, (n_samps, d))
    if pos:
        s = np.abs(s)    
    lam = (1/r)*np.sqrt(np.sum(s**2, axis=1)).reshape((-1, 1))
    return s / lam

def integ_lattice_in_ball(r, d, eps=.000001):
    int_size = int(np.ceil(2*r + 1))
    int_size_ind = np.floor(int_size/2).astype(int)
    midpt = range(int(int_size))[int(int_size_ind/2)]
    pts = np.array(list(it.product(range(int_size), repeat=d)))
    pts_est = volume_nball(r, d)
    ds = gu.euclidean_distance(pts, (midpt,)*d)
    pts_rea = np.sum(ds <= r + eps)
    return pts_est, pts_rea

def generate_partitions(c, randsamp_thr=100000):
    if 2**c > randsamp_thr:
        r = np.random.rand(randsamp_thr, c)
        ps = r >= .5
        ps = ps.astype(int)
    else:
        ps = np.array(list(it.product([0, 1], repeat=c)))
    return ps

def estimate_dimensionality(data, randsamp_thr=100000, theta=.8):
    c, dims, samps = data.shape
    partitions = generate_partitions(c, randsamp_thr)
    print(len(partitions))
    implementable = np.zeros(len(partitions))
    for i, p in enumerate(partitions):
        c1_inds = np.where(p == 1)[0]
        c2_inds = np.where(p == 0)[0]
        if len(c1_inds) > 0 and len(c2_inds) > 0:
            c1_seq = [data[j] for j in c1_inds]
            c1 = np.concatenate(c1_seq, axis=1)
            c1 = np.reshape(c1, c1.shape + (1,))

            c2_seq = [data[j] for j in c2_inds]
            c2 = np.concatenate(c2_seq, axis=1)
            c2 = np.reshape(c2, c2.shape + (1,))
        
            out = na.svm_decoding(c1, c2, format_=False)
            implementable[i] = np.mean(out[0]) > theta
        else:
            implementable[i] = 1
    return implementable

def pure_filter(n, d, scale=1):
    f = np.zeros((n, d))
    share = int(n/d)
    for i in range(d):
        f[share*i:share*(i+1), i] = 1
    rot = 0
    return scale*f

def mixed_filter(n, d, m=0, v=1, c=0, norm=True):
    mean_vec = np.ones(d)*m
    cov_mat = np.identity(d)*v
    cov_mat[cov_mat == 0] = c
    if c == 0:
        f = np.reshape(np.random.randn(n*d)*np.sqrt(v), (n, d))
    else:
        dist = sts.multivariate_normal(mean_vec, cov_mat)
        f = dist.rvs(n)
    if norm:
        f = f - np.mean(f, axis=0).reshape((1, -1))
        f = f/np.sqrt(np.sum(f**2, axis=0).reshape((1, -1)))
    return np.sqrt(n/d)*f

def get_dimensionality(pts, transform):
    pts_inspace = transform(pts)
    orig_r = np.linalg.matrix_rank(pts.T)
    trans_r = np.linalg.matrix_rank(pts_inspace.T)
    return orig_r, trans_r

def generate_points_on_sphere(n, dim, cent=None, rad=1):
    if cent is None:
        cent = np.zeros(dim)
    cent = np.array(cent).reshape((1, dim))
    samps = sts.norm(0, 1).rvs(dim*n).reshape((n, dim))
    normalizer = np.sqrt(np.sum(samps**2, axis=1)).reshape((n, 1))
    pts = (samps/normalizer)*rad + cent
    return pts

def sample_uniform_pts(n, dim, upper, lower):
    pre_pts = np.random.rand(n, dim)
    pts = (upper - lower)*pre_pts + lower
    return pts

def generate_transform(m, order=2, pure=False, replace=True):
    """
    Generates transform into coding space for given specifications.

    Parameters 
    ----------
    m : int
        Dimensionality of the space to generate tuning within
    order : int
        Order of the tuning, orders >1 will create nonlinear selectivity.
    pure : bool
        If the tuning should be pure or mixed.

    Returns
    -------
    transform : function(X) --> Y
        Mapping from M-dimensional space to F-dimensional space for use with the
        tuning matrix.
    """
    combos = generate_combos(m, order=order, pure=pure, replace=replace)
    transform = create_transform(combos, m)
    return transform, combos

def generate_combos(m, order=2, pure=False, replace=True, excl=False):
    if excl:
        combos = []
        ord_range = (order,)
    else:
        combos = [(d,) for d in range(m)]
        ord_range = range(2, order + 1)
    for o in ord_range:
        if pure and replace:
            add = [(x,)*o for x in range(m)]
        elif replace:
            add = list(it.combinations_with_replacement(range(m), o))
        else:
            add = list(it.combinations(range(m), o))            
        combos = combos + add
    return combos

def create_transform(combos, m):
    def transform(s):
        s = np.array(s).reshape((-1, m))
        new_s = np.zeros((s.shape[0], len(combos)))
        for i, c in enumerate(combos):
            coll = np.array([s[:, j] for j in c])
            new_s[:, i] = np.product(coll, axis=0)
        return new_s
    return transform

