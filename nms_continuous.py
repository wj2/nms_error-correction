
import numpy as np
import scipy.optimize as sio

from mixedselectivity_theory.utility import *

def rep_func(stim, trans, code_pt):
    new_pt = trans(stim)
    err = np.sum((new_pt - code_pt)**2)
    return err

def simulate_pop_resp(pts, trans, filt, noise_distrib=None, noise_var=5,
                      control_snr=None):
    pts_t = trans(pts)
    rep_pts = np.dot(pts_t, filt.T)
    pwr = np.sum(np.var(rep_pts, axis=0))
    if noise_distrib is None:
        noise_distrib = sts.norm(0, np.sqrt(noise_var))
    add_noise = noise_distrib.rvs(rep_pts.shape)
    nv = np.var(add_noise)
    snr = pwr/(nv*filt.shape[0])
    if control_snr is not None:
        filt = filt*np.sqrt(control_snr/snr)
        rep_pts = np.dot(pts_t, filt.T)
        pwr = np.sum(np.var(rep_pts, axis=0))
        snr = pwr/(nv*filt.shape[0])
    rep_pts = rep_pts + add_noise
    return rep_pts, filt, snr

def decode_stimulus(rep_pts, trans, filt, orig_dim):
    inv_filt = np.linalg.pinv(filt)
    est_pts = np.dot(inv_filt, rep_pts.T).T
    sols = np.zeros((len(rep_pts), orig_dim))
    sol_succ = np.zeros(len(rep_pts), dtype=bool)
    init_ests = np.zeros_like(sols)
    for i, pt in enumerate(est_pts):
        init_est = pt[:orig_dim]
        sol = sio.minimize(rep_func, init_est, (trans, pt))
        sols[i] = sol.x
        sol_succ[i] = sol.success
        init_ests[i] = init_est
    return sols, sol_succ, init_ests

def evaluate_performance(pts, trans, filt, noise_distrib=None, noise_var=5,
                         control_snr=None):
    rep_pts, filt, snr = simulate_pop_resp(pts, trans, filt, 
                                           noise_distrib=noise_distrib,
                                           noise_var=noise_var,
                                           control_snr=control_snr)
    dec_pts, succs, init_ests = decode_stimulus(rep_pts, trans, filt, 
                                                pts.shape[1])
    mse = np.nansum((dec_pts - pts)**2, axis=1)
    mse_init = np.nansum((init_ests - pts)**2, axis=1)
    mse[np.logical_not(succs)] = np.nan
    return mse, snr, mse_init
    
def obtain_msesnr(pts, trans, filt, noise_var=5, control_snr=None):
    mse, snr, _ = evaluate_performance(pts, trans, filt, noise_var=noise_var,
                                       control_snr=control_snr)
    return mse, snr

def obtain_stretchfactor_cent(cent, rad, trans, filt, n_pts=1000):
    pts = generate_points_on_sphere(n_pts, len(cent), cent=cent, rad=rad)
    rep_pts, _, _ = simulate_pop_resp(pts, trans, filt, noise_var=0)
    rep_cent, _, _ = simulate_pop_resp([cent], trans, filt, noise_var=0)
    stretches = np.sum(((rep_pts - rep_cent)/rad)**2, axis=1)
    return stretches

def obtain_distrib_stretchfactor(pts, rad, trans, filt, control_sigpower=None,
                                 n_ptsper=1000):
    trans_cents, _, _ = simulate_pop_resp(pts, trans, filt, noise_var=0)
    pwr = np.sum(np.var(trans_cents, axis=0))
    if control_sigpower is not None:
        filt = filt*np.sqrt(control_sigpower/pwr)
        trans_cents, _, _ = simulate_pop_resp(pts, trans, filt, noise_var=0)
        pwr = np.sum(np.var(trans_cents, axis=0))
    sfs = np.zeros((len(pts), n_ptsper))
    for i, pt in enumerate(pts):
        sfs[i] = obtain_stretchfactor_cent(pt, rad, trans, filt, n_pts=n_ptsper)
    return sfs, pwr
