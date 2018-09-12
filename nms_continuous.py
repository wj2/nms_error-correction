
import numpy as np
import scipy.optimize as sio
import scipy.stats as sts

from mixedselectivity_theory.utility import *
import general.utility as gu
import general.rf_models as rfm

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
    snr = np.sqrt(pwr/nv)
    if control_snr is not None:
        filt = filt*np.sqrt(control_snr/snr)
        rep_pts = np.dot(pts_t, filt.T)
        pwr = np.sum(np.var(rep_pts, axis=0))
        snr = np.sqrt(pwr/nv)
    rep_pts = rep_pts + add_noise
    return rep_pts, filt, snr

def estimate_code_performance_overpwr(c, o, n, snrs, rf_size, buff=None,
                                      reses=None, rf_tiling=None, excl=True,
                                      dist_samps=10000, p_measure=np.median,
                                      neurs=None, noise_var=10,
                                      filter_func=pure_filter,
                                      power_metric='variance', samps=1000,
                                      distortion=mse_distortion,
                                      give_real=False):
    dist_overpwr = np.zeros((len(snrs), samps))
    for i, snr in enumerate(snrs):
        d = estimate_code_performance(c, o, n, snr, rf_size, buff, reses,
                                      rf_tiling, excl, dist_samps, p_measure,
                                      neurs, noise_var, filter_func,
                                      power_metric, samps, distortion,
                                      give_real=give_real)
        dist_overpwr[i] = d
    return dist_overpwr

def estimate_code_performance(c, o, n, snr, rf_size, buff=None, reses=None,
                              rf_tiling=None, excl=True, dist_samps=10000,
                              p_measure=np.median, neurs=None, noise_var=10,
                              filter_func=pure_filter, power_metric='variance',
                              samps=1000, distortion=mse_distortion,
                              give_real=False):
    if buff is None:
        buff = rf_size
    v = (snr**2)*noise_var
    rfs, ts, trs = make_code_with_power(c, o, n, v, rf_size, buff=buff,
                                        reses=reses, rf_tiling=rf_tiling,
                                        excl=excl, dist_samps=dist_samps,
                                        p_measure=p_measure, neurs=neurs,
                                        filter_func=filter_func,
                                        power_metric=power_metric)
    pts = sample_uniform_pts(samps, c, n - buff, buff)
    dummy_filt = np.identity(len(rfs))
    noise_distrib = sts.norm(0, np.sqrt(noise_var))
    pts_rep, _, snr = simulate_pop_resp(pts, trs, dummy_filt,
                                        noise_distrib=noise_distrib)
    if give_real:
        give_pts = pts
    else:
        give_pts = None
    decoded_pts = decode_pop_resp(c, pts_rep, trs, n - buff, buff,
                                  real_pts=give_pts)
    dist = distortion(pts, decoded_pts, axis=1)
    return dist
    
def decode_pop_resp(c, noisy_pts, trs, upper, lower, real_pts=None):
    answers = np.zeros((noisy_pts.shape[0], c))
    mid_pt = lower + (upper - lower)/2
    init_guess = np.array((mid_pt,)*c)
    for i, npt in enumerate(noisy_pts):
        func = lambda x: mse_distortion(npt, trs(np.reshape(x, (1, -1))))
        if real_pts is not None:
            init_guess = real_pts[i]
        r = sio.minimize(func, init_guess)
        if r.success:
            answers[i] = r.x
        else:
            answers[i] = np.nan
    return answers
    
def construct_gaussian_encoding_function(option_list, rf_sizes, order, excl=True,
                                         reses=None, rf_tiling=None):
    if reses is None:
        reses = np.ones(len(option_list))
    if order is None:
        order = len(option_list)
    if reses is not None:
        reses = np.array(reses)
    rf_sizes = np.array(rf_sizes)
    if rf_tiling is None:
        rf_tiling = np.ones_like(reses)
    option_list = np.array(option_list)
    combos, types = organize_types(option_list, order, excl, reses)
    rfs = []
    for i, c in enumerate(combos):
        c = np.array(c)
        if reses is not None:
            sub_reses = reses[c]
        else:
            sub_reses = None
        new_rfs = rfm.construct_rf_pop(rf_tiling[c], option_list[c], 
                                        reses=sub_reses, sub_dim=c,
                                        rf_func=rfm.make_gaussian_rf,
                                        rf_sizes=rf_sizes[c])
        rfs = rfs + new_rfs
    transform = lambda stims: rfm.get_codewords(stims, rfs)
    assert len(types) == len(set([tuple(l) for l in transform(types)]))
    return rfs, types, transform    

def compute_power_distribution(trs, c, n, buff, n_samps=10000,
                               metric='variance'):
    samps = np.random.rand(n_samps, c)
    samps = (n - 2*buff)*samps + buff
    pts = trs(samps)
    if metric.lower() == 'variance':
        power = np.sum(np.var(pts, axis=0))
    elif metric.lower() == 'distance':
        power = np.sum(pts**2, axis=1)
    return power

def make_code_with_power(c, o, n, v, rf_size, buff=None, reses=None,
                            rf_tiling=None, excl=True, dist_samps=10000,
                            p_measure=np.median, neurs=None, 
                            filter_func=pure_filter, power_metric='variance'):
    if buff is None:
        buff = rf_size
    rf_sizes = np.ones(c)*rf_size
    rfs, ts, trs = construct_gaussian_encoding_function((n,)*c, rf_sizes,
                                                        o, rf_tiling=rf_tiling,
                                                        excl=excl)
    ds = compute_power_distribution(trs, c, n, buff, metric=power_metric,
                                       n_samps=dist_samps)
    p = p_measure(ds)
    if neurs is None:
        neurs = len(rfs)
    beta = filter_func(neurs, len(rfs))*np.sqrt(v/p)
    trs_beta = lambda x: np.dot(trs(x), beta.T)
    return rfs, ts, trs_beta    

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
    rep_cent, _, _ = simulate_pop_resp(cent.reshape((1, -1)), trans, filt,
                                       noise_var=0)
    stretches = gu.euclidean_distance(rep_pts, rep_cent)/rad
    return stretches

def obtain_distrib_stretchfactor(pts, rad, trans, filt,
                                 control_sigpower=None,
                                 n_ptsper=10000):
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
