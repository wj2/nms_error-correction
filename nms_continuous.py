
import numpy as np
import scipy.optimize as sio
import scipy.stats as sts
import multiprocessing as mp

from mixedselectivity_theory.utility import *
import general.utility as gu
import general.rf_models as rfm

class ContinuousCode(object):

    def __init__(self, c, o, n, v, rf_size, buff=None, reses=None,
                 rf_tiling=None, excl=True, dist_samps=10000,
                 p_measure=np.median, neurs=None, filter_func=pure_filter,
                 power_metric='variance'):
        rf_sizes = np.ones(c)*rf_size
        ef_ret = construct_gaussian_encoding_function((n,)*c, rf_sizes,
                                                      o, reses=reses,
                                                      rf_tiling=rf_tiling,
                                                      return_objects=False)
        prelim_rfs, self.types, _ = ef_ret
        self.rf_func = prelim_rfs[0][0]
        self.rfs = [prf[1] for prf in prelim_rfs]
        self.order = o
        self.n = n
        self.c = c
        self.buff = buff
        self.power_metric = power_metric
        self.p_measure = p_measure
        self.dim = len(self.rfs)
        if neurs is None:
            self.neurs = self.dim
        self.f = filter_func(self.neurs, self.dim)
        p = self.evaluate_power(n_samps=dist_samps)
        self.f = self.f*np.sqrt(v/p)
        
    def evaluate_power(self, n_samps=1000):
        p = compute_power_distribution(self.response, self.c, self.n,
                                       self.buff, metric=self.power_metric,
                                       n_samps=n_samps,
                                       cent_func=self.p_measure)
        return p

    def __call__(self, stims):
        return self.response(stims)
    
    def response(self, stims):
        stim = np.array(stims)
        if len(stims.shape) == 1:
            stim = stims.reshape((1, -1))
        resp = np.zeros((stims.shape[0], self.dim))
        for i, rf in enumerate(self.rfs):
            r = self.rf_func(stims, *rf)
            resp[:, i] = r
        resp = np.dot(resp, self.f.T)
        return resp

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
                                      give_real=False, basin_hop=True,
                                      parallel=False, oo=False, n_hops=100):
    dist_overpwr = np.zeros((len(snrs), samps))
    for i, snr in enumerate(snrs):
        d = estimate_code_performance(c, o, n, snr, rf_size, buff, reses,
                                      rf_tiling, excl, dist_samps, p_measure,
                                      neurs, noise_var, filter_func,
                                      power_metric, samps, distortion,
                                      give_real=give_real, basin_hop=basin_hop,
                                      parallel=parallel, oo=oo, n_hops=n_hops)
        dist_overpwr[i] = d
    return dist_overpwr

def estimate_code_performance(c, o, n, snr, rf_size, buff=None, reses=None,
                              rf_tiling=None, excl=True, dist_samps=10000,
                              p_measure=np.median, neurs=None, noise_var=10,
                              filter_func=pure_filter, power_metric='variance',
                              samps=1000, distortion=mse_distortion,
                              give_real=False, basin_hop=True, parallel=False,
                              oo=False, n_hops=100):
    if buff is None:
        buff = rf_size
    v = (snr**2)*noise_var
    rfs, _, trs = make_code_with_power(c, o, n, v, rf_size, buff=buff,
                                       reses=reses, rf_tiling=rf_tiling,
                                       excl=excl, dist_samps=dist_samps,
                                       p_measure=p_measure, neurs=neurs,
                                       filter_func=filter_func,
                                       power_metric=power_metric,
                                       object_oriented=oo)
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
                                  real_pts=give_pts, basin_hop=basin_hop,
                                  parallel=parallel, niter=n_hops)
    dist = distortion(pts, decoded_pts, axis=1)
    return dist
    
def _decode_opt_func(args):
    npt, init_guess, basin_hop, step_size, niter, trs = args
    func = lambda x: mse_distortion(npt, trs(np.reshape(x, (1, -1))))
    if basin_hop:
        r = sio.basinhopping(func, init_guess, stepsize=step_size,
                                 niter=niter)
    else:
        r = sio.minimize(func, init_guess)
    return r.x

def decode_pop_resp(c, noisy_pts, trs, upper, lower, real_pts=None,
                    step_size=1, niter=100, basin_hop=True,
                    parallel=False):
    answers = np.zeros((noisy_pts.shape[0], c))
    mid_pt = lower + (upper - lower)/2
    arg_els = len(noisy_pts)
    if real_pts is None:
        guess = np.array((mid_pt,)*c)
        init_guesses = np.ones((arg_els, c))*guess
    else:
        init_guesses = real_pts
    step_sizes = (step_size,)*arg_els
    niters = (niter,)*arg_els
    basin_hops = (basin_hop,)*arg_els
    trses = (trs,)*arg_els
    args = zip(noisy_pts, init_guesses, basin_hops, step_sizes, niters, trses)
    if parallel:
        try:
            pool = mp.Pool(processes=mp.cpu_count())
            answers = pool.map(_decode_opt_func, args)
        finally:
            pool.close()
            pool.join()
    else:
        answers = map(_decode_opt_func, args)
    return np.array(list(answers))
    
def construct_gaussian_encoding_function(option_list, rf_sizes, order, excl=True,
                                         reses=None, rf_tiling=None,
                                         return_objects=True):
    if return_objects:
        rf_func = rfm.make_gaussian_rf
    else:
        rf_func = rfm.eval_gaussian_rf
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
                                       rf_func=rf_func,
                                       rf_sizes=rf_sizes[c],
                                       return_objects=return_objects)
        rfs = rfs + new_rfs
    if return_objects:
        transform = lambda stims: rfm.get_codewords(stims, rfs)
        assert len(types) == len(set([tuple(l) for l in transform(types)]))
    else:
        transform = None
    return rfs, types, transform    

def compute_power_distribution(trs, c, n, buff, n_samps=10000,
                               metric='variance', cent_func=np.median):
    samps = np.random.rand(n_samps, c)
    samps = (n - 2*buff)*samps + buff
    pts = trs(samps)
    power = compute_power(pts, metric=metric, cent_func=cent_func)
    return power

def compute_power(pts, metric='variance', cent_func=np.median):
    if metric.lower() == 'variance':
        power = np.sum(np.var(pts, axis=0))
    elif metric.lower() == 'distance':
        power = cent_func(np.sqrt(np.sum(pts**2, axis=1)))
    elif (metric.lower() == 'squared_distance'
          or metric.lower() == 'distance_squared'):
        power = cent_func(np.sum(pts**2, axis=1))
    else:
        raise IOError('the entered metric ({}) does not match an allowed '
                      'value'.format(metric.lower()))
    return power

def make_code_with_power(c, o, n, v, rf_size, buff=None, reses=None,
                         rf_tiling=None, excl=True, dist_samps=10000,
                         p_measure=np.median, neurs=None, object_oriented=False,
                         filter_func=pure_filter, power_metric='variance'):
    if object_oriented:
        f = _make_code_with_power_oo
    else:
        f = _make_code_with_power_func
    out = f(c, o, n, v, rf_size, buff=buff, reses=reses, rf_tiling=rf_tiling,
            excl=excl, dist_samps=dist_samps, p_measure=p_measure, neurs=neurs,
            filter_func=filter_func, power_metric=power_metric)
    return out

def _make_code_with_power_oo(c, o, n, v, rf_size, buff=None, reses=None,
                             rf_tiling=None, excl=True, dist_samps=10000,
                             p_measure=np.median, neurs=None,
                             filter_func=pure_filter,
                             power_metric='variance'):
    code = ContinuousCode(c, o, n, v, rf_size, buff=buff, reses=reses,
                          rf_tiling=rf_tiling, excl=excl, dist_samps=dist_samps,
                          p_measure=p_measure, neurs=neurs,
                          filter_func=filter_func,
                          power_metric=power_metric)
    return code.rfs, code.types, code

def _make_code_with_power_func(c, o, n, v, rf_size, buff=None, reses=None,
                               rf_tiling=None, excl=True, dist_samps=10000,
                               p_measure=np.median, neurs=None,
                               filter_func=pure_filter,
                               power_metric='variance'):
    if buff is None:
        buff = rf_size
    rf_sizes = np.ones(c)*rf_size
    rfs, ts, trs = construct_gaussian_encoding_function((n,)*c, rf_sizes,
                                                        o, rf_tiling=rf_tiling,
                                                        excl=excl, reses=reses)
    p = compute_power_distribution(trs, c, n, buff, metric=power_metric,
                                   n_samps=dist_samps, cent_func=p_measure)
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

def obtain_distrib_stretchfactor(pts, rad, trans, filt, control_sigpower=None,
                                 n_ptsper=10000, power_metric='variance',
                                 power_cent_func=np.median):
    trans_cents, _, _ = simulate_pop_resp(pts, trans, filt, noise_var=0)
    
    pwr = compute_power(trans_cents, metric=power_metric,
                        cent_func=power_cent_func)
    if control_sigpower is not None:
        filt = filt*np.sqrt(control_sigpower/pwr)
        trans_cents, _, _ = simulate_pop_resp(pts, trans, filt, noise_var=0)
        pwr = compute_power(trans_cents, metric=power_metric,
                            cent_func=power_cent_func)
    sfs = np.zeros((len(pts), n_ptsper))
    for i, pt in enumerate(pts):
        sfs[i] = obtain_stretchfactor_cent(pt, rad, trans, filt, n_pts=n_ptsper)
    return sfs, pwr
