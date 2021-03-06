
import itertools as it
import numpy as np
import scipy.stats as sts
import scipy.optimize as sio
import scipy.special as sm
import general.utility as u
import general.rf_models as ccec
from sklearn import svm

from mixedselectivity_theory.utility import *

def efficacy_tuning(eps_r, nh, pxh, a=10):
    nlpxl = 1 - nh*pxh
    out = nh*pxh*np.exp(-(a + eps_r)) + nlpxl*np.exp(-(a - eps_r))
    return out

def empirical_variance_power_func(samps):
    return np.sum(np.var(samps, axis=0))

def empirical_radius_power_func(samps):
    return np.mean(np.sum(samps**2, axis=1))

def l2_filt_scale(cp, p):
    return np.sqrt(cp/p)

def empirical_l1_power_func(samps):
    return np.mean(np.sqrt(np.sum(samps**2, axis=1)))

def l1_filt_scale(cp, p):
    return cp/p

def empirical_l1r_power_func(samps):
    return np.mean(np.sum(np.abs(samps), axis=1))

def l1r_filt_scale(cp, p):
    return cp/p

empirical_l1r_power = {'pwr':empirical_l1r_power_func,
                       'scale':l1r_filt_scale}
analytical_l1r_power = {'pwr':empirical_l1r_power_func,
                        'scale':l1r_filt_scale}

empirical_variance_power = {'pwr':empirical_variance_power_func,
                            'scale':l2_filt_scale}
empirical_radius_power = {'pwr':empirical_radius_power_func,
                            'scale':l2_filt_scale}
empirical_l1_power = {'pwr':empirical_l1_power_func,
                      'scale':l1_filt_scale}

def analytical_effective_dimension(k, o, n):
    k_o = sm.comb(k, o)
    no = n**o
    do = k_o*no
    num = do
    denom = 1 + (1/do)*(k_o - 1)**2
    return num/denom

def max_overlap(k, o, n):
    return n**sm.comb(k - 3, o - 1)

def compare_cov_theory(k, o, n):
    out = code_effective_dimension(k, o, n)
    do = analytical_code_terms(o, k, n, excl=True)
    no = n**o
    n2o = n**(2*o)
    num = do/n2o
    denom = 1/n2o + (do - 1)*out[2]
    theorvar = (sm.comb(sm.comb(k, o), 2)*2/((do - 1)**2))**2
    num2 = do*((1/(do))*2*sm.comb(sm.comb(k, o), 2))**2
    theor2var = num2/(do - 1)**2
    denom_t = 1/n2o + (do - 1)*theorvar
    print('theor2actual', theor2var)
    print('emp   actual', out[2])
    print('theor actual', num/denom)
    print('theor approx', analytical_effective_dimension(k, o, n))
    print(out)

def compute_cov(data, demean=True):
    if demean:
        cov_mat = np.cov(data)
    else:
        cov_mat = np.zeros((data.shape[0], data.shape[0]))
        for row in data.T:
            out_mat = np.outer(row, row)
            cov_mat = cov_mat + np.outer(row, row)
        cov_mat = cov_mat / (data.shape[1] - 1)
    return cov_mat

def code_effective_dimension(k, o, n, excl=True, rf_sizes=None,
                             return_proj=False):
    if rf_sizes is None:
        _, bt, trs, _ = generate_types((n,)*k, order=o, excl=excl)
        pop_size = analytical_code_terms(o, k, n, excl=excl)
    else:
        _, bt, trs = generate_cc_types((n,)*k, rf_sizes, order=o, excl=excl)
        pop_size = analytical_code_terms_cc(o, k, n, rf_sizes[0], excl=excl)
    resps = trs(bt)
    cov_mat = compute_cov(resps.T, demean=False)
    nd_mask = ~np.eye(cov_mat.shape[0], dtype=bool)
    nondiag = cov_mat[nd_mask]
    cij_emp = np.mean(nondiag**2)
    evs, evecs = np.linalg.eig(cov_mat)
    print(evs[evs > 1e-10], sum(evs > 1e-10))
    dim = (np.sum(evs)**2)/np.sum(evs**2)
    if return_proj:
        print(evs)
        print(evecs[:, evs > 0])
        length_evecs = evs[evs > 0]*evecs[:, evs > 0]
        print(length_evecs)
        proj_stim = np.dot(length_evecs.T, resps.T) 
        out = (dim.real, pop_size, cij_emp, proj_stim)
    else:
        out = dim.real, pop_size, cij_emp
    return out

def generate_types(option_list, order=None, pure=False, excl=False):
    types = list(it.product(*[range(x) for x in option_list]))
    binary_types = np.zeros((len(types), np.sum(option_list)))
    for i, t in enumerate(types):
        for j, e in enumerate(option_list):
            b_ind = int(np.sum(option_list[:j]) + types[i][j])
            binary_types[i, b_ind] = 1
    if order is None:
        order = len(option_list)
    combos = generate_combos(binary_types.shape[1], order,
                             pure=pure, replace=False, excl=excl)
    combos_filt = []
    for c in combos:
        cols = np.sum(np.array([binary_types[:, ci] for ci in c]), 
                      axis=0)
        if np.any(cols >= len(c)):
            combos_filt.append(c)
    transform = create_transform(combos_filt, np.sum(option_list))
    return types, binary_types, transform, combos_filt

def generate_cc_types(option_list, rf_sizes, order=None, excl=False, 
                      reses=None):
    if reses is None:
        reses = np.ones(len(option_list))
    if order is None:
        order = len(option_list)
    if reses is not None:
        reses = np.array(reses)
    rf_sizes = np.array(rf_sizes)
    option_list = np.array(option_list)
    combos, types = organize_types(option_list, order, excl, reses)
    rfs = []
    for i, c in enumerate(combos):
        c = np.array(c)
        if reses is not None:
            sub_reses = reses[c]
        else:
            sub_reses = None
        new_rfs = ccec.construct_rf_pop(rf_sizes[c], option_list[c], 
                                        reses=sub_reses, sub_dim=c)
        rfs = rfs + new_rfs
    transform = lambda stims: ccec.get_codewords(stims, rfs)
    assert len(types) == len(set([tuple(l) for l in transform(types)]))
    return rfs, types, transform

def compare_mindist(c, o, n, excl=True, eps=.0001):
    active = int(np.round(code_radius(c, o, excl=excl)**2))    
    code_md = analytical_code_distance(o, c, excl=excl)
    mo = int(np.round(active - (code_md**2)/2))
    p_as = random_determined_mindist(c, o, n, mo, excl=excl, eps=eps)
    return p_as

def get_md_var_dim_cn_allorders(c, n):
    mds = np.zeros(c)
    vs = np.zeros_like(mds)
    terms = np.zeros_like(mds)
    orders = range(1, c + 1)
    for i, o in enumerate(orders):
        mds[i], vs[i], terms[i] = get_md_var_dim_cn(c, n, o)
    return mds, vs, terms, orders

def get_md_var_dim_cn(c, n, order=None):
    return get_md_var_dim((n,)*c, order=order)

def get_md_var_dim(option_list, order=None):
    if order is None:
        order = len(option_list)
    ts, bts, trs, cf = generate_types(option_list, order=order)
    trs_types = trs(bts)
    md = np.min(find_distances(trs_types))
    var = np.sum(np.var(trs_types, axis=0))
    dim = trs_types.shape[1]
    return md, var, dim

def find_distances(trans_types):
    ind_pairs = list(it.combinations(range(len(trans_types)), 2))
    dists = np.zeros(len(ind_pairs))
    for i, p in enumerate(ind_pairs):
        t1 = trans_types[p[0]]
        t2 = trans_types[p[1]]
        dists[i] = u.euclidean_distance(t1, t2)
    return dists

def decode_word(word, types):
    dists = u.euclidean_distance(word, types)
    ind = np.argmin(dists)
    return types[ind], ind

def normalize_snr(samps, targ_snr, filt, nv, nonoise=False):
    filt_samps = np.dot(samps, filt.T)
    pwr = np.sum(np.var(filt_samps, axis=0))
    snr = pwr/(nv*filt.shape[0])
    if nonoise:
        targ_snr = targ_snr/(nv*filt.shape[0])
    filt = filt*np.sqrt(targ_snr/snr)
    return filt

def simulate_resp(samps, filt, noise_var):
    filt_samps = np.dot(samps, filt.T)
    nstd = np.sqrt(noise_var)
    filt_noisy_samps = filt_samps + sts.norm(0, nstd).rvs(filt_samps.shape)
    return filt_noisy_samps

def get_filt_mindist(types, trans, neurs, pwr, with_filt=mixed_filter):
    types_trans = trans(types)
    filt = with_filt(neurs, types_trans.shape[1])
    filt = normalize_snr(types_trans, pwr, filt, 1, nonoise=True)
    filt_types = np.dot(types_trans, filt.T)
    ds = find_distances(filt_types)
    return np.min(ds)

def gaussian_noise(samps, noise_pwr):
    noisy_samps = samps + sts.norm(0, noise_pwr).rvs(samps.shape)
    return noisy_samps

def poisson_noise(samps, noise_pwr, baseline=None):
    if baseline is not None:
        samps[samps < baseline] = baseline
    noisy_samps = sts.poisson.rvs(samps)
    return noisy_samps

def estimate_real_perc_correct(c, o, n_i, noisevar, v, n_samps=1000,
                               excl=False, cc_rf=None, subdim=False,
                               distortion_func=hamming_distortion, 
                               eps=1, bs=True, input_noise=0,
                               pwr_func=empirical_variance_power,
                               noise_func=gaussian_noise,
                               local_input_noise=True):
    if cc_rf is None:
        ts, bt, trs, _ = generate_types((n_i,)*c, order=o, excl=excl)
    else:
        _, ts, trs = generate_cc_types((n_i,)*c, cc_rf, order=o, excl=excl)
        bt = ts
    out = simulate_transform_code_full(bt, trs, noisevar, v, 
                                       neurs=trs(bt).shape[1], 
                                       n_samps=n_samps, subdim=subdim,
                                       distortion_func=distortion_func,
                                       eps=eps, pwr_func=pwr_func,
                                       noise_func=noise_func,
                                       num_types=ts, input_noise=input_noise,
                                       local_input_noise=local_input_noise)
    _, _, _, corr = out
    if bs:
        corr = u.bootstrap_list(corr, np.mean, n=n_samps)
    return corr

def simulate_transform_code_out(c, o, n_i, noisevar, v, neurs=None, 
                                times_samp=10, excl=False, cc_rf=None, 
                                subdim=False, eps=1,
                                distortion_func=hamming_distortion):
    if cc_rf is None:
        words, bt, trs, sel = generate_types((n_i,)*c, order=o, excl=excl)
        words = np.array(words)
    else:
        _, bt, trs = generate_cc_types((n_i,)*c, cc_rf, order=o, excl=excl)    
        words = bt
    n_samps = len(bt)*times_samp
    if neurs is None:
        neurs = trs(bt).shape[1]
    out = simulate_transform_code_full(bt, trs, noisevar, v, 
                                       neurs=neurs, 
                                       n_samps=n_samps, subdim=subdim,
                                       distortion_func=distortion_func,
                                       eps=eps)
    out = out + (bt, words, trs)
    if cc_rf is None:
        out = out + (sel,)
    return out    
    
def estimate_code_dimensionality(corr, oword, ns, bt, randsamp_thr=100000, 
                                 theta=.8):
    corr = np.logical_not(corr.astype(bool))
    ns_corr_format = _format_by_original_word(bt, oword[corr], ns[corr])
    impl_corr = estimate_dimensionality(ns_corr_format, randsamp_thr, theta)
    
    ns_incorr_format = _format_by_original_word(bt, oword[np.logical_not(corr)],
                                                ns[np.logical_not(corr)])
    impl_incorr = estimate_dimensionality(ns_incorr_format, 
                                          randsamp_thr, theta)
    corr_dim = np.log2(np.sum(impl_corr))
    incorr_dim = np.log2(np.sum(impl_incorr))
    return corr_dim, incorr_dim

def get_original_from_binary(bt, words, owords):
    nbwords = np.zeros((len(owords), words.shape[1]))
    for i, ow in enumerate(owords):
        nbwords[i] = words[np.argmin(np.sum(np.abs(ow - bt), axis=1))]
    return nbwords

def _format_by_original_word(all_words, orig_words, noise_words):
    collection = []
    sizes = np.zeros(len(all_words))
    for i, word in enumerate(all_words):
        mask = np.array([(word == row).all() for row in orig_words])
        collection.append(noise_words[mask].T)
        sizes[i] = np.sum(mask)
    minsize =int(min(sizes))
    c = np.array(list(map(lambda x: x[:, :minsize], collection)))
    return c

def decode_single_attribute(all_words, enc_words, orig_words, noise_words, 
                            resample=100, c=1000):
    accuracy_dims = []
    for i in range(all_words.shape[1]):
        feat_vals = np.unique(all_words[:, i])
        corr_matrix = np.zeros((len(feat_vals), resample))
        for j, v in enumerate(feat_vals):
            c1_mask = orig_words[:, i] == v
            c1_words = enc_words[all_words[:, i] == v]
            c1 = noise_words[c1_mask].T
            c1 = np.reshape(c1, c1.shape + (1,))

            c2_words = enc_words[np.logical_not(all_words[:, i] == v)]
            c2 = noise_words[np.logical_not(c1_mask)].T
            c2 = np.reshape(c2, c2.shape + (1,))

            # find ideal bound, then decide which noise words are on 
            # either side
            x = np.concatenate((c1_words, c2_words), axis=0)
            y = np.concatenate((np.ones(c1_words.shape[0]), 
                                -np.ones(c2_words.shape[0])))
            clf = svm.SVC(kernel='linear', C=c)
            clf.fit(x, y)
            c1_class = clf.predict(c1[:, :, 0].T) == 1
            c2_class = clf.predict(c2[:, :, 0].T) == -1
            corr = np.concatenate((c1_class, c2_class), axis=0)
            corr_matrix[j] = u.bootstrap_list(corr, np.mean, resample)
        accuracy_dims.append(corr_matrix)
    return accuracy_dims

def estimate_real_perc_correct_overpwr(c, o, n_i, noisevar, var, n_samps=1000,
                                       excl=False, cc_rf=None, subdim=False,
                                       distortion_func=hamming_distortion,
                                       eps=1, bs=True, pwr_func=None,
                                       noise_func=gaussian_noise,
                                       input_noise=0):
    vs = np.zeros((len(var), n_samps))
    for i, v in enumerate(var):
        vs[i] = estimate_real_perc_correct(c, o, n_i, noisevar, v, 
                                           n_samps=n_samps, excl=excl, 
                                           cc_rf=cc_rf, subdim=subdim, eps=eps,
                                           distortion_func=distortion_func, 
                                           bs=bs, pwr_func=pwr_func,
                                           noise_func=noise_func,
                                           input_noise=input_noise)
    return vs

def simulate_transform_code_full(types, trans, noise_var, code_pwr, neurs,
                                 with_filt=pure_filter, n_samps=1000, 
                                 pwr_func=empirical_variance_power,
                                 samp_inds=None, subdim=False, eps=1,
                                 distortion_func=hamming_distortion,
                                 noise_func=gaussian_noise, input_noise=0,
                                 num_types=None, local_input_noise=True):
    if pwr_func is None:
        pwr_func = empirical_variance_power
    pwr_comp, filt_scale = pwr_func['pwr'], pwr_func['scale']
    if samp_inds is None:
        samp_inds = np.random.randint(0, len(types), size=n_samps)
    types_trans = trans(types)
    if subdim:
        code_pwr = code_pwr - types_trans.shape[1]
    code_pwr = code_pwr/eps
    nofilt_pwr = np.sum(np.var(types_trans, axis=0))
    filt = with_filt(neurs, types_trans.shape[1])
    filt_all = np.dot(types_trans, filt.T)
    samps = types_trans[samp_inds]
    if input_noise > 0 and local_input_noise:
        nt_transform_dict = dict(zip([tuple(nt) for nt in num_types],
                                     types_trans))
        num_types = np.array(num_types)
        input_samps = num_types[samp_inds]
        noise_arr = np.random.rand(*input_samps.shape) < input_noise
        flip_side = (np.random.rand(*input_samps.shape) < .5)*-1
        num_vals = int(np.round(num_types.shape[0]**(1/num_types.shape[1]), 0))
        noise_samps = (input_samps + noise_arr*flip_side) % num_vals 
        samps_in = np.array(list([nt_transform_dict[tuple(ns)]
                                  for ns in noise_samps]))
    elif input_noise > 0 and not local_input_noise:
        orig_samps = types[samp_inds]
        types = np.array(num_types)
        inp_noise = np.random.rand(*orig_samps.shape) < input_noise
        noise_samps = (orig_samps + inp_noise) % 2
        samps_in = trans(noise_samps)
    else:
        samps_in = samps
    pwr = pwr_comp(filt_all)
    # print('pre', pwr)
    filt = filt*filt_scale(code_pwr, pwr)
    # filt = filt*np.sqrt(code_pwr/pwr)
    inv_filt = np.linalg.pinv(filt)
    filt_samps = np.dot(samps_in, filt.T)
    m = types_trans.shape[1]
    filt_all = np.dot(types_trans, filt.T)
    pwr2 = pwr_comp(filt_samps)
    # print('post', pwr2)
    # print('targ', code_pwr)
    snr = np.sqrt(pwr2/noise_var)
    nstd = np.sqrt(noise_var)
    filt_noisy_samps = noise_func(filt_samps, nstd)
    noisy_samps = np.dot(inv_filt, filt_noisy_samps.T).T
    noise_dists = np.sqrt(np.sum((noisy_samps - samps)**2, axis=1))
    dec_words = np.zeros((n_samps, types.shape[1]))
    corr = np.zeros(n_samps)
    for i, ns in enumerate(noisy_samps):
        _, ind = decode_word(ns, types_trans)
        orig_word = types[ind]
        true_orig_word = types[samp_inds[i]]
        corr[i] = distortion_func(true_orig_word, orig_word)
        dec_words[i] = orig_word
    return types[samp_inds], dec_words, noisy_samps, corr

def construct_codeword_to_int_dict(c, o, n_i):
    _, bt, trs, _ = generate_types((n_i,)*c, order=o)
    tr_bt = trs(bt)
    pairs_trs = [(tuple(row), i) for i, row in enumerate(tr_bt)]
    pairs_untrs = [(tuple(row), i) for i, row in enumerate(bt)]
    d_trs = {}
    d_trs.update(pairs_trs)
    d_untrs = {}
    d_untrs.update(pairs_untrs)
    return d_trs, d_untrs

def estimate_mi_transform_code(c, o, n_i, nv, code_pwr, 
                               with_filt=pure_filter, samps_per_alt=10,
                               ests=5, pwr_func=empirical_variance_power,
                               neurs=None, excl=False):
    _, bt, trs, _ = generate_types((n_i,)*c, order=o, excl=excl)
    if neurs is None:
        neurs = trs(bt).shape[1]        
    dic_t, dic_ut = construct_codeword_to_int_dict(c, o, n_i)
    num = bt.shape[0]
    samps = num*samps_per_alt
    est_inds = np.random.randint(0, num, size=ests)
    ent = np.zeros(ests)
    corr = np.zeros_like(ent)
    for i, e in enumerate(est_inds):
        samp_inds = np.ones(samps, dtype=int)*e
        _, wds, _, _ = simulate_transform_code_full(bt, trs, nv, code_pwr, neurs,
                                                    with_filt, samps, pwr_func, 
                                                    samp_inds)
        int_wds = [dic_ut[tuple(row)] for row in wds]
        bd, _ = np.histogram(int_wds, bins=np.arange(0, num + 1, 1))
        norm_bd = bd/samps
        norm_bd_nz = norm_bd[norm_bd > 0]
        ent[i] = -np.sum(norm_bd_nz*np.log2(norm_bd_nz))
        corr[i] = norm_bd[e]
    mi = np.log2(num) - ent
    return mi, corr

def simulate_transform_code(types, trans, noise_var=1, n_samps=1000, 
                            control_snr=None, with_filt=None, neurs=100):
    samp_inds = np.random.randint(0, len(types), size=n_samps)
    types_trans = trans(types)
    if with_filt is not None:
        filt = with_filt(neurs, types_trans.shape[1])
    else:
        filt = np.identity(types_trans.shape[1])
    samps = types_trans[samp_inds]
    if control_snr is not None:
        filt = normalize_snr(samps, control_snr, filt, noise_var)
    inv_filt = np.linalg.pinv(filt)
    filt_samps = np.dot(samps, filt.T)
    pwr = np.sum(np.var(filt_samps, axis=0))
    snr = pwr/(noise_var*filt.shape[0])
    nstd = np.sqrt(noise_var)
    filt_noisy_samps = filt_samps + sts.norm(0, nstd).rvs(filt_samps.shape)
    noisy_samps = np.dot(inv_filt, filt_noisy_samps.T).T
    dec_words = np.zeros((n_samps, types.shape[1]))
    corr = np.zeros(n_samps)
    for i, ns in enumerate(noisy_samps):
        _, ind = decode_word(ns, types_trans)
        dec_words[i] = types[ind]
        corr[i] = samp_inds[i] == ind
    return dec_words, corr

def simulate_trans_code_wrapper(*args, boot_times=1000):
    _, _, corr = simulate_transform_code(*args)
    corr = u.bootstrap_list(corr, np.mean, n=boot_times)
    return corr

def simulate_trans_code_full_wrapper(*args, boot_times=1000):
    _, _, _, corr = simulate_transform_code_full(*args)
    corr = u.bootstrap_list(corr, np.mean, n=boot_times)
    return corr
    
def analytical_code_variance(o, c, n, rf=1, excl=False):
    assert(o <= c)
    if excl:
        p = sm.comb(c, o)*(1 - (1/n)**o)*rf
    else:
        p = np.sum([sm.comb(c, o_i)*(1 - (1/n)**o_i) 
                    for o_i in range(1, o + 1)])*rf
    return p

def analytical_code_variance_l1(o, c, n, rf=1, excl=False):
    pwr = analytical_code_variance(o, c, n, rf=rf, excl=excl)
    l1_pwr = np.sqrt(pwr)
    return l1_pwr

def analytical_code_l1r(o, c, n_i, rf=1, excl=False):
    pwr = code_radius(c, o, rf=rf, excl=excl)**2
    return pwr

def analytical_code_variance_cc(o, c, n, sig_rf, excl=False):
    assert(o <= c)
    if excl:
        p = sm.comb(c, o)*(1 - (1/n)**o)*sig_rf
    else:
        p = np.sum([sm.comb(c, o_i)*(1 - (1/n)**o_i)*sig_rf 
                    for o_i in range(1, o + 1)])
    return p

def analytical_code_distance(o, c, excl=False):
    assert(o <= c)
    if excl:
        delt2 = 2*sm.comb(c - 1, o - 1)
    else:
        delt2 = np.sum([2*sm.comb(c - 1, o_i - 1) for o_i in range(1, o + 1)])
    return np.sqrt(delt2)

def analytical_code_terms(o, c, n, excl=False):
    assert(o <= c)
    if excl:
        terms = sm.comb(c, o)*(n**o)
    else:
        terms = np.sum([sm.comb(c, o_i)*(n**o_i) for o_i in range(1, o + 1)])
    return terms

def analytical_code_terms_cc(o, c, n, sig_rf, excl=False):
    assert(o <= c)
    if o == 1 or sig_rf == 1:
        terms = analytical_code_terms(o, c, n, excl=excl)
    else:
        if excl:
            terms = sm.comb(c, o)*sig_rf*((n/sig_rf) + 1)**o
        else:
            terms = np.sum([sm.comb(c, o_i)*sig_rf*((n/sig_rf) + 1)**(o_i)
                            for o_i in range(1, o + 1)])
    return terms

def analytical_var_dist(o, c, n, mults=1, excl=False):
    p = analytical_code_variance(o, c, n, excl=excl)
    delt = analytical_code_distance(o, c, excl=excl)
    terms = analytical_code_terms(o, c, n, excl=excl)
    ps = p*mults
    delts = delt*mults
    terms = terms*mults
    return ps, delts, terms

def analytical_ratio(o1, o2, c, n, v, excl=False):
    o1_delt = (np.sqrt(v/analytical_code_variance(o1, c, n, excl=excl))
               *analytical_code_distance(o1, c, excl=excl))
    o2_delt = (np.sqrt(v/analytical_code_variance(o2, c, n, excl=excl))
               *analytical_code_distance(o2, c, excl=excl))
    return o1_delt/o2_delt

def analytical_error_rhs(c, o, v, noisevar, n_i=5, excl=False):
    delt = analytical_code_distance(c, o, excl=excl)
    p = analytical_code_variance(o, c, n_i, excl=excl)
    rhs = (v*(delt**2))/(4*p*np.sqrt(noisevar))
    return rhs

def analytical_error_probability(c, o, v, noisevar, n_i=5, excl=False):
    m = analytical_code_terms(o, c, n_i, excl=excl)
    rhs = analytical_error_rhs(c, o, v, noisevar, n_i=5, excl=excl)
    p_e = 1 - sts.chi2(m).cdf(rhs)
    return p_e

def _close_words(c, o, n_i=5, excl=False, rf=1, eps=.0001):
    if excl:
        if rf == 1:
            if o < c:
                cw = c*(n_i - 1)
            else: 
                cw = (n_i**c) - 1
        else:
            if o < c:
                cw = 2*c
            else:
                cw = 2*((2**o) - 1)
    else:
        cw = c*(n_i - 1)
    return cw

def _ident_close_words(c, o, n_i, rf, cent=True):
    _, s, trs = generate_cc_types((n_i,)*c, (rf,)*c, order=o, excl=True)
    ind = np.argmin(np.sum(np.abs(s - (n_i/2,)*c), axis=1))
    all_cw = trs(s)
    ds_cw = u.euclidean_distance(all_cw, all_cw[ind])
    ds_nextmin = ds_cw[ds_cw > 0]
    md = np.min(ds_nextmin)
    md_mask = ds_cw == md
    return s[md_mask], s[ind]    

def analytical_error_probability_nnup(c, o, v, noisevar, n_i=5, excl=False,
                                      bound_at_one=True, correction=False,
                                      pwr_func=analytical_code_variance,
                                      full_series=False,
                                      scale_func=l2_filt_scale):
    if full_series:
        p_e = 0
        for delt in range(1, c + 1):
            n_e = num_neighbors_difference(c, n_i, delt)
            arg = _nnup_density_arg(c, o, v, noisevar, n_i, excl=excl,
                                    pwr_func=pwr_func, dist_diff=delt,
                                    scale_func=scale_func)
            dense_val = sts.norm(0, 1).cdf(arg)
            p_e_i = n_e*(1 - dense_val)
            p_e = p_e + p_e_i
    else:
        n_e = _close_words(c, o, n_i=n_i, excl=excl)
        arg = _nnup_density_arg(c, o, v, noisevar, n_i, excl=excl,
                                pwr_func=pwr_func, scale_func=scale_func)
        dense_val = sts.norm(0, 1).cdf(arg)
        p_e = n_e*(1 - dense_val)
        if correction: 
            n_other = n_i**c - n_e - 1
            arg_other = _nnup_density_arg(c, o, v, noisevar, n_i, excl=excl,
                                          pwr_func=pwr_func, dist_diff=c,
                                          scale_func=scale_func)
            dv_other = sts.norm(0, 1).cdf(arg_other)
            pe_other = n_other*(1 - dv_other)
            p_e = p_e + pe_other
    if bound_at_one:
        p_e = min(1, p_e)
    return p_e

def ratio_l1(c, o, excl=True):
    n_i = None
    cd = analytical_code_distance(o, c, excl=excl)
    cp = analytical_code_l1r(o, c, n_i, excl=excl)
    return cd/cp

def sparsity_l1(c, o, n_i=5, excl=True, rf=1):
    cp = analytical_code_l1r(o, c, n_i, rf=rf, excl=excl)
    pop = analytical_code_terms(o, c, n_i, excl=excl)
    return cp/pop

def _nnup_density_arg(c, o, v, noisevar, n_i=5, rf=1, excl=False,
                      subdim=False, eps=10, pwr_func=analytical_code_variance,
                      dist_diff=1, scale_func=l2_filt_scale):
    pwr = pwr_func(o, c, n_i, rf=rf, excl=excl)
    if subdim:
        ds = analytical_code_terms(o, c, n_i, rf, excl=excl)
        v = (v - ds)/eps
    trans_term = scale_func(v, pwr)
    if dist_diff == 1:
        delt = analytical_code_distance(o, c, excl=excl)
    else:
        delt = distance_given_difference(c, o, dist_diff)
    nv = 2*np.sqrt(noisevar)
    arg = trans_term*(delt/nv)
    return arg

def maximize_distance(c, v, noisevar, n_i, excl=False, subdim=True, eps=10,
                      rf=None, rf_lim=None):
    if rf_lim is None:
        rf_lim = n_i
    orders = np.arange(1, c + 1, 1)
    if rf is not None:
        rf = np.arange(1, rf_lim + 1, 1)
    else:
        rf = (rf,)
    all_combs = it.combinations(orders, rf)
    vals = np.zeros(len(all_combs))
    for i, comb in enumerate(all_combs):
        o, r = comb
        argv = _nnup_density_arg(c, o, v, noisevar, n_i=n_i, rf=r, excl=excl,
                                 subdim=subdim, eps=eps)
        vals[i] = argv
    maxpar = all_combs[np.argmax(vals)]
    return maxpar, vals, argv

def optimal_o(n, e):
    num = sm.lambertw(np.exp(1)*e) - 1
    denom = np.log(n)
    return num/denom
        
def analytical_correct_probability_nnup(c, o, v, noisevar, n_i=5, excl=False):
    return 1 - analytical_error_probability_nnup(c, o, v, noisevar, n_i=n_i,
                                                 excl=excl)

def analytical_correct_probability_nnup_wrapper(c, o, v, noisevar, n_i=5, 
                                                neurs=None, excl=False):
    return analytical_correct_probability_nnup(c, o, v, noisevar, n_i, excl=excl)

def probe_orders_analytical(c, n, mults, excl=False):
    os = range(1, c + 1)
    ps = np.zeros((len(os), len(mults)))
    delts = np.zeros_like(ps)
    terms = np.zeros_like(ps)
    for i, o in enumerate(os):
        ps[i], delts[i], terms[i] = analytical_var_dist(o, c, n, mults, 
                                                        excl=excl)
    return ps, delts, terms, os 

def code_radius(c, o, rf=1, l=1, excl=False):
    if excl:
        rad = l*np.sqrt(sm.comb(c, o)*rf)
    else:
        rad = l*np.sqrt(np.sum([sm.comb(c, o_i)*rf for o_i in range(1, o + 1)]))
    return rad

def shannon_power(c, o, n_i=5, l=1, block_ratio=1):
    rad = code_radius(c, o, l=l*np.sqrt(block_ratio))**2
    blocklen = shannon_blocklen(c, o, n_i)*block_ratio
    return rad/blocklen

def shannon_blocklen(c, o, n_i=5):
    return analytical_code_terms(o, c, n_i)

def number_codewords(c, n_i=5):
    return n_i**c

def shannon_noise(c, o, v, noisevar, block_ratio=1):
    rad = code_radius(c, o, l=np.sqrt(block_ratio))**2
    return rad*noisevar/v
    
def shannon_rate(c, o, n_i=5):
    blocklen = shannon_blocklen(c, o, n_i)
    codewords = number_codewords(c, n_i)
    return (1/blocklen)*np.log2(codewords)

def shannon_capacity_per_use_trans_constblock(c, o, v, noisevar, block, n_i=5):
    curr_block = shannon_blocklen(c, o, n_i)
    block_ratio = block/curr_block
    sp = code_radius(c, o, l=np.sqrt(block_ratio))**2
    sp_norm = sp/block
    nv = sp*noisevar/v
    cap = .5*np.log2((sp_norm/nv) + 1)
    return cap

def shannon_capacity_block_trans_constblock(c, o, v, noisevar, block, n_i=5):
    return block*shannon_capacity_per_use_trans_constblock(c, o, v, noisevar, 
                                                           block, n_i)

def shannon_capacity_per_use_trans(c, o, v, noisevar, n_i=5):
    sp = shannon_power(c, o, n_i)
    noisev = shannon_noise(c, o, v, noisevar)
    return .5*np.log2((sp/noisev) + 1)

def shannon_capacity_block_trans(c, o, v, noisevar, n_i=5):
    m = shannon_blocklen(c, o, n_i)
    return m*shannon_capacity_per_use_trans(c, o, v, noisevar, n_i)

def shannon_capacity_per_use(c, o, noisevar, n_i=5, blocklen=None):
    orig_blocklen = shannon_blocklen(c, o, n_i)
    if blocklen is not None:
        block_ratio = blocklen/orig_blocklen
    else:
        block_ratio = 1
        blocklen = orig_blocklen
    all_p = code_radius(c, o, l=np.sqrt(block_ratio))**2
    sp = all_p/blocklen
    return .5*np.log2((sp/noisevar) + 1)

def shannon_capacity_block(c, o, noisevar, n_i):
    m = shannon_blocklen(c, o, n_i)
    return m*shannon_capacity_per_use(c, o, noisevar, n_i)

def shannon_snr(c, o, n_i, noisevar):
    sp = shannon_power(c, o, n_i)
    return np.sqrt(sp/noisevar)

def radius_binary_types(bt):
    return np.mean(np.sqrt(np.sum(bt, axis=1)))

def capacity_block_trans_overpwr(c, o, vs, noisevar, n_i=5, block=None):
    if block is None:
        caps = [shannon_capacity_block_trans(c, o, v_i, noisevar, n_i) 
                for v_i in vs]
    else:
        caps = [shannon_capacity_block_trans_constblock(c, o, v_i, noisevar, 
                                                        block, n_i)
                for v_i in vs]
    return np.array(caps)

def error_upper_bound_overpwr(c, o, vs, noisevar, n_i=5, excl=False,
                              pwr_func=analytical_code_variance,
                              correction=False, full_series=False,
                              scale_func=l2_filt_scale):
    eps = [analytical_error_probability_nnup(c, o, v_i, noisevar, n_i,
                                             excl=excl, pwr_func=pwr_func,
                                             correction=correction,
                                             full_series=full_series,
                                             scale_func=scale_func)
           for v_i in vs]
    return np.array(eps)

def pwr_require_err(target_pe, c, o, n_i, eps=1, rf=1, subdim=False,
                    noisevar=1, excl=True, distortion='hamming',
                    lg=10**(5), pwr_func=analytical_code_variance,
                    correction=False, full_series=False,
                    scale_func=l2_filt_scale):
    def func(v):
        err = perr_per_energy(c, o, n_i, v, eps, rf, noisevar,
                              distortion=distortion, excl=excl,
                              subdim=subdim, bound_at_one=False,
                              pwr_func=pwr_func, correction=correction,
                              full_series=full_series,
                              scale_func=scale_func)
        return np.abs(err - target_pe)
    if subdim:
        e_bound = analytical_code_terms_cc(o, c, n_i, rf, excl=excl)
    else:
        e_bound = 0
    opt_res = sio.minimize(func, e_bound + 1, bounds=((e_bound, None),))
    snr = np.sqrt(opt_res.x/noisevar)
    return snr, opt_res.success

def agwn_pe_opt(c, o, v, noisevar, n_i=5, blocklen=None):
    """ 
    valid for large n and rate close to capacity
    -- need to include control for blocklen
    """
    orig_blocklen = shannon_blocklen(c, o, n_i=5)
    if blocklen is not None:
        block_ratio = blocklen/orig_blocklen
    else:
        blocklen = orig_blocklen
        block_ratio = 1
    rate = shannon_rate(c, o, n_i)
    rate = (1/blocklen)*np.log2(n_i**c)
    pwr = shannon_power(c, o, n_i, block_ratio=block_ratio)
    nv = pwr*noisevar/v
    cap = .5*np.log2((pwr/nv) + 1)
    popt_arg1 = np.sqrt(blocklen)*(rate - cap)
    popt_arg2 = np.sqrt(2*pwr*(pwr + nv)/(nv*(pwr + 2*nv)))
    popt_arg = popt_arg1*popt_arg2
    popt = sts.norm(0, 1).cdf(popt_arg)
    return popt

def agwn_low_rate_upper_bound(c, o, v, noisevar, n_i=5, blocklen=None):
    orig_blocklen = shannon_blocklen(c, o, n_i)
    if blocklen is not None:
        block_ratio = blocklen/orig_blocklen
    else:
        blocklen = orig_blocklen
        block_ratio = 1
    pwr = shannon_power(c, o, n_i, block_ratio=block_ratio)
    nv = shannon_noise(c, o, v, noisevar, block_ratio)
    snr = np.sqrt(pwr/nv)
    coeff = 1/(snr*np.sqrt(np.pi*blocklen))
    rate = (1/blocklen)*np.log2(n_i**c)
    lam = shannon_lambda(c, n_i, blocklen)
    ex = np.exp(blocklen*(rate - (lam**2)*(snr**2))/4)
    # this division by four is at an ambiguous position in the equation in
    # the paper...
    return coeff*ex

def shannon_lambda(c, n_i, blocklen):
    r = (1/blocklen)*np.log2(n_i**c)
    expon = 2**(r/(1 - (1/blocklen)))
    mult1 = np.sqrt(2)/np.sin(2)
    lam = np.sin(mult1*expon)
    return lam 
    
def distance_given_difference(c, o, v):
    x = 0
    for i in range(1, v + 1):
        y = sm.comb(v, i)*sm.comb(c - v, o - i)
        x = x + y
    return np.sqrt(2*x)

def num_neighbors_difference(c, n, v):
    m = sm.comb(c, v)*(n - 1)**v
    return m

def distance_combinations(c, o, k):
    x = 0
    for i in range(1, k + 1):
        y = sm.comb(k, i)*sm.comb(c - k, o - i)
        x = x + y
    print('theor2', x)
    combs = np.array(list(it.combinations(range(c), o)))
    el = np.zeros(len(combs))
    for i in range(k):
        el = el + np.sum((combs == i), axis=1)
    print('actual', np.sum(el > 0))
    
def power_modulation_beta(n_mult, c, o, n_i):
    _, bt, trs, _ = generate_types((n_i,)*c, order=o)
    s = bt.shape[1]
    mf = mixed_filter(s*n_mult, s)
    pf = pure_filter(s*n_mult, s)
    mf_inv = np.linalg.pinv(mf)
    pf_inv = np.linalg.pinv(pf)
    print('---- orig ----')
    print('row len m', np.sqrt(np.sum(mf**2, axis=1)))
    print('col len m', np.sqrt(np.sum(mf**2, axis=0)))
    print('m', np.sqrt(np.sum(mf**2)))
    print('row len p ', np.sqrt(np.sum(pf**2, axis=1)))
    print('col len p', np.sqrt(np.sum(pf**2, axis=0)))
    print('p', np.sqrt(np.sum(pf**2)))
    print('---- inv ----')
    print('row len m', np.sqrt(np.sum(mf_inv**2, axis=1)))
    print('col len m', np.sqrt(np.sum(mf_inv**2, axis=0)))
    print('m', np.sqrt(np.sum(mf_inv**2)))
    print('row len p ', np.sqrt(np.sum(pf_inv**2, axis=1)))
    print('col len p', np.sqrt(np.sum(pf_inv**2, axis=0)))
    print('p', np.sqrt(np.sum(pf_inv**2)))

def perr_per_energy(c, o, n_i, e, eps, sig, nv=10, excl=True, subdim=True,
                    distortion='mse', bound_at_one=True,
                    pwr_func=analytical_code_variance, correction=False,
                    full_series=False, scale_func=l2_filt_scale):
    if subdim:
        dims = analytical_code_terms_cc(o, c, n_i, sig, excl=excl)
        v = (1/eps)*(e - dims)
    else:
        v = e/eps
    if v >= 0:
        est_err = hetero_error_full_ana_nnub(c, o, v, nv, n_i, sig,
                                             bound_at_one=bound_at_one,
                                             distortion=distortion,
                                             pwr_func=pwr_func,
                                             full_series=full_series,
                                             correction=correction,
                                             scale_func=scale_func)   
    else:
        est_err = np.inf
    return est_err

def get_sig_opt(c, o, n_i, e):
    if o > 1:
        mt = e/(o*n_i*sm.comb(c, o))
        so_true = n_i/(mt**(1/(o - 1)) - 1)
        sig_opt = max(so_true, 1)
        # sig_opt = min(sig_opt, n_i)
    else:
        so_true = 1
        sig_opt = 1
    return so_true, sig_opt

def distance_energy_per_unit_cc(c, o, e, n_i, eps, excl=False):
    so_true, sig_opt = get_sig_opt(c, o, n_i, e)
    o_delta = analytical_code_distance(o, c, excl=excl)
    p_so = analytical_code_variance_cc(o, c, n_i, sig_opt, excl=excl)
    v_so = (1/eps)*(e - analytical_code_terms_cc(o, c, n_i, sig_opt, excl=excl))
    delta_so = np.sqrt(v_so/p_so)*o_delta
    if np.isnan(delta_so):
        delta_so = -1
    return delta_so, sig_opt

def get_stim_cw_neighbors(c, o, n_i, rf, r2=2, cw_r2=None):
    _, s, trs = generate_cc_types((n_i,)*c, (rf,)*c, order=o, excl=True)
    ind = np.argmin(np.sum(np.abs(s - (n_i/2,)*c), axis=1))
    r = np.sqrt(r2)
    ds = u.euclidean_distance(s, s[ind])
    close_stim_mask = ds <= r
    all_cw = trs(s)
    ds_cw = u.euclidean_distance(all_cw, all_cw[ind])
    if cw_r2 is None:
        ds_nextmin = ds_cw[ds_cw > 0]
        md = np.min(ds_nextmin)
    else:
        md = np.sqrt(cw_r2)
    md_mask = ds_cw == md
    n_md_neigh_close = np.sum(np.logical_and(close_stim_mask, md_mask))
    n_md_neigh = np.sum(md_mask)
    
    # blah
    ana_c_cw = volume_nball(r, o) - 1
    # if o == 1 or (rf > 1 and o < c):
    if o < c:
        ana_c_cw = 2*r*c
    ana_all_cw = _close_words(c, o, n_i, rf=rf, excl=True)
    print('close and nn', ana_c_cw, '||', n_md_neigh_close)
    print('          nn', ana_all_cw, '||', n_md_neigh)
    print('min dist', md)
    print('close/all', n_md_neigh_close/n_md_neigh)
    return md, n_md_neigh_close, n_md_neigh

def prop_close_error(c, o, n_i, rf, r2=2):
    r = np.sqrt(r2)
    if o == c:
        if rf == 1:
            ana_c_est, ana_c_cw = integ_lattice_in_ball(r, c)
        else:
            ana_c_cw = _close_words(c, o, n_i, rf=rf, excl=True)
    else:
        ana_c_cw = 2*np.floor(r)*c
    ana_all_cw = _close_words(c, o, n_i, rf=rf, excl=True)
    return ana_c_cw/ana_all_cw, ana_all_cw

def hetero_error_count_nnub(c, o, v, noisevar, n_i, rf):
    excl = True
    arg = _nnup_density_arg(c, o, v, noisevar, n_i, rf=rf, excl=excl)
    dense_val = 1 - sts.norm(0, 1).cdf(arg)
    cw_ident, ref_wrd = _ident_close_words(c, o, n_i, rf, cent=True)
    mse_tot = np.sum((cw_ident - ref_wrd)**2, axis=1)
    print('actual', np.sum(mse_tot))
    print('actual mean', np.mean(mse_tot))
    if c == o:
        if rf > 1:
            mse_tot = 2*np.sum([sm.comb(o, i)*i for i in range(1, o + 1)])
        else:
            sum_end = int(np.floor((n_i - 1)/2))
            mse_tot1 = 2*c*(n_i**(c - 1))*np.sum([i**2 
                                                  for i in range(1, sum_end + 1)])
            mse_tot2 = c*(n_i**(c - 1))*(n_i - 1)*(n_i + 1)*n_i/12
            print('approx', mse_tot1, mse_tot2)
            mse_tot = mse_tot2
    else:
        if rf > 1:
            mse_tot = 2*c
        else:
            mse_tot = 10*c
    print('approx f', mse_tot)
    num_neigh = _close_words(c, o, n_i, excl=excl, rf=rf)
    est_e = (mse_tot/num_neigh)*dense_val
    print('approx mean', mse_tot/num_neigh)
    return est_e

def get_sse(c, o, n_i, rf=1):
    if c == o:
        if rf > 1:
            mse_tot = 2*(2**(c - 1))*c
        else:
            mse_tot = c*(n_i**(c - 1))*(n_i - 1)*(n_i + 1)*n_i/12
    else:
        if rf > 1:
            mse_tot = 2*c
        else:
            mse_tot = 10*c
    return mse_tot

def hetero_error_full_ana_nnub(c, o, v, noisevar, n_i, rf, bound_at_one=True,
                               distortion='mse', correction=False,
                               pwr_func=analytical_code_variance,
                               full_series=False, scale_func=l2_filt_scale):
    excl = True
    if distortion == 'mse':
        arg = _nnup_density_arg(c, o, v, noisevar, n_i, rf=rf, excl=excl,
                                pwr_func=pwr_func, scale_func=scale_func)
        dense_val = 1 - sts.norm(0, 1).cdf(arg)
        num_neigh = _close_words(c, o, n_i, excl=excl, rf=rf)
        mse_tot = get_sse(c, o, n_i, rf=rf)
        est_e = mse_tot*dense_val
    elif distortion == 'hamming':
        est_e = analytical_error_probability_nnup(c, o, v, noisevar, n_i,
                                                  excl=excl,
                                                  bound_at_one=bound_at_one,
                                                  pwr_func=pwr_func,
                                                  correction=correction,
                                                  full_series=full_series,
                                                  scale_func=scale_func)
    else:
        raise Exception('distortion is not one of "mse", "hamming" or blank')
    return est_e    

def hetero_error_nnub(c, o, v, noisevar, n_i, rf, r2=1, excl=True, 
                      compute_stimdists=False):
    arg = _nnup_density_arg(c, o, v, noisevar, n_i, rf=rf, excl=excl)
    dense_val = 1 - sts.norm(0, 1).cdf(arg)
    prop, nn = prop_close_error(c, o, n_i, rf, r2=r2)
    close_e = nn*prop*dense_val
    far_e = nn*(1 - prop)*dense_val
    return close_e, far_e, prop, nn
    
def compute_num_neighbors(c, o, n_i, rf, one_feat=False):
    _, s, trs = generate_cc_types((n_i,)*c, (rf,)*c, order=o, excl=True)
    ind = np.argmin(np.sum(np.abs(s - (n_i/2,)*c), axis=1))
    if one_feat:
        fv = s[0, 0]
        mask = s[:, 0] != fv
        ind = 0
        print((n_i-1)*n_i**(c - 1))
    else:
        mask = np.ones(len(s))
    cw = trs(s)
    ds = u.euclidean_distance(cw[mask], cw[ind])
    ds = ds[ds > 0]
    md = np.min(ds)
    n_neigh = np.sum(ds == md)
    return md, n_neigh

def distance_energy_per_unit(c, o, e, n_i, eps, streams=1, excl=False,
                             include_cost=True, pwr_sqrt=True,
                             pwr_func=analytical_code_variance):
    # sub_c = (c/streams) + 1
    sub_c = c
    assert(int(sub_c) == sub_c)
    assert(sub_c >= o)
    o_delta2 = analytical_code_distance(o, sub_c, excl=excl)
    p = streams*pwr_func(o, sub_c, n_i, excl=excl)
    d = analytical_code_terms(o, sub_c, n_i, excl=excl)
    v = (1/eps)*(e - include_cost*streams*d)
    if pwr_sqrt:
        delta2 = o_delta2*np.sqrt(v/p)
    else:
        delta2 = o_delta2*v/p
    if np.isnan(delta2) or d > e:
        delta2 = -1
    return delta2

def total_energy_modules(c, o, ns, e, m, eps=1):
    prefact = 2*o/(c*eps)
    subterm = m*sm.comb(c/m, o)*(ns**o)
    d = prefact*(e - subterm)
    return d

def total_energy_order_transitions(c, o, ns):
    first_term = ns*c - (ns + 1)*o
    second_term = sm.comb(c, o)*(ns**o)
    es = first_term*second_term
    return es

def total_energy_order_transitions_all(c, ns):
    bounds = np.zeros((c - 1, len(ns)))
    for i, o in enumerate(range(1, c)):
        bounds[i] = total_energy_order_transitions(c, o, ns)
    return bounds

def rigotti_repl(c, o, n_i, snrs, times_samp=10, excl=False, 
                 neurs=1000, nuis=False, nuis_prob=.5, bs_samps=1000):
    sigs = np.array(snrs)**2
    nv = 1
    c_dims = np.zeros(len(snrs))
    ic_dims = np.zeros_like(c_dims)

    for i, s in enumerate(sigs):
        out = simulate_transform_code_out(c, o, n_i, nv, s, 
                                          neurs=neurs,
                                          times_samp=times_samp, 
                                          excl=excl)
        owords, dec_words, ns, corr, bt, words, trs, sel = out
        owords_nb = get_original_from_binary(bt, words, owords)
        if nuis:
            nonlin_sel = np.array([len(s) > 1 for s in sel]).reshape((1, -1))
            nuis_occur = np.random.rand(owords.shape[0]) > nuis_prob
            nuis_occur = nuis_occur.reshape((-1, 1))
            sub_mask = nuis_occur*nonlin_sel*trs(owords)
            ns = ns - sub_mask
            corr = np.logical_not(np.logical_and(np.logical_not(corr), 
                                                 nuis_occur[:, 0]))
        c_dims[i], ic_dims[i] = estimate_code_dimensionality(corr, owords, 
                                                             ns, bt)
        incorr_mask = corr.astype(bool)
        corr_mask = np.logical_not(corr.astype(bool))
        acd_corr = decode_single_attribute(words, owords_nb[corr_mask], 
                                              ns[corr_mask])
        acd_incorr = decode_single_attribute(words, owords_nb[incorr_mask], 
                                            ns[incorr_mask])
        if i == 0:
            feat_corrs = np.zeros((words.shape[1], len(sigs), bs_samps))
            feat_incorrs = np.zeros_like(feat_corrs)
        for j, d in enumerate(acd_corr):
            feat_corrs[j, i] = u.bootstrap_list(d.flatten(), np.mean, bs_samps)
            feat_incorrs[j, i] = u.bootstrap_list(acd_incorr[j].flatten(),
                                                  np.mean, bs_samps)
    return c_dims, ic_dims, feat_corrs, feat_incorrs

