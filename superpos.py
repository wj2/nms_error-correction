
import numpy as np
import scipy.optimize as spo
import scipy.special as ss
import itertools as it

def awgn_capacity(snr):
    return .5*np.log(1 + snr)

def calculate_min_partitionlen(num_partitions, snr, rate=None, snr_thr=15.8):
    a = calculate_a(snr, rate=rate, snr_thr=snr_thr)
    b = num_partitions**a
    return a, b
    
def calculate_a(snr, rate=None, snr_thr=15.8):
    if np.all(snr < snr_thr):
        if rate is not None:
            denom = ((1 + snr)*np.log(1 + snr) - snr*np.log(np.e))**2
            num = rate*8*snr*(1 + snr)*np.log(np.e)
        else:
            num = 4*snr*(1 + snr)*np.log(1 + snr)
            denom = ((1 + snr)*np.log(1 + snr) - snr)**2
    else:
        if rate is not None:
            denom = (1 + snr)*np.log(1 + snr) - 2*snr*np.log(np.e)
            num = rate*2*(1 + snr)
        else:
            num = (1 + snr)*np.log(1 + snr)
            denom = (1 + snr)*np.log(1 + snr) - 2*snr
    return num/denom

def calculate_avl(l, snr, rate=None, rate_percent=.90):
    if rate is None:
        rate = awgn_capacity(snr)*rate_percent
    alpha = np.arange(1, l)/l
    dav = d1_alpha_snr(alpha, snr)
    avl = rate*np.log(ss.comb(l, alpha*l))/(dav*l*np.log(l))
    a = np.max(avl)
    return a

def omp_alph2(alpha, snr):
    return alpha*(1 - alpha)*snr/(1 + alpha*snr)

def delt_alphabar(alpha, snr):
    dab = awgn_capacity(alpha*snr) - alpha*awgn_capacity(snr)
    return dab

def d1_alpha_snr(alpha, snr):
    ompa = omp_alph2(alpha, snr)
    dab = delt_alphabar(alpha, snr)
    das = d1(dab, ompa)
    return das

def d1(delt, omp, eps=0):
    f = lambda l: l*delt + .5*np.log(1 - (l**2)*omp)
    f_opt = lambda l: -np.sum(f(l))
    l_guess = np.ones_like(delt)*.5
    bounds = ((eps, 1 - eps),)*len(l_guess)
    d1_star = spo.minimize(f_opt, l_guess, bounds=bounds)
    return f(d1_star.x)
    
class SuperpositionCode(object):

    def __init__(self, snr, partition_len, num_partitions, rate_percent,
                 noise_var=5):
        self.capacity = .5*np.log(1 + snr)
        approx_rate = self.capacity*rate_percent
        approx_n = np.log(partition_len)*num_partitions/approx_rate
        self.little_n = int(np.ceil(approx_n))
        self.rate = num_partitions*np.log(partition_len)/self.little_n
        self.noise_var = noise_var
        self.snr = snr
        self.num_partitions = num_partitions
        
        self.power = snr*noise_var
        x_var = self.power/num_partitions
        self.big_n = partition_len*num_partitions
        self.x_mat = np.random.randn(self.little_n, self.big_n)*np.sqrt(x_var)
    
        self.betas = self._construct_betas(partition_len, num_partitions)
        self.n_stim = self.betas.shape[1]
        assert self.big_n == self.betas.shape[0]
        self.codewords = np.matmul(self.x_mat, self.betas)

    def _construct_betas(self, b, l):
        lists = np.identity(b)
        sets = it.product(lists, repeat=l)
        bs = np.stack([np.concatenate(x) for x in sets], axis=1)
        return bs

    def estimate_performance(self, n=5000):
        stim_inds = np.random.randint(0, self.n_stim, size=n)
        cw_mat = self.codewords[:, stim_inds]
        noise_mat = np.sqrt(self.noise_var)*np.random.randn(*cw_mat.shape)
        noisy_cw = cw_mat + noise_mat
        dec_inds = self.decode_noisy_codewords(noisy_cw)
        corr_rate = np.mean(stim_inds == dec_inds)
        return 1 - corr_rate

    def decode_noisy_codewords(self, ncw):
        exp_ncw = np.expand_dims(ncw, axis=2)
        cw_exp = np.expand_dims(self.codewords, axis=1)
        diff = np.sum((exp_ncw - cw_exp)**2, axis=0)
        inds = np.argmin(diff, axis=1)
        return inds

    def upperbound(self):
        num_p = self.num_partitions
        alpha = np.arange(1, num_p)/num_p
        delt_alph = awgn_capacity(alpha*self.snr) - alpha*self.rate
        d1_channel = d1(delt_alph, alpha*self.snr/(1 + alpha*self.snr))
        ub = ss.comb(num_p, alpha*num_p)*np.exp(-self.little_n*d1_channel)
        return ub, d1_channel
        
def sp_upperbound(snr, num_p, part_len, rate_percent=.9):
    approx_rate = awgn_capacity(snr)*rate_percent
    approx_n = np.log(part_len)*num_p/approx_rate
    little_n = int(np.ceil(approx_n))
    rate = num_p*np.log(part_len)/little_n
    alpha = np.arange(1, num_p)/num_p
    delt_alph = awgn_capacity(alpha*snr) - alpha*rate
    d1_channel = d1(delt_alph, alpha*snr/(1 + alpha*snr))
    ub = ss.comb(num_p, alpha*num_p)*np.exp(-little_n*d1_channel)
    return np.max(ub), little_n
    
