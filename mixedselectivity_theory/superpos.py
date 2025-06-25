
import numpy as np
import scipy.optimize as spo
import scipy.special as ss
import itertools as it

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

def awgn_capacity(snr):
    return .5*np.log2(1 + snr)

def calculate_min_partitionlen(num_partitions, snr, rate=None, snr_thr=15.8):
    a = calculate_a(snr, rate=rate, snr_thr=snr_thr)
    b = num_partitions**a
    return a, b
    
def calculate_a(snr, rate=None, snr_thr=15.8):
    if np.all(snr < snr_thr):
        if rate is not None:
            denom = ((1 + snr)*np.log2(1 + snr) - snr*np.log2(np.e))**2
            num = rate*8*snr*(1 + snr)*np.log2(np.e)
        else:
            num = 4*snr*(1 + snr)*np.log2(1 + snr)
            denom = ((1 + snr)*np.log2(1 + snr) - snr)**2
    else:
        if rate is not None:
            denom = (1 + snr)*np.log2(1 + snr) - 2*snr*np.log2(np.e)
            num = rate*2*(1 + snr)
        else:
            num = (1 + snr)*np.log2(1 + snr)
            denom = (1 + snr)*np.log2(1 + snr) - 2*snr
    return num/denom

def calculate_avl(l, snr, rate=None, rate_percent=.90):
    if rate is None:
        rate = awgn_capacity(snr)*rate_percent
    alpha = np.arange(1, l)/l
    dav = d1_alpha_snr(alpha, snr)
    avl = rate*np.log2(ss.comb(l, alpha*l))/(dav*l*np.log2(l))
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
    f = lambda l: l*delt + .5*np.log2(1 - (l**2)*omp)
    f_opt = lambda l: -np.sum(f(l))
    l_guess = np.ones_like(delt)*.5
    bounds = ((eps, 1 - eps),)*len(l_guess)
    d1_star = spo.minimize(f_opt, l_guess, bounds=bounds)
    return f(d1_star.x)

class SCDecoder(object):

    def _compile(self, optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                 loss=tf.losses.CategoricalCrossentropy()):
        self.model.compile(optimizer, loss)
        self.compiled = True
    
    def train(self, train_inp, train_out, eval_inp=None, eval_out=None,
              epochs=25, batch_size=32):
        if not self.compiled:
            self._compile()

        if eval_inp is not None and eval_out is not None:
            eval_set = (eval_inp, eval_out)
        else:
            eval_set = None

        out = self.model.fit(x=train_inp, y=train_out, epochs=epochs,
                             validation_data=eval_set, batch_size=batch_size)
        return out

    def __call__(self, x):
        return self.model(x)

class OLSDecoder(SCDecoder):

    def __init__(self, code, first_only=False, **kwargs):
        if code.codewords is None:
            code.generate_all_codewords()
        self.cw = code.codewords
        self.targets = code.betas
        self.first_only = first_only
        self.m = code.partition_len

    def model(self, x):
        exp_ncw = np.expand_dims(x, axis=2)
        cw_exp = np.expand_dims(self.cw, axis=1)
        diff = np.sum((exp_ncw - cw_exp)**2, axis=0)
        inds = np.argmin(diff, axis=1)
        out = self.targets[inds]
        if self.first_only:
            out = out[:, :self.m]
        return self.targets[inds]

    def train(self, *args, **kwargs):
        return tf.keras.callbacks.History()

class IterativeDecoder(SCDecoder):

    def __init__(self, code, first_only=False, tau=None):
        self.x = code.x_mat
        self.tau = tau
        self.l = code.num_partitions
        self.m = code.partition_len
        if tau is None:
            a = self._compute_a(code)
            self.tau = np.sqrt(2*np.log(self.m)) + a
        self.first_only = first_only

    def _compute_a(self, code):
        num = (3/2)*np.log(np.log(code.partition_len))
        denom = np.sqrt(2*np.log(code.partition_len))
        print('snr', np.log(code.snr*(1 + 1/code.capacity)/(np.pi**(1/4))))
        a_bar_num = 2*np.log(code.snr*(1 + 1/code.capacity)/(np.pi**(1/4)))
        print(num, a_bar_num)
        print(num/denom, a_bar_num/denom)
        a = num/denom + a_bar_num/denom
        return a
        
    def _decoder(self, y):
        k = 0
        dec_k = set()
        f_k1 = np.zeros(y.shape)
        while k < self.m and len(dec_k) < self.l:
            r = y - f_k1
            rl = np.sqrt(np.sum(r**2))
            z_kj = np.dot(self.x.T, r)/rl
            dec_k = dec_k.union(np.where(z_kj >= self.tau)[0])
            f_k1 = np.sum(self.x[:, list(dec_k)], axis=1)
            k = k + 1
        return list(dec_k)
    
    def model(self, y):
        dec = np.zeros((y.shape[0], self.l*self.m), dtype=bool)
        for i, y_i in enumerate(y):
            inds = self._decoder(y_i)
            dec[i, inds] = 1
        if self.first_only:
            dec = dec[:, :self.m]
        return dec
        
    def train(self, *args, **kwargs):
        return tf.keras.callbacks.History()
    
class FFDecoder(SCDecoder):

    def __init__(self, code, first_only=True, layers=None, act_func=tf.nn.relu,
                 layer_type=tfkl.Dense, **layer_params):
        self.inp_dim = code.little_n
        self.out_dim = code.partition_len
        if layers is None:
            layers = ((self.inp_dim,), (self.out_dim,), (self.out_dim,))
        self.decoder = self._make_decoder(self.inp_dim, layers, self.out_dim,
                                          act_func=act_func,
                                          layer_type=layer_type,
                                          **layer_params)
        self.model = tfk.Model(inputs=self.decoder.inputs,
                               outputs=self.decoder.outputs[0])
        self.compiled = False
        self.first_only = first_only
        if not self.first_only:
            raise NotImplementedError('the FFDecoder can only decode the first '
                                      'section at this time')

    def _make_decoder(self, inp, layer_shapes, out, act_func=tf.nn.relu,
                      layer_type=tfkl.Dense, **layer_params):
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=inp))
            
        for lp in layer_shapes:
            l_i = layer_type(*lp, activation=act_func, **layer_params)
            layer_list.append(l_i)

        layer_list.append(layer_type(out, activation=act_func, **layer_params))
        layer_list.append(tfkl.Softmax())

        dec = tfk.Sequential(layer_list)
        return dec
    
class SuperpositionCode(object):

    def __init__(self, snr, partition_len, num_partitions, rate_percent,
                 set_n=None, noise_var=5, lazy_codewords=True):
        self.capacity = .5*np.log2(1 + snr)
        set_rate = self.capacity*rate_percent
        if set_n is not None:
            self.little_n = set_n
            num_partitions = int(round(set_n*set_rate/np.log2(set_n)))
            partition_len = set_n
        else:
            approx_n = np.log2(partition_len)*num_partitions/set_rate
            self.little_n = int(np.ceil(approx_n))
        self.rate = num_partitions*np.log2(partition_len)/self.little_n
        self.noise_var = noise_var
        self.snr = snr
        self.num_partitions = num_partitions
        self.partition_len = partition_len
        
        self.power = snr*noise_var
        x_var = self.power/num_partitions
        self.big_n = partition_len*num_partitions
        self.x_mat = np.random.randn(self.little_n, self.big_n)*np.sqrt(x_var)
        
        self.n_stim = partition_len**num_partitions
        if not lazy_codewords:
            self.generate_all_codewords()
        else:
            self.betas = None
            self.codewords = None
        self.decoder = None
            
    def generate_noisy_egs(self, n=10**6):
        gen_shape = n*self.num_partitions
        w_inds = np.random.randint(0, self.partition_len, size=(gen_shape, 1))
        z_combs = list(it.product(range(n), range(self.num_partitions)))
        z_inds = np.concatenate((z_combs, w_inds), axis=1)
        ws = np.zeros((n, self.num_partitions, self.partition_len), dtype=bool)
        ws[z_inds[:, 0], z_inds[:, 1], z_inds[:, 2]] = 1
        ws = np.reshape(ws, (-1, self.num_partitions*self.partition_len))
        cw = np.matmul(self.x_mat, ws.T).T
        ncw = cw + np.sqrt(self.noise_var)*np.random.randn(*cw.shape)
        return ws, cw, ncw
        
    def generate_all_codewords(self):
        self.betas = self._construct_betas(partition_len, num_partitions)
        self.codewords = np.matmul(self.x_mat, self.betas)
            
    def _construct_betas(self, b, l):
        lists = np.identity(b)
        sets = list(it.product(lists, repeat=l))
        print(len(sets))
        bs = np.stack([np.concatenate(x) for x in sets], axis=1)
        return bs

    def estimate_performance(self, n=5000):
        if self.decoder is None:
            raise Exception('no decoder')
        targ, cw, ncw = self.generate_noisy_egs(n=n)
        rep = self.decoder(ncw)
        rep_ind = np.argmax(rep, axis=1)
        targ_ind = np.argmax(targ, axis=1)
        p_corr = np.sum(rep_ind == targ_ind)/n
        return targ, rep, p_corr

    def decode_noisy_codewords(self, ncw):
        exp_ncw = np.expand_dims(ncw, axis=2)
        if self.codewords is None:
            self.generate_all_codewords()
        cw_exp = np.expand_dims(self.codewords, axis=1)
        diff = np.sum((exp_ncw - cw_exp)**2, axis=0)
        inds = np.argmin(diff, axis=1)
        return inds

    def make_decoder(self, decoder_class, first_only=True, **kwargs):
        dec = decoder_class(self, first_only=first_only, **kwargs)
        self.decoder = dec

    def train_decoder(self, train_n=10**6, eval_n=1000, decoder_class=FFDecoder,
                      first_only=True, **kwargs):
        if self.decoder is None:
            self.make_decoder(decoder_class, first_only=first_only)
        train_targs, _, train_inps = self.generate_noisy_egs(n=train_n)
        eval_targs, _, eval_inps = self.generate_noisy_egs(n=eval_n)
        if first_only:
            train_targs = train_targs[:, :self.partition_len]
            eval_targs = eval_targs[:, :self.partition_len]
        h = self.decoder.train(train_inps, train_targs, eval_inps, eval_targs,
                               **kwargs)
        return h
    
    def upperbound(self):
        num_p = self.num_partitions
        alpha = np.arange(1, num_p)/num_p
        delt_alph = awgn_capacity(alpha*self.snr) - alpha*self.rate
        d1_channel = d1(delt_alph, alpha*self.snr/(1 + alpha*self.snr))
        ub = ss.comb(num_p, alpha*num_p)*np.exp(-self.little_n*d1_channel)
        return ub, d1_channel
        
def sp_upperbound(snr, num_p, part_len, rate_percent=.9):
    approx_rate = awgn_capacity(snr)*rate_percent
    approx_n = np.log2(part_len)*num_p/approx_rate
    little_n = int(np.ceil(approx_n))
    rate = num_p*np.log2(part_len)/little_n
    alpha = np.arange(1, num_p)/num_p
    delt_alph = awgn_capacity(alpha*snr) - alpha*rate
    d1_channel = d1(delt_alph, alpha*snr/(1 + alpha*snr))
    ub = ss.comb(num_p, alpha*num_p)*np.exp(-little_n*d1_channel)
    return np.max(ub), little_n
    
