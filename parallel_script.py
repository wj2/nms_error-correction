#!/usr/bin/env python

import numpy as np
import mixedselectivity_theory.nms_continuous as nmc
import pickle as p
import datetime

c = 3
n = 5
v = 8
rf_size = 1
excl = True
buff = None
if buff is None:
    buff = rf_size
    
orders = np.arange(1, c + 1, 1)

snrs = np.linspace(2, 10, 5)
n_samps = 2
pm = 'distance'
give_real = True
basin_hop = False
parallel = True
oo = True

dt = str(datetime.datetime.now()).replace(' ', '-')


gr_str = {True:'_givereal', False:''}
bh_str = {True:'_basinhop', False:''}
par_str = {True:'_par', False:''}
oo_str = {True:'_oo', False:''}

name = 'continuous-nms_{}{}{}{}{}_{}.pkl'.format(dt, gr_str[give_real],
                                                  bh_str[basin_hop],
                                                  par_str[parallel],
                                                  oo_str[oo],
                                                  pm)

perf = np.zeros((len(orders), len(snrs), n_samps))
for i, o in enumerate(orders):
    print('O = {}'.format(o))
    d = nmc.estimate_code_performance_overpwr(c, o, n, snrs, rf_size, 
                                              power_metric=pm,
                                              samps=n_samps, excl=excl,
                                              give_real=give_real, 
                                              basin_hop=basin_hop,
                                              parallel=parallel, oo=oo)
    perf[i] = d

out_dict = {'orders':orders, 'snrs':snrs, 'perf':perf, 'c':c, 'n':n,
            'rf_size':rf_size}
p.dump(out_dict, open(name, 'wb'))
