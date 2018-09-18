#!/usr/bin/env python

import argparse
import numpy as np
import pickle as p
import datetime
import os

def create_parser():
    parser = argparse.ArgumentParser(description='run continuous NMS '
                                     'simulations')
    parser.add_argument('-c', '--n_features', default=3, type=int,
                        help='number of features for stim set')
    parser.add_argument('-n', '--n_values', default=5, type=int,
                        help='number of values each feature can take on')
    parser.add_argument('-r', '--rf_size', default=1, type=float,
                        help='width of Gaussian rfs')
    parser.add_argument('-b', '--buff', default=None,
                        help='distance from the edges of stimulus space'
                        'to sample from so as to avoid edge effects')
    parser.add_argument('snr_begin', type=float, help='start SNR samples '
                        'at')
    parser.add_argument('snr_end', type=float, help='end SNR samples at')
    parser.add_argument('snr_n', type=int, help='take this many different '
                        'SNRs')
    parser.add_argument('-s', '--n_samps', type=int, default=1000,
                        help='sample each order/SNR this many times')
    parser.add_argument('-p', '--power_measure', type=str, default='distance',
                        help='measure to use to evaluate the power consumption '
                        'of a code')
    parser.add_argument('--give_real', default=False,
                        action='store_true', help='start decoder optimization '
                        'at the correct value of the stimulus')
    parser.add_argument('--basin_hop', default=True, action='store_false',
                        help='use basinhopping')
    parser.add_argument('--parallel', default=True, action='store_false',
                        help='run in parallel')
    parser.add_argument('--oo', default=True, action='store_false',
                        help='use object oriented code interface')
    parser.add_argument('--outfolder', help='path to store pickle at',
                        default='./', type=str)
    parser.add_argument('--runfolder', default='./', type=str,
                        help='path to run the script from')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    os.chdir(args.runfolder)
    import mixedselectivity_theory.nms_continuous as nmc

    c = args.n_features
    n = args.n_values
    rf_size = args.rf_size
    excl = True
    buff = args.buff
    if buff is None:
        buff = rf_size
    orders = np.arange(1, c + 1, 1)
    snrs = np.linspace(args.snr_begin, args.snr_end, args.snr_n)
    n_samps = args.n_samps
    pm = args.power_measure
    give_real = args.give_real
    basin_hop = args.basin_hop
    parallel = args.parallel
    oo = args.oo

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
    name = os.path.join(args.outfolder, name)

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
                'rf_size':rf_size, 'args':args}
    p.dump(out_dict, open(name, 'wb'))
