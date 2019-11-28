
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import matplotlib.colors as mpl_c
import matplotlib.lines as lines
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import matplotlib.colors as mpl_c

import mixedselectivity_theory.nms_discrete as nmd
import mixedselectivity_theory.utility as u2
import general.plotting as gpl

basefolder = ('/Users/wjj/Dropbox/research/uc/freedman/analysis/'
              'mixedselectivity_theory/figs/')

colors = np.array([(127,205,187),
                   (65,182,196),
                   (29,145,192),
                   (34,94,168),
                   (37,52,148),
                   (8,29,88)])/256

def setup():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc('axes', prop_cycle=cycler('color', colors))
    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'axes.labelsize':'medium'})
    plt.rcParams.update({'lines.linewidth':3})

""" figure 2 """
def rep_energy_errors(c, ords, var, noisevar, n_i, n_samps, pwr_func,
                      noise_func, excl=True, **kwargs):
    emp_corr = np.zeros((len(ords), len(var), n_samps))
    bound_corr = np.zeros((len(ords), len(var)))
    for i, o in enumerate(ords):
        bound_corr[i] = nmd.error_upper_bound_overpwr(c, o, var, noisevar,
                                                      n_i, excl=excl)
        out = nmd.estimate_real_perc_correct_overpwr(c, o, n_i, noisevar, 
                                                     var, n_samps=n_samps,
                                                     pwr_func=pwr_func,
                                                     excl=excl,
                                                     noise_func=noise_func,
                                                     **kwargs)
        emp_corr[i] = out
    return emp_corr, bound_corr

def plot_rep_energy_ord(emp_corr, bound_corr, ords, var, noisevar, ax,
                        x_ins=(5,6), y_ins=(.0001, .12), xlab='SNR',
                        log_x = False, ylab='error rate (PE)', ylim=(0, 1),
                        inset=True, plot_bound_main=False, ins_logy=True,
                        bound_edge_color='k', bound_edge_wid=2):
    if inset:
        ax_i = zoomed_inset_axes(ax, 3.5, loc=7)
        ax_i.set_xlim(x_ins)
        ax_i.set_ylim(y_ins)
    snrs = np.sqrt(var/noisevar)
    for i, o in enumerate(ords):
        l = gpl.plot_trace_werr(snrs, emp_corr[i].T, log_x=log_x,
                                 error_func=gpl.conf95_interval, ax=ax, 
                                 label='O = {}'.format(o))
        col = l[0].get_color()
        lw = l[0].get_linewidth()
        bound_pe = [pe.Stroke(linewidth=lw + bound_edge_wid,
                              foreground=bound_edge_color),
                    pe.Normal()]
        if plot_bound_main:
            _ = ax.plot(snrs, bound_corr[i], '--', color=col,
                        path_effects=bound_pe)
        if inset:
            gpl.plot_trace_werr(snrs, emp_corr[i].T, color=col, log_y=ins_logy,
                                error_func=gpl.conf95_interval, ax=ax_i,
                                label='O = {}'.format(o), legend=False)
            ax_i.plot(snrs, bound_corr[i], '--', color=col,
                      path_effects=bound_pe)
    ax.set_ylim(ylim)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    if inset:
        ax_i.set_xticks(x_ins)
        mark_inset(ax, ax_i, loc1=2, loc2=4, fc="none", ec="0.5")
    ax.legend(frameon=False)
    return ax

def plot_err_at_snr(nc_s, n_i, snr, pwr_func, ax1, ax2, text_buff=.07,
                    excl=True, logy=True, ord_cols=colors):
    kwargs = {'pwr_func':pwr_func,'excl':excl}
    if logy:
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    nv = 10
    var = nv*(snr**2)
    ax2.set_xlim([min(nc_s) - .75, max(nc_s) + .5])
    err_ratios = np.zeros(len(nc_s))
    best_ords = np.zeros(len(nc_s), dtype=int)
    for i, nc in enumerate(nc_s):
        os = np.arange(1, nc + 1)
        errs = [nmd.analytical_error_probability_nnup(nc, o, var, nv, n_i,
                                                      **kwargs)
                for o in os]
        errs_arr = np.array(list(errs))
        l = gpl.plot_trace_werr(os, errs_arr, label=r'$K = '+r'{}$'.format(nc),
                                ax=ax1, color=ord_cols[i])
        err_ratio = np.array(errs_arr[0]/min(errs_arr))
        err_ratios[i] = err_ratio
        gpl.plot_trace_werr(np.array(nc), err_ratio, marker='o', 
                            color=ord_cols[i], ax=ax2)
        best_ord = os[np.argmin(errs_arr)]
        best_ords[i] = best_ord
    yl2 = np.log10(ax2.get_ylim())
    xl2 = ax2.get_xlim()
    for i, nc in enumerate(nc_s):
        err_ratio = err_ratios[i]
        best_ord = best_ords[i]
        xtxt = (nc - xl2[0])/(xl2[1] - xl2[0])
        ytxt = (np.log10(err_ratio) - yl2[0])/(yl2[1] - yl2[0])
        ax2.text(xtxt, ytxt + text_buff, 'O = {}'.format(best_ord),
                 color=colors[i], horizontalalignment='center',
                 transform=ax2.transAxes)
    ax1.legend(frameon=False)
    ax1.set_xlabel('order (O)')
    ax1.set_ylabel('error rate at SNR = {}'.format(snr))
    ax2.set_ylabel('pure/mixed error ratio')
    ax2.set_xticks(nc_s)
    ax2.set_xlabel(r'number of features ($K$)')
    return ax1, ax2

def plot_target_error(nc_s, n_i, target_err, pwr_func, ax1, ax2, text_buff=15,
                      excl=True, subdim=False, eps=1, logy=False, rf=1,
                      distortion='hamming', nv=10):

    kwargs = {'subdim':subdim, 'pwr_func':pwr_func, 'eps':eps, 'rf':rf, 
              'distortion':distortion}
    if logy:
        ax1.set_yscale('log')
    for i, nc in enumerate(nc_s):
        os = np.arange(1, nc + 1)
        snrs = np.array(list([nmd.pwr_require_err(target_err, nc, o, n_i,
                                                  **kwargs)[0] 
                              for o in os]))[:, 0]
        re = nv*snrs**2
        l = gpl.plot_trace_werr(os, re, label=r'$K = '+r'{}$'.format(nc),
                                ax=ax1, color=colors[i])
        re_ratio = np.array(re[0]/min(re)*100)
        gpl.plot_trace_werr(np.array(nc), re_ratio, marker='o', 
                            color=colors[i], ax=ax2)
        best_ord = os[np.argmin(re)]
        ax2.text(nc, re_ratio + text_buff, 'O = {}'.format(best_ord),
                 color=colors[i], horizontalalignment='center')
    
    ax1.legend(frameon=False)
    ax1.set_xlabel('order (O)')
    te_perc = int(target_err*100)
    ax1.set_ylabel('represenation energy for\n{}% error rate'.format(te_perc))
    
    ax2.set_ylabel('% representation energy for pure\nrelative to best mixed code')
    ax2.set_yticks([100, 250, 400])
    ax2.set_xticks(nc_s)
    ax2.set_xlabel(r'number of features ($K$)')
    ax2.set_xlim([min(nc_s) - .75, max(nc_s) + .5])

    return ax1, ax2
    
def compute_optimal_order(c, n_is, es, eps, pwr_func, excl=True, dim_cost=True):
    ds = np.zeros((c, len(n_is), len(es)))
    for j, n_i in enumerate(n_is):
        for i, o in enumerate(range(1, c + 1)):
            ds[i, j] = [nmd.distance_energy_per_unit(c, o, e, n_i, eps,
                                                     excl=excl,
                                                     include_cost=dim_cost,
                                                     pwr_func=pwr_func) 
                        for e in es]
    return ds

def make_order_colorbar(c, colmesh, f, axs):
    ticks = range(0, c + 1)
    ticklabels = ['invalid'] + [r'$O = {}$'.format(o) for o in range(1, c+1)]
    colbar = f.colorbar(colmesh, ax=axs)
    colbar.set_ticks(ticks)
    colbar.set_ticklabels(ticklabels)
    return colbar

def plot_order_map(ds, c, n_is, es, ax, es_tbs=None, n_is_es=None,
                   xlabel=r'total available energy ($E = \epsilon V + D_{O}$)',
                   ylabel=r'number of values ($n$)', box_bound=(10**5, 10**6)):
    colors_map = [(1, 1, 1)] + list(colors)
    labels = np.arange(-1, c + 1) + .5
    cmap, norm = mpl_c.from_levels_and_colors(labels, colors_map)

    best_ord = np.arange(1, c+1)[np.argmax(ds, axis=0)]
    invalid = np.all(ds < 0, axis=0)
    best_ord[invalid] = 0

    es_plot = gpl.pcolormesh_axes(es, ds.shape[2])
    n_is_plot = gpl.pcolormesh_axes(n_is, ds.shape[1])

    ax.set_xscale('log')
    ax.set_yscale('log')
    p = ax.pcolormesh(es_plot, n_is_plot, best_ord, cmap=cmap, vmin=-.5,
                      vmax=c+.5)
    p.set_edgecolor('face')
    p.set_rasterized(True)
    if es_tbs is not None:
        for i, l in enumerate(es_tbs):
            l_mask = np.logical_and(l > min(es), l < max(es))
            ax.plot(l[l_mask], n_is_es[l_mask], 'k', linestyle='dotted',
                    alpha=.4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    r_xy = (box_bound[0], n_is[0])
    r_wid = box_bound[1] - box_bound[0]
    r_hei = n_is[-1]
    r_col_fill = (.8,)*3
    r_col_edge = (.4,)*3
    r_alpha = .5
    r1 = plt.Rectangle(r_xy, r_wid, r_hei, facecolor=r_col_fill, 
                       edgecolor=r_col_edge, alpha=r_alpha)
    ax.add_artist(r1)
    return ax, p
    
def figure2(gen_panels=None, data=None):
    if gen_panels is None:
        gen_panels = ('a', 'b', 'c', 'd', 'sab')
    setup()
    # whole fig
    n_samps = 10000
    pwr_func = nmd.empirical_variance_power
    noise_func = nmd.gaussian_noise
    nv = 10
    var = np.linspace(.1, 1000, 300)
    if data is None:
        data = {}
    else:
        print('using supplied data, may be incorrect')
    # panel A
    if 'a' in gen_panels:
        c_a = 3
        ords_a = range(1, c_a + 1)
        ni_a = 5
        insetx_a = (5.5, 6.5)
        insety_a = (.00005, .1)
        
        panel_a_size = (4, 2)
        f_a = plt.figure(figsize=panel_a_size)
        ax_a = f_a.add_subplot(1,1,1)
        if 'ec_a' in data.keys():
            ec_a = data['ec_a']
            bc_a = data['bc_a']
        else:
            out = rep_energy_errors(c_a, ords_a, var, nv, ni_a, n_samps,
                                    pwr_func, noise_func, excl=True)
            ec_a, bc_a = out
            data['ec_a'] = ec_a
            data['bc_a'] = bc_a
        ax_a = plot_rep_energy_ord(ec_a, bc_a, ords_a, var, nv, ax_a,
                                   x_ins=insetx_a, y_ins=insety_a)
        fa_name = basefolder + 'e-snr-k{}.pdf'.format(c_a)
        f_a.savefig(fa_name, bbox_inches='tight', transparent=True)

    # panel B
    if 'b' in gen_panels:
        c_b = 5
        ords_b = range(1, c_b + 1)
        ni_b = 3
        insetx_b = (6, 7)
        insety_b = (.00005, .1)

        panel_b_size = panel_a_size 
        f_b = plt.figure(figsize=panel_b_size)
        ax_b = f_b.add_subplot(1,1,1)
        if 'ec_b' in data.keys():
            ec_b = data['ec_b']
            bc_b = data['bc_b']
        else:
            out = rep_energy_errors(c_b, ords_b, var, nv, ni_b, n_samps,
                                    pwr_func, noise_func, excl=True)
            ec_b, bc_b = out
            data['ec_b'] = ec_b
            data['bc_b'] = bc_b
        ax_b = plot_rep_energy_ord(ec_b, bc_b, ords_b, var, nv, ax_b,
                                   x_ins=insetx_b, y_ins=insety_b)
        fb_name = basefolder + 'e-snr-k{}.pdf'.format(c_b)
        f_b.savefig(fb_name, bbox_inches='tight', transparent=True)

    # panel C
    if 'c' in gen_panels:
        cs_c = range(2, 7)
        ni_c = 5
        snr = 9
        pwr_func_c = nmd.analytical_code_variance
        
        panel_c_size = (2.5, 4)
        f_c = plt.figure(figsize=panel_c_size)
        ax1, ax2 = f_c.add_subplot(2, 1, 1), f_c.add_subplot(2, 1, 2)
    
        ax1, ax2 = plot_err_at_snr(cs_c, ni_c, snr, pwr_func_c,
                                   ax1, ax2)
        fc_name = basefolder + 'error-at-snr_l2.svg'
        f_c.savefig(fc_name, bbox_inches='tight', transparent=True)

    # panel D
    if 'd' in gen_panels:
        eps = 200
        c_d = 6

        ni_d_beg = 0
        ni_d_end = 3
        ni_d_pts = 100
        ni_d = np.logspace(ni_d_beg, ni_d_end, ni_d_pts) + 1
        pwr_func_d = nmd.analytical_code_variance
        
        energ_beg = 3
        energ_end = 8
        energ_pts = 100
        es = np.logspace(energ_beg, energ_end, energ_pts)
        if 'ds_free' in data.keys():
            ds_free = data['ds_free']
            ds_ener = data['ds_ener']
        else:
            ds_free = compute_optimal_order(c_d, ni_d, es, eps, pwr_func_d,
                                            dim_cost=False)
            ds_ener = compute_optimal_order(c_d, ni_d, es, eps, pwr_func_d,
                                            dim_cost=True)
            data['ds_free'] = ds_free
            data['ds_ener'] = ds_ener
        panel_d_size = (7.5, 3)
        f_d = plt.figure(figsize=panel_d_size)
        ax_free = f_d.add_subplot(1,2,1)
        ax_ener = f_d.add_subplot(1,2,2, sharey=ax_free)
        out = plot_order_map(ds_free, c_d, ni_d, es, ax_free,
                             xlabel=r'available pop size ($N$)')
        ax_num, p_num = out
        
        transition_pts = 10000
        n_is_es = np.logspace(ni_d_beg, ni_d_end, transition_pts) + 1
        es_tbs = nmd.total_energy_order_transitions_all(c_d, n_is_es)
        out = plot_order_map(ds_ener, c_d, ni_d, es, ax_ener,
                             ylabel='', es_tbs=es_tbs, n_is_es=n_is_es)
        ax_free, p_free = out
        cb = make_order_colorbar(c_d, p_num, f_d, [ax_num, ax_free])
        fd_name = basefolder + 'opt-order-map.svg'
        f_d.savefig(fd_name, bbox_inches='tight', transparent=True)

    # panel supp AB
    if 'sab' in gen_panels:
        cs_sab = range(2, 7)
        ni_sab = 5
        targ_err = .01
        pwr_func_sab = nmd.analytical_code_variance
        
        panel_sab_size = (6, 2.5)
        f_sab = plt.figure(figsize=panel_sab_size)
        ax1, ax2 = f_sab.add_subplot(1, 2, 1), f_sab.add_subplot(1, 2, 2)
    
        ax1, ax2 = plot_target_error(cs_sab, ni_sab, targ_err, pwr_func_sab,
                                     ax1, ax2)
        fsab_name = basefolder + 'energy-req_l2.svg'
        f_sab.savefig(fsab_name, bbox_inches='tight', transparent=True)

    return data

def rep_energy_errors_dfunc_rf(c, ni, nv, var, n_samps, cc_rfs,
                               dfuncs, subdim=False, excl=True, bs=False,
                               eps=100, pwr_func=nmd.empirical_variance_power,
                               noise_func=nmd.gaussian_noise):
    ords = range(1, c + 1)
    corr = np.zeros((len(dfuncs), len(cc_rfs), len(ords), len(var), n_samps))
    for j, df in enumerate(dfuncs):
        for i, crf in enumerate(cc_rfs):
            out = rep_energy_errors(c, ords, var, nv, ni, n_samps, pwr_func,
                                    noise_func, cc_rf=(crf,)*c, excl=excl,
                                    subdim=subdim, bs=bs, distortion_func=df,
                                    eps=eps)
            corr[j, i] = out[0]
    return corr                               

def plot_rep_energy_ord_rf(corr, ords, rfs, var, noisevar, ax,
                           xlab='total energy (E)', ylab='error rate (PE)',
                           l_styles=None, ord_cols=colors, legend=True,
                           stylecol=(.2, .2, .2), vline_pt=None, vert_alpha=.3,
                           error_func=gpl.sem, logy=False, logx=False,
                           inset=False, x_ins=(1.5*10**4, 2.5*10**4),
                           y_ins=(.005, 5), logy_ins=True, logx_ins=False):
    if l_styles is None:
        l_styles = ('solid', 'dashed', 'dotted', 'dashdot')
    if inset:
        ax_i = zoomed_inset_axes(ax, 2.5, loc=1)
        ax_i.set_xlim(x_ins)
        ax_i.set_ylim(y_ins)

    for i, o in enumerate(ords):
        for j, rf in enumerate(rfs):
            l = gpl.plot_trace_werr(var, corr[j, i].T, error_func=error_func,
                                    ax=ax, linestyle=l_styles[j],
                                    color=ord_cols[i], log_y=logy, log_x=logx,
                                    legend=False)
            if inset:
                gpl.plot_trace_werr(var, corr[j, i].T, error_func=error_func,
                                    ax=ax_i, linestyle=l_styles[j],
                                    color=ord_cols[i], log_y=logy_ins,
                                    log_x=logx_ins,
                                    legend=False)
    if inset:
        ax_i.set_xticks(x_ins)
        mark_inset(ax, ax_i, loc1=2, loc2=4, fc="none", ec="0.5")
    if vline_pt is not None:
        yax_low, yax_high = ax.get_ylim()
        ax.vlines([vline_pt], yax_low, yax_high, linestyle='dashed',
                  alpha=vert_alpha)
    hands = []
    for i, o in enumerate(ords):
        h = lines.Line2D([], [], color=ord_cols[i], label='O = {}'.format(o))
        hands.append(h)
    for i, rf in enumerate(rfs):
        h = lines.Line2D([], [], color=stylecol, linestyle=l_styles[i],
                         label=r'$\sigma_{rf} = '+r'{}$'.format(rf))
        hands.append(h)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if legend:
        ax.legend(handles=hands, frameon=False)
    return ax

def plot_rf_err(corr, ords, rfs, ax, error_func=gpl.sem, ord_cols=colors,
                logy=False, logx=False, ylab='error rate (PE)',
                xlab=r'receptive field size ($\sigma_{rf}$)'):
    rfs = np.array(rfs)
    for i, o in enumerate(ords):
        gpl.plot_trace_werr(rfs, corr[:, i].T, error_func=error_func, ax=ax,
                            color=ord_cols[i], log_y=logy, log_x=logx)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    return ax

def plot_cumu_errors(corr, ords, rfs, axs, ord_cols=colors, l_styles=None,
                     logy=False, logx=False, start=None, use_err=False,
                     bins=100, xlab='squared error', ax_title=r'$O = {}$',
                     ylab='cumulative density of errors', yticks=None):
    if l_styles is None:
        l_styles = ('solid', 'dashed', 'dotted', 'dashdot')
    for i, o in enumerate(ords):
        ax_i = axs[i]
        for j, rf in enumerate(rfs):
            if not use_err:
                cuse = corr[j, i][corr[j, i] > 0]
            else:
                cuse = corr[j, i]
            gpl.plot_smooth_cumu(cuse, bins=bins, linestyle=l_styles[j],
                                 color=ord_cols[i], log_y=logy, log_x=logx,
                                 normed=True, ax=ax_i, start=start)
        ax_i.set_xlabel(xlab)
        ax_i.set_title(ax_title.format(o))
        if i == 0:
            ax_i.set_ylabel(ylab)
        if i == 0 and yticks is not None:
            ax_i.set_yticks(yticks)
    return axs

def figure3(gen_panels=None, data=None):
    if gen_panels is None:
        gen_panels = ('bc', 'd', 'e')
    setup()
    # whole fig
    pwr_func = nmd.empirical_variance_power
    noise_func = nmd.gaussian_noise

    ni = 10
    c = 3
    ords = range(1, c + 1)
    cc_rfs = [1, 2, 3, 4]
    rf_mask = np.array([True, True, True, False])
    masked_rfs = np.array(cc_rfs)[rf_mask]

    snrs = np.linspace(1, 8, 14)
    nv = 10
    eps = 50 

    var = (snrs**2)*nv*eps + ni**c

    n_samps = 10000
    bs = False
    subdim = True
    dfuncs = [u2.hamming_distortion, u2.mse_distortion]
    bound_distort = ['hamming', 'mse']
    ham_ind = 0
    mse_ind = 1
    l_styles = ('solid', 'dashed', 'dotted', 'dashdot')

    if data is not None:
        print('using supplied data, may be incorrect')
        corr = data['corr']
    else:
        data = {}
        corr = rep_energy_errors_dfunc_rf(c, ni, nv, var, n_samps, cc_rfs,
                                          dfuncs, subdim=subdim, bs=bs, eps=eps,
                                          pwr_func=pwr_func,
                                          noise_func=noise_func)
        data['corr'] = corr
    use_ind = 6
    use_pt = var[use_ind]
        
    # panel B, C
    if 'bc' in gen_panels:
        panel_bc_size = (7, 2.5)
        logy_bc = False
        logx_bc = True
        ins_b = False
        ins_c = True
        
        f_bc = plt.figure(figsize=panel_bc_size)
        ax_ham = f_bc.add_subplot(1, 2, 1)
        ax_mse = f_bc.add_subplot(1, 2, 2)
        plot_rep_energy_ord_rf(corr[ham_ind], ords, masked_rfs, var, nv, ax_ham,
                               vline_pt=use_pt, l_styles=l_styles, logy=logy_bc,
                               logx=logx_bc, legend=False, inset=ins_b)
        plot_rep_energy_ord_rf(corr[mse_ind], ords, masked_rfs, var, nv, ax_mse,
                               vline_pt=use_pt, ylab='MSE', l_styles=l_styles,
                               logy=logy_bc, logx=logx_bc, inset=ins_c)
        fbc_name = basefolder + 'rf-snr.pdf'
        f_bc.savefig(fbc_name, bbox_inches='tight', transparent=True)

    # panel D
    if 'd' in gen_panels:
        panel_d_size = (2, 2.7)
        f_d = plt.figure(figsize=panel_d_size)
        ax_rf_ham = f_d.add_subplot(2, 1, 1)
        ax_rf_mse = f_d.add_subplot(2, 1, 2, sharex=ax_rf_ham)
        plot_rf_err(corr[ham_ind, :, :, use_ind], ords, cc_rfs, ax_rf_ham,
                    xlab='')
        plot_rf_err(corr[mse_ind, :, :, use_ind], ords, cc_rfs, ax_rf_mse,
                    ylab='MSE')
        fd_name = basefolder + 'rf-ham-mse.svg'
        f_d.savefig(fd_name, bbox_inches='tight', transparent=True)


    # panel E
    if 'e' in gen_panels:
        panel_e_size = (6, 2.5)
        yticks_e = [0, .5, 1]
        logx = False
        f_e = plt.figure(figsize=panel_e_size)
        f_e, axs_distr = plt.subplots(1, len(ords), sharex=True, sharey=True,
                                      figsize=panel_e_size)
        plot_cumu_errors(corr[mse_ind, :, :, use_ind], ords, masked_rfs,
                         axs_distr, l_styles=l_styles, yticks=yticks_e,
                         logx=logx)
        f_e.tight_layout()
        fe_name = basefolder + 'rf-cumu.svg'
        f_e.savefig(fe_name, bbox_inches='tight', transparent=True)

    return data
