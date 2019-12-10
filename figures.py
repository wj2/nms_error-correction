
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import matplotlib.colors as mpl_c
import matplotlib.lines as lines
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import matplotlib.colors as mpl_c
import itertools as it
import scipy.spatial as ss
import matplotlib.patches as patches

import mixedselectivity_theory.nms_discrete as nmd
import mixedselectivity_theory.utility as u2
import general.plotting as gpl
import general.plotting_styles as gps
import general.utility as u
import general.neural_analysis as na

basefolder = ('/Users/wjj/Dropbox/research/uc/freedman/analysis/'
              'mixedselectivity_theory/figs/')

colors = np.array([(127,205,187),
                   (65,182,196),
                   (29,145,192),
                   (34,94,168),
                   (37,52,148),
                   (8,29,88)])/256

def setup():
    gps.set_paper_style(colors)

""" figure 1 """
def three_metric_diagram(ax):
    r = 1
    pt_r = .23
    buff = .27
    text_buff = .09
    pt_colors = np.array([(12,192,170), (14,80,62), (143,218,89)])/256
    pt_colors = np.array([(127,201,127), (190,174,212), (253,192,134)])/256
    pwr_color = np.array([100, 100, 100])/256
    pwr_wid = 2
    pwr_alpha = .4
    origin = np.array([0, 0])

    pts = np.array([[0, r], [r,0], [-r/np.sqrt(2), -r/np.sqrt(2)]])
    vor = ss.Voronoi(pts)
    c1 = plt.Circle(origin, r, fill=False, linestyle='dashed', alpha=pwr_alpha, 
                    edgecolor=pwr_color, linewidth=pwr_wid)

    ss.voronoi_plot_2d(vor, ax, show_points=False, show_vertices=False,
                       line_alpha=0)
    ax.add_artist(c1)
    for i, pt in enumerate(pts):
        cp = plt.Circle(pt, pt_r, color=pt_colors[i])
        ax.add_artist(cp)
        ax.text(pt[0], pt[1], r'$t_{O}(x_{' + '{}'.format(i+1) + '})$',
                horizontalalignment='center', verticalalignment='center')
    
    pwr_pt = np.array([-r, r])/np.sqrt(2)
    pwr_line = np.stack((origin, pwr_pt), axis=0)
    pwr_label_offset = -.225
    ax.plot(pwr_line[:, 0], pwr_line[:, 1], linestyle='dashed', alpha=pwr_alpha,
            color=pwr_color, linewidth=pwr_wid)
    gpl.label_line(pwr_pt, origin, 'r', ax, buff=text_buff, 
                   color=pwr_color, alpha=pwr_alpha,
                   lat_offset=pwr_label_offset)
    
    regions, vertices = u.voronoi_finite_polygons_2d(vor)
    # colorize
    for i, region in enumerate(regions):
        polygon = vertices[region]
        plt.fill(*zip(*polygon), alpha=0.4, color=pt_colors[i])
    
    pt_combos = list(it.combinations(range(1, len(pts) + 1), 2))
    ds = np.array([u.euclidean_distance(pts[pt_c[0] - 1], pts[pt_c[1] - 1]) 
                   for pt_c in pt_combos])
    dmin = np.min(ds)
    ds_arr = dmin == ds
    for i, pt_inds in enumerate(pt_combos):
        ptc = [pts[i - 1] for i in pt_inds]                        
        ptc_arr = np.array(ptc)
        ad = np.diff(ptc_arr, axis=0)
        ad_norm = ad/np.sqrt(np.sum(ad**2))
        ptc_arr[0, :] = ptc_arr[0, :] + ad_norm*buff
        ptc_arr[1, :] = ptc_arr[1, :] - ad_norm*buff
        if ds_arr[i]:
            col = 'r'
            linewidth = 2
            str_suff = r' = \Delta$'
        else:
            col = 'k'
            linewidth = 1
            str_suff = '$'
        ax.plot(ptc_arr[:, 0], ptc_arr[:, 1], color=col, linewidth=linewidth)
    
        dstr = r'$d_{'+'{}{}'.format(*pt_inds)+'}' + str_suff
        gpl.label_line(ptc[0], ptc[1], dstr, ax, buff=text_buff, color=col)

    # noise examples
    nv_1 = np.array((.8, -.3))
    nv_2 = np.array((.4, -.95))
    orig_ind = 0
    dec_ind = 1
    origin_pt = pts[orig_ind]
    dec_pt = pts[dec_ind]
    a1_vec = nv_1/np.sqrt(np.sum(nv_1**2))
    a2_vec = nv_2/np.sqrt(np.sum(nv_2**2))
    l1 = .5
    l2 = .6
    arr_wid = .01
    
    a1_end = origin_pt + a1_vec*buff + a1_vec*l1
    a2_end = origin_pt + a2_vec*buff + a2_vec*l2

    gray_noise = np.array((175, 175, 175))/256
    corr_arr_color = gray_noise
    incorr_arr_color = gray_noise

    ax.arrow(origin_pt[0] + a1_vec[0]*buff, origin_pt[1] + a1_vec[1]*buff, 
             a1_vec[0]*l1, a1_vec[1]*l1, length_includes_head=True,
             width=arr_wid, color=corr_arr_color)
    ax.arrow(origin_pt[0] + a2_vec[0]*buff, origin_pt[1] + a2_vec[1]*buff,
             a2_vec[0]*l2, a2_vec[1]*l2, length_includes_head=True,
             width=arr_wid, color=incorr_arr_color)
    noise_arr_buff = .05
    gpl.label_line(origin_pt + a1_vec*buff, a1_end, r'noise', ax, 
                   buff=noise_arr_buff, color=corr_arr_color)
    gpl.label_line(origin_pt + a2_vec*buff, a2_end, r'noise', ax, 
                   buff=noise_arr_buff, color=incorr_arr_color)

    arr_len = 8
    arr2_wid = 4
    style = 'Simple,head_length={},head_width={}'.format(arr_len, arr2_wid)
    
    dec_vec = dec_pt - a2_end
    dv_len = np.sqrt(np.sum(dec_vec**2))
    dv_norm = dec_vec/dv_len
    arr_buff = .01
    error_dec_color = np.array((251,180,174))/256 # 
    incorr = patches.FancyArrowPatch(a2_end + dv_norm*arr_buff,
                                     dec_pt - buff*dv_norm,
                                     arrowstyle=style, color=error_dec_color,
                                     linestyle='dashed')
    ax.add_patch(incorr)
    gpl.label_line(a2_end + dv_norm*arr_buff, dec_pt - buff*dv_norm,
                   'error!', ax, buff=noise_arr_buff, color=error_dec_color)


    corr_vec = origin_pt - a1_end
    corr_buff_vec = buff*np.array([1,1])/np.sqrt(2)
    acorr = patches.FancyArrowPatch(a1_end + np.array([0, arr_buff]), 
                                    origin_pt + corr_buff_vec,
                                    connectionstyle="arc3,rad=.5", 
                                    arrowstyle=style, color=pt_colors[orig_ind],
                                    linestyle='dashed')
    ax.add_patch(acorr)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    return ax

def codes_diagram(fsize_v, fsize_h):
    c = 2
    n_i = 2
    s_conf = (n_i,)*c

    t1, bt1, trs1, cf1 = nmd.generate_types(s_conf, order=1, excl=True)
    t2, bt2, trs2, cf2 = nmd.generate_types(s_conf, order=2, excl=True)

    p_resp = trs1(bt1)/np.sqrt(2)
    m_resp = trs2(bt2)
    
    f_p = plt.figure(figsize=(fsize_h, fsize_v))
    f_m = plt.figure(figsize=(fsize_h, fsize_v))

    cmap = 'Blues'
    vlim = 1
    ticks = (0, np.max(p_resp), np.max(m_resp))
    ticklabels = ('0', 'O = 1', 'O = 2')

    for i in range(p_resp.shape[1]):
        ax_p = f_p.add_subplot(2, 2, i+1, aspect='equal')
        ax_m = f_m.add_subplot(2, 2, i+1, aspect='equal')
        p_ri = p_resp[:, i].reshape((n_i, n_i))
        m_ri = m_resp[:, i].reshape((n_i, n_i))
        ax_range = np.arange(.5, n_i + 1.5, 1)
        ax_p.pcolormesh(ax_range, ax_range, p_ri, cmap=cmap, edgecolor='k', 
                        vmax=vlim)
        p = ax_m.pcolormesh(ax_range, ax_range, m_ri, cmap=cmap, edgecolor='k',
                            vmax=vlim)
        ax_p.set_xticks(range(1, n_i + 1, 1))
        ax_m.set_xticks(range(1, n_i + 1, 1))
        ax_p.set_yticks(range(1, n_i + 1, 1))
        ax_m.set_yticks(range(1, n_i + 1, 1))
        ax_m.set_title('unit {}'.format(i+1))
        ax_p.set_title('unit {}'.format(i+1))
        if i+1 in [2, 4]:
            ax_p.set_yticklabels([])
            ax_m.set_yticklabels([])
        if i+1 in [1, 2]:
            ax_p.set_xticklabels([])
            ax_m.set_xticklabels([])
        if i+1 in [3, 4]:
            ax_p.set_xlabel(r'stimulus feature 1 ($C_{1}$)')
            ax_m.set_xlabel(r'stimulus feature 1 ($C_{1}$)')
        if i+1 in [1, 3]:
            ax_p.set_ylabel(r'stimulus feature 2 ($C_{2}$)')
            ax_m.set_ylabel(r'stimulus feature 2 ($C_{2}$)')

    f_p.suptitle(r'pure selectivity ($O = 1$)')
    f_m.suptitle(r'nonlinear mixed selectivity ($O = 2$)')

    f_cb = plt.figure(figsize=(fsize_h, fsize_v))
    ax_cb = f_cb.add_subplot(1,1,1)
    colbar = f_cb.colorbar(p, shrink=.7)
    colbar.set_ticks(ticks)
    colbar.set_ticklabels(ticklabels)
    colbar.set_label('response magnitude')
    return f_p, f_m, f_cb

def three_d_diagram(fsize):
    plot_axes = (0, 1, 2)

    c = 2
    n_i = 2
    s_conf = (n_i,)*c

    t1, bt1, trs1, cf1 = nmd.generate_types(s_conf, order=1, excl=True)
    t2, bt2, trs2, cf2 = nmd.generate_types(s_conf, order=2, excl=True)

    p_resp = trs1(bt1)/np.sqrt(2)
    m_resp = trs2(bt2)
    
    p_resp[1, 1] = 1/np.sqrt(2)
    p_plot = p_resp[:-1, plot_axes]
    m_plot = m_resp[:-1, plot_axes]
    center = (0, 0, 0)
    
    n = 1000
    xs_circ = np.linspace(0, 1, n)
    ys_circ = np.sqrt(1 - xs_circ**2)
    zs_circ = np.zeros_like(ys_circ)

    f = plt.figure(figsize=fsize)
    ax = f.add_subplot(1,1,1, projection='3d')
    ax.view_init(30, 45)

    circ_color = (.7,)*3
    ms = 10
    origin_alpha = .2

    ax.plot(xs_circ, ys_circ, zs_circ, color=circ_color, linestyle='dashed')
    ax.plot(zs_circ, ys_circ, xs_circ, color=circ_color, linestyle='dashed')
    ax.plot(xs_circ, zs_circ, ys_circ, color=circ_color, linestyle='dashed')
    
    for i, comb in enumerate(it.combinations((0,1,2), 2)):
        p_pair = list(zip(p_plot[comb[0]], p_plot[comb[1]]))
        m_pair = list(zip(m_plot[comb[0]], m_plot[comb[1]]))
        ax.plot(*p_pair, color=colors[0])
        ax.plot(*m_pair, color=colors[1])
        dists = (u.euclidean_distance(p_plot[comb[0]], p_plot[comb[1]]),
                 u.euclidean_distance(m_plot[comb[0]], m_plot[comb[1]]))
    
    for i, stim in enumerate(p_plot):
        p_xyz = list(zip(center, stim))
        m_xyz = list(zip(center, m_plot[i]))
        ax.plot(*p_xyz, color=colors[0], alpha=origin_alpha)
        ax.plot(*m_xyz, color=colors[1], alpha=origin_alpha)
        ax.plot(*map(lambda x: [x], stim), marker='.', color=colors[0], 
                markersize=ms)
        ax.plot(*map(lambda x: [x], m_plot[i]), marker='.', color=colors[1], 
                markersize=ms)
    
    ticks = (0, .5, 1)
    ax.set_xlabel('neuron 1')
    ax.set_xticks(ticks)
    ax.set_ylabel('neuron 2')
    ax.set_yticks(ticks)
    ax.set_zlabel('neuron 3/4')
    ax.set_zticks(ticks)

    h_p = lines.Line2D([], [], color=colors[0], label='O = 1')
    h_m = lines.Line2D([], [], color=colors[1], label='O = 2')
    ax.legend(handles=(h_p, h_m), frameon=False)
    return f, dists

def min_distance_diagram(fsize, dists):
    f = plt.figure(figsize=fsize)
    ax = f.add_subplot(1,1,1)
    mindist_linewidth = 6
    ax.plot((0, dists[0]), (.25, .25),  color=colors[0], linewidth=mindist_linewidth)
    ax.plot((0, dists[1]), (1, 1), color=colors[1], linewidth=mindist_linewidth)
    ax.set_ylim([0, 1.2])
    ax.set_yticks([.25, 1])
    ax.set_yticklabels([r'$\Delta_{1}$', r'$\Delta_{2}$'])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(r'minimum distance ($\Delta$)')
    return f

def metric_plots(ps, delts, terms, orders, fsize):
    f2 = plt.figure(figsize=fsize)
    ax3 = f2.add_subplot(1, 3, 1)
    ax5 = f2.add_subplot(1, 3, 2)
    ax4 = f2.add_subplot(1, 3, 3)
    for i, o in enumerate(orders):
        l = ax3.semilogy([o], terms[i][0], 'o',
                         label=r'$O = {}$'.format(o))
        ax4.plot([o], ps[i][0], 'o', color=l[0].get_color(), 
                 label=r'$O = {}$'.format(o))
        ax5.plot([o], delts[i][0], 'o', label=r'$O = {}$'.format(o),
                 color=l[0].get_color())
    ax3.set_xlabel(r'order ($O$)')
    ax3.set_ylabel(r'pop size ($D$)')
    ax4.set_xlabel(r'order ($O$)')
    ax4.set_ylabel(r'rep energy ($P$)')
    ax5.set_xlabel(r'order ($O$)')
    ax5.set_ylabel(r'min distance ($\Delta$)')
    axs = [ax3, ax4, ax5]
    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(range(1, orders[-1]+1))
    return f2

def ratio_plots(ps, delts, terms, orders, fsize):
    f = plt.figure(figsize=fsize)

    ax1 = f.add_subplot(1, 2, 1)
    ax2 = f.add_subplot(1, 2, 2)

    for i, o in enumerate(orders):
        l = ax1.plot([o], (delts[i][0]**2)/ps[i][0], 'o',
                     label=r'$O = {}$'.format(o))
        ax2.semilogy([o], (ps[i][0])/terms[i][0], 'o', color=l[0].get_color(), 
                     label=r'$O = {}$'.format(o))
    
    ax1.set_xlabel(r'order ($O$)')
    ax1.set_ylabel('min distance$^{2}$ \nper rep energy')
    ax2.set_xlabel(r'order ($O$)')
    ax2.set_ylabel('rep energy \nper pop size')
    axs = [ax1, ax2]
    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(range(1, orders[-1]+1))
    f.tight_layout()
    return f

def figure1(gen_panels=None):
    print('generating Figure 1')
    print('Note: panels A and C were made in latex/inkscape and will not be '
          +'generated')
    if gen_panels is None:
        gen_panels = ('b', 'd', 'e', 'f', 'g')
    setup()
    if 'b' in gen_panels:
        fsize_a = (1.75, 1.75)
        f_a = plt.figure(figsize=fsize_a)
        ax_a = f_a.add_subplot(1,1,1, aspect='equal')
        ax_a = three_metric_diagram(ax_a)
        fname = basefolder + 'code_schematic.svg'
        f_a.savefig(fname, bbox_inches='tight', transparent=True)
    if 'd' in gen_panels:
        fsize_d = (1.5, 2)
        f_p, f_m, f_cb = codes_diagram(*fsize_d)
        f_cb_name = basefolder + 'nms_schem_cb.svg'
        f_cb.savefig(f_cb_name, transparent=True, bbox_inches='tight')
        f_m_name = basefolder + 'nms_schem_mixed.svg'
        f_m.savefig(f_m_name, transparent=True, bbox_inches='tight')
        f_p_name = basefolder + 'nms_schem_pure.svg'
        f_p.savefig(f_p_name, transparent=True, bbox_inches='tight')
    if 'e' in gen_panels:
        fsize_e_3d = (1.5, 1.5)
        fsize_e_bar = (1.25,.2)
        
        f_e_3d, dists = three_d_diagram(fsize_e_3d)
        f_e_bar = min_distance_diagram(fsize_e_bar, dists)
        
        f_ebar_name = basefolder + 'geom_code-bar.svg'
        f_e_bar.savefig(f_ebar_name, transparent=True, bbox_inches='tight')

        f_geom_name = basefolder + 'geom_code.svg'
        f_e_3d.savefig(f_geom_name, transparent=True, bbox_inches='tight')
        
    if 'f' in gen_panels:
        fsize = (3.4, 1.2)
        
        dim = 6
        n = 10
        mults = np.arange(1, 2, .01)
        excl = True
        ps, delts, terms, orders = nmd.probe_orders_analytical(dim, n, mults,
                                                              excl=excl)
        f_f_ratios = ratio_plots(ps, delts, terms, orders, fsize)
        f_f_metric = metric_plots(ps, delts, terms, orders, fsize)
        
        f_ratios_name = basefolder + 'pwr_dim_scaling.svg'
        f_f_ratios.savefig(f_ratios_name, bbox_inches='tight', transparent=True)
        f_metric_name = basefolder + 'pwr_dim_scaling-raw.svg'
        f_f_metric.savefig(f_metric_name, bbox_inches='tight', transparent=True)
            
""" figure 2 """
def rep_energy_errors(c, ords, var, noisevar, n_i, n_samps, pwr_func,
                      noise_func, excl=True, correction=False, only_ana=False,
                      full_series=False, **kwargs):
    emp_corr = np.zeros((len(ords), len(var), n_samps))
    bound_corr = np.zeros((len(ords), len(var)))
    for i, o in enumerate(ords):
        bound_corr[i] = nmd.error_upper_bound_overpwr(c, o, var, noisevar,
                                                      n_i, excl=excl,
                                                      full_series=full_series,
                                                      correction=correction)
        if not only_ana:
            out = nmd.estimate_real_perc_correct_overpwr(c, o, n_i, noisevar, 
                                                         var, n_samps=n_samps,
                                                         pwr_func=pwr_func,
                                                         excl=excl,
                                                         noise_func=noise_func,
                                                         **kwargs)
        else:
            out = None
        emp_corr[i] = out
    return emp_corr, bound_corr

def plot_rep_energy_ord(emp_corr, bound_corr, ords, var, noisevar, ax,
                        x_ins=(5,6), y_ins=(.0001, .12), xlab='SNR',
                        log_x = False, ylab='error rate (PE)', ylim=(0, 1),
                        inset=True, plot_bound_main=False, ins_logy=True,
                        bound_edge_color='k', bound_edge_wid=2, snr_start=None):
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
    if snr_start is not None:
        ax.set_xlim([snr_start, snrs[-1]])
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    if inset:
        ax_i.set_xticks(x_ins)
        mark_inset(ax, ax_i, loc1=2, loc2=4, fc="none", ec="0.5")
    ax.legend(frameon=False)
    return ax

def plot_err_at_snr(nc_s, n_i, snr, pwr_func, ax1, ax2, text_buff=.07,
                    excl=True, logy=True, ord_cols=colors, correction=False,
                    label_pts=True, full_series=False):
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
                                                      correction=correction,
                                                      full_series=full_series,
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
        if label_pts:
            ax2.text(xtxt, ytxt + text_buff, 'O = {}'.format(best_ord),
                     color=colors[i], horizontalalignment='center',
                     transform=ax2.transAxes)
    ax1.legend(frameon=False)
    ax1.set_xlabel('order (O)')
    ax1.set_ylabel('error rate\nSNR = {}'.format(snr))
    ax2.set_ylabel('error ratio\npure/mixed')
    ax2.set_xticks(nc_s)
    ax2.set_xlabel(r'number of features ($K$)')
    return ax1, ax2

def plot_target_error(nc_s, n_i, target_err, pwr_func, ax1, ax2, text_buff=15,
                      excl=True, subdim=False, eps=1, logy=False, rf=1,
                      distortion='hamming', nv=10, correction=False,
                      full_series=False):

    kwargs = {'subdim':subdim, 'pwr_func':pwr_func, 'eps':eps, 'rf':rf, 
              'distortion':distortion}
    if logy:
        ax1.set_yscale('log')
    for i, nc in enumerate(nc_s):
        os = np.arange(1, nc + 1)
        snrs = np.array(list([nmd.pwr_require_err(target_err, nc, o, n_i,
                                                  correction=correction,
                                                  full_series=full_series,
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
    te_perc = target_err*100
    ax1.set_ylabel('represenation energy for\n{}% error rate'.format(te_perc))
    
    ax2.set_ylabel('% representation energy for pure\nrelative to best mixed code')
    ax2.set_yticks([100, 200, 300])
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
                   xlabel=('total available energy\n'
                           +r'($E = \epsilon V + D_{O}$)'),
                   ylabel=r'number of values ($n$)', box_bound=(10**5, 10**6)):
    colors_map = [(1, 1, 1)] + list(colors)
    labels = np.arange(-1, c + 1) + .5
    cmap, norm = mpl_c.from_levels_and_colors(labels, colors_map)

    best_ord = np.arange(1, c+1)[np.argmax(ds, axis=0)]
    invalid = np.all(ds < 0, axis=0)
    best_ord[invalid] = 0

    es_plot = gpl.pcolormesh_axes(es, best_ord.shape[1])
    n_is_plot = gpl.pcolormesh_axes(n_is, best_ord.shape[0])

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
    ax.set_xlim((es[0], es[-1]))
    ax.set_ylim((n_is[0], n_is[-1]))
    return ax, p
    
def figure2(gen_panels=None, data=None, redo_ana=False):
    if gen_panels is None:
        gen_panels = ('a', 'b', 'c', 'd', 'sab')
    print('generating Figure 2')
    setup()
    # whole fig
    n_samps = 10000
    pwr_func = nmd.empirical_variance_power
    noise_func = nmd.gaussian_noise
    nv = 10
    var = np.linspace(.1, 1000, 300)
    correction = True
    full_series = True
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
        snr_start_a = 2
        panel_a_size = (1.75, 1)
        f_a = plt.figure(figsize=panel_a_size)
        ax_a = f_a.add_subplot(1,1,1)
        if 'ec_a' in data.keys() and not redo_ana:
            ec_a = data['ec_a']
            bc_a = data['bc_a']
        elif 'ec_a' in data.keys() and redo_ana:
            out = rep_energy_errors(c_a, ords_a, var, nv, ni_a, n_samps,
                                    pwr_func, noise_func, excl=True,
                                    correction=correction, only_ana=True,
                                    full_series=full_series)
            _, bc_a = out
            data['bc_a'] = bc_a
            ec_a = data['ec_a']
        else:
            out = rep_energy_errors(c_a, ords_a, var, nv, ni_a, n_samps,
                                    pwr_func, noise_func, excl=True,
                                    correction=correction,
                                    full_series=full_series)
            ec_a, bc_a = out
            data['ec_a'] = ec_a
            data['bc_a'] = bc_a
        ax_a = plot_rep_energy_ord(ec_a, bc_a, ords_a, var, nv, ax_a,
                                   x_ins=insetx_a, y_ins=insety_a,
                                   snr_start=snr_start_a)
        fa_name = basefolder + 'e-snr-k{}.pdf'.format(c_a)
        f_a.savefig(fa_name, bbox_inches='tight', transparent=True)

    # panel B
    if 'b' in gen_panels:
        c_b = 5
        ords_b = range(1, c_b + 1)
        ni_b = 3
        insetx_b = (6, 7)
        insety_b = (.00005, .1)
        snr_start_b = 2
        
        panel_b_size = panel_a_size 
        f_b = plt.figure(figsize=panel_b_size)
        ax_b = f_b.add_subplot(1,1,1)
        if 'ec_b' in data.keys() and not redo_ana:
            ec_b = data['ec_b']
            bc_b = data['bc_b']
        elif 'ec_b' in data.keys() and redo_ana:
            out = rep_energy_errors(c_b, ords_b, var, nv, ni_b, n_samps,
                                    pwr_func, noise_func, excl=True,
                                    correction=correction, only_ana=True,
                                    full_series=full_series)
            _, bc_b = out
            data['bc_b'] = bc_b
            ec_b = data['ec_b']
        else:
            out = rep_energy_errors(c_b, ords_b, var, nv, ni_b, n_samps,
                                    pwr_func, noise_func, excl=True,
                                    correction=correction,
                                    full_series=full_series)
            ec_b, bc_b = out
            data['ec_b'] = ec_b
            data['bc_b'] = bc_b
        ax_b = plot_rep_energy_ord(ec_b, bc_b, ords_b, var, nv, ax_b,
                                   x_ins=insetx_b, y_ins=insety_b,
                                   snr_start=snr_start_b)
        fb_name = basefolder + 'e-snr-k{}.pdf'.format(c_b)
        f_b.savefig(fb_name, bbox_inches='tight', transparent=True)

    # panel C
    if 'c' in gen_panels:
        cs_c = range(2, 7)
        ni_c = 5
        snr = 9
        pwr_func_c = nmd.analytical_code_variance
        
        panel_c_size = (1.5, panel_a_size[1]*2)
        f_c = plt.figure(figsize=panel_c_size)
        ax1, ax2 = f_c.add_subplot(2, 1, 1), f_c.add_subplot(2, 1, 2)
    
        ax1, ax2 = plot_err_at_snr(cs_c, ni_c, snr, pwr_func_c,
                                   ax1, ax2, correction=correction,
                                   label_pts=False,
                                   full_series=full_series)
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
        panel_d_size = (2.5*panel_c_size[0], 1.25*panel_a_size[1])
        
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
        f_d.savefig(fd_name, bbox_inches='tight', transparent=True, dpi=600)

    # panel supp AB
    if 'sab' in gen_panels:
        cs_sab = range(2, 7)
        ni_sab = 5
        targ_err = .001
        pwr_func_sab = nmd.analytical_code_variance
        
        panel_sab_size = (3.2, 1.4)
        f_sab = plt.figure(figsize=panel_sab_size)
        ax1, ax2 = f_sab.add_subplot(1, 2, 1), f_sab.add_subplot(1, 2, 2)
    
        ax1, ax2 = plot_target_error(cs_sab, ni_sab, targ_err, pwr_func_sab,
                                     ax1, ax2, correction=correction,
                                     full_series=full_series)
        fsab_name = basefolder + 'energy-req_l2.svg'
        f_sab.savefig(fsab_name, bbox_inches='tight', transparent=True)

    return data

""" figure 3 """
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
                     ylab='cumulative density\nof errors', yticks=None):
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
    print('generating Figure 3')
    print('Note: panel A was made in latex/inkscape and will not be generated')
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
        panel_bc_size = (3.5, 1)
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
        panel_d_size = (1.4, 1.8)
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
        panel_e_size = (3.2, 1)
        yticks_e = [0, .5, 1]
        logx = False
        f_e = plt.figure(figsize=panel_e_size)
        f_e, axs_distr = plt.subplots(1, len(ords), sharex=True, sharey=True,
                                      figsize=panel_e_size)
        plot_cumu_errors(corr[mse_ind, :, :, use_ind], ords, masked_rfs,
                         axs_distr, l_styles=l_styles, yticks=yticks_e,
                         logx=logx)
        fe_name = basefolder + 'rf-cumu.svg'
        f_e.savefig(fe_name, bbox_inches='tight', transparent=True)

    return data

""" figure 4 """

def single_feature_decoder(times_samp=10000, c=2, n=2, excl=True, snrs=None,
                           bs_samps=1000):
    if snrs is None:
        snrs = np.linspace(2, 5, 15)
    os = range(1, c + 1)
    nv = 1
    sigs = nv*snrs**2

    decs = []
    feat_decs = []
    for j, o in enumerate(os):
        feat_dec = np.zeros((len(sigs), bs_samps))
        dec = np.zeros((len(sigs), bs_samps))
        for i, s in enumerate(sigs):
            out = nmd.simulate_transform_code_out(c, o, n, nv, s,
                                                 times_samp=times_samp,
                                                 excl=excl)
            owords, dec_words, ns, corr, bt, words, trs, sel = out
            nonlin_dec = dec_words == owords
            nonlin_dec = np.all(nonlin_dec[:, :n], axis=1)
            feat_dec[i] = u.bootstrap_list(nonlin_dec.flatten(), 
                                                 np.mean, bs_samps)
            dec[i] = u.bootstrap_list(np.logical_not(corr).flatten(),
                                            np.mean, bs_samps)
        feat_decs.append(feat_dec)
        decs.append(dec)
    return snrs, feat_decs, decs, os

def plot_single_feature(snrs, decs, os, fsize):
    f = plt.figure(figsize=fsize)
    ax = f.add_subplot(1,1,1)
    for i, d in enumerate(decs):
        gpl.plot_trace_werr(snrs, 1 - d.T, ax=ax, log_y = True,
                            error_func=gpl.conf95_interval,
                            label='O = {}'.format(os[i]))

    ax.set_xticks([2, 3, 4, 5])
    ax.set_xlabel('SNR')
    ax.set_ylabel('error rate (PE)')
    return f

def organize_data(d, window_size=150, window_step=20):
    sacc_latt = 100
    beg = 0
    pre_fix_time = 500
    sample_time = 650
    presacc_time = 300
    saccade_time = 250
    delay_nosacc = 1500
    delay_postsacc = 1000
    test_time = 650

    binsize = 150
    binstep = 20

    stradsacc_trial_struct = {'transitions':[beg - delay_postsacc, ],
                              'labels':['saccade']}

    disc_fields = ('sample_category', 'saccade', 'saccade_intoRF',
                   'saccade_towardRF')
    funcs = [np.equal]*len(disc_fields)

    cat1_svals = (1, True, False, True)
    cat1_vals = (1, True, False, False)
    cat2_svals = (2, True, False, True)
    cat2_vals = (2, True, False, False)

    cat1_saccfunc = u.make_trial_constraint_func(disc_fields, cat1_svals, 
                                                 funcs)
    cat2_saccfunc = u.make_trial_constraint_func(disc_fields, cat2_svals,
                                                 funcs)
    cat1_nosaccfunc = u.make_trial_constraint_func(disc_fields, cat1_vals, 
                                                   funcs)
    cat2_nosaccfunc = u.make_trial_constraint_func(disc_fields, cat2_vals,
                                                   funcs)
    markerfunc_test1 = u.make_time_field_func('test1_on')
    
    posttime_sacc = sample_time + presacc_time

    post_pretime_sacc = beg - delay_postsacc - presacc_time - sacc_latt
    post_posttime_sacc = beg

    discrims = (cat1_saccfunc, cat2_saccfunc, cat1_nosaccfunc, cat2_nosaccfunc)
    markers = (markerfunc_test1, markerfunc_test1, markerfunc_test1, markerfunc_test1)
    ns_spost, xs_spost = na.organize_spiking_data(d, discrims, markers,
                                                  post_pretime_sacc,
                                                  post_posttime_sacc,
                                                  binsize, binstep)
    xs_spost_mod = xs_spost + delay_postsacc
    return xs_spost_mod, ns_spost

def format_data(ns, xs, window_center=95, req_trials=15):
    c1_s, c2_s, c1_ns, c2_ns = ns
    pre_arr_spost = [[c1_s, c2_s], [c1_ns, c2_ns]]
    sacc_cat_arr = na.array_format(pre_arr_spost, req_trials)
    time_component = np.mean(sacc_cat_arr, axis=(0, 3, 4), keepdims=True)
    sacc_cat_arr_detimed = sacc_cat_arr - time_component

    closest_xs = xs[np.argmin(np.abs(xs - window_center))]
    xmask = xs == closest_xs
    nms_dat, nms_conds, _ = na.condition_mask(sacc_cat_arr,
                                              cond_labels=['sacc', 'cat'], 
                                              interactions=((0,1),),
                                              double_factors=(False, False))

    nms_conds = nms_conds.astype(bool)
    cond_mask = nms_conds[0].sum(axis=0) > 0
    nms_conds = nms_conds[:, :, cond_mask]
    nms_dat = nms_dat[xmask]
    return nms_dat, nms_conds

def fit_saccdmc_glms(paths, n_perms=5000, window_size=150, window_center=95,
                     req_trials=15, demean=True, zscore=True):
    pattern = 'rcl[mj][0-9]*.mat'
    d = u.load_saccdmc(paths, pattern)
    xs, ns = organize_data(d, window_size)
    closest_xs = xs[np.argmin(np.abs(xs - window_center))]
    xmask = xs == closest_xs
    ind_structure = [(0, 0), (0, 1), (1, 0), (1, 1)]
    out = na.glm_fitting_diff_trials(ns, ind_structure, req_trials=req_trials,
                                     cond_labels=('sacc','cat'),
                                     interactions=((0, 1),),
                                     double_factors=(False, False),
                                     perms=n_perms, demean=demean,
                                     zscore=zscore, xs_mask=xmask)
    cos, pcos, _ = out
    return cos, pcos

def figure4(gen_panels=None, data=None, data_paths=None):
    print('generating Figure 4')
    print('Note: panels C and D rely on previously published data, available '
          +'by contacting the original authors of Rishel, Huang, Freedman '
          +'(2013).')
    if data is None:
        data = {}
    else:
        print('using supplied data, may be incorrect')
    if gen_panels is None:
        gen_panels = ('c', 'd', 'e')
    setup()

    demean = True
    zscore = True
    if ('c' in gen_panels or 'd' in gen_panels) and 'cd' not in data.keys():
        n_perms = 5000
        window_size = 200
        window_center = 95 # ms post saccade cue
        out = fit_saccdmc_glms(data_paths, n_perms, window_size, window_center,
                               demean=demean, zscore=zscore)
        data['cd'] = out
    if 'c' in gen_panels or 'd' in gen_panels:
        rcs, rps = data['cd']
    eps = .001
    p_thr = .05/8
    t_pt = 0
    if demean:
        ps_group = (0, 1, 2, 3)
        nms_group = (4,5,6,7)
    else:
        ps_group = (1, 2, 3, 4)
        nms_group = (5, 6, 7, 8)
    ps_factor_labels = ('S1', 'S2', 'C1', 'C2')
    nms_factor_labels = ('S1-C1', 'S1-C2', 'S2-C1', 'S2-C2')
    gtls = (ps_factor_labels, nms_factor_labels)
    groups = (ps_group, nms_group)
    group_xlabels = ('O = 1', 'O = 2')
    ylabel_heat = 'neuron number'
    ylabel_prop = '% selective'
    ylabel_mag = 'strength\n(|z-score|)'
    cb_label = 'z-score'
    label_rotation = 'vertical'
    if 'c' in gen_panels:
        fsize_heat = (1.4, 3.8)
        sep_cb = True
        out = gpl.plot_glm_indiv_selectivity(rcs[:, t_pt], rps[:, t_pt], groups,
                                     group_xlabels=group_xlabels, p_thr=p_thr,
                                     ylabel=ylabel_heat, figsize=fsize_heat,
                                     group_term_labels=gtls, 
                                     cb_label=cb_label, sep_cb=sep_cb, 
                                     label_rotation=label_rotation)
        f_heat, f_heat_cb = out
        f_heat_name = basefolder + 'nms_rishel-heat.svg'
        f_heat.savefig(f_heat_name, bbox_inches='tight', transparent=True)
        f_heat_cb_name = basefolder + 'nms_rishel-heat_cb.svg'
        f_heat_cb.savefig(f_heat_cb_name, bbox_inches='tight', transparent=True)
    if 'd' in gen_panels:
        fsize_prop = (1.4, 2)
        fsize_mag = (1.4, 2)
        comb_fig = True
        out_mag = gpl.plot_glm_pop_selectivity_mag(rcs[:, t_pt], rps[:, t_pt],
                                                   groups, colors=colors,
                                                   group_xlabels=group_xlabels,
                                                   ylabel=ylabel_mag, eps=eps,
                                                   p_thr=p_thr,
                                                   group_term_labels=gtls,
                                                   figsize=fsize_mag,
                                                   combined_fig=comb_fig,
                                                   label_rotation=label_rotation,
                                                   group_test=True)
        f_mag, f_ps = out_mag
        f_prop = gpl.plot_glm_pop_selectivity_prop(rcs[:, t_pt], rps[:, t_pt],
                                                   groups, colors=colors,
                                                   group_xlabels=group_xlabels,
                                                   ylabel=ylabel_prop, eps=eps,
                                                   p_thr=p_thr,
                                                   group_term_labels=gtls,
                                                   figsize=fsize_prop,
                                                   fig=f_mag,
                                                   label_rotation=label_rotation)
        f_prop_name = basefolder + 'nms_rishel-prop.svg'
        f_prop.savefig(f_prop_name, bbox_inches='tight', transparent=True)
        f_mag_name = basefolder + 'nms_rishel-mag.svg'
        f_mag.savefig(f_mag_name, bbox_inches='tight', transparent=True)
    if 'c' in gen_panels or 'd' in gen_panels:
        print('Units and number of conditions:')
        print(rps[:, t_pt].shape)
        print('Any significant selectivity:')
        print(np.sum(np.any(rps[:, t_pt] < p_thr, axis=1)),
              '/', rps.shape[0])
        print('Significant pure selectivity:')
        print(np.sum(np.any(rps[:, t_pt, ps_group] < p_thr, axis=1)),
              '/', rps.shape[0])
        print('Significant mixed selectivity:')
        print(np.sum(np.any(rps[:, t_pt, nms_group] < p_thr, axis=1)),
              '/', rps.shape[0])
    if 'e' in gen_panels:
        fsize_e = (1.4, 1)
        if 'e' not in data.keys():
            c = 2
            n = 2
            n_samps = 100000
            out = single_feature_decoder(n_samps, c, n)
            data['e'] = out
        snrs, feats, decs, orders = data['e']
        f_e = plot_single_feature(snrs, feats, orders, fsize_e)
        fname = basefolder + 'rishel_decoding.svg'
        f_e.savefig(fname, bbox_inches='tight', transparent=True)
    return data
        
