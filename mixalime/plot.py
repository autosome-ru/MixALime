#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
from betanegbinfit import ModelMixture
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from .utils import openers, get_init_file, get_model_creator, dictify_params, scorefiles_qc
from betanegbinfit.utils import get_params_at_slice
from scipy.interpolate import UnivariateSpline
import dill
import os

_fontsize = 16
_ref = '#DC267F'
_alt = '#FFB000'
_count = '#648FFF'
_cmap = LinearSegmentedColormap.from_list("", ['white', _count])
_markersize = 8

def update_style():
    file_path = os.path.split(os.path.realpath(__file__))[0]
    font_files = font_manager.findSystemFonts(fontpaths=os.path.join(file_path, 'data'))
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
    plt.rcParams['font.weight'] = "medium"
    plt.rcParams['axes.labelweight'] = 'medium'
    plt.rcParams['figure.titleweight'] = 'medium'
    plt.rcParams['axes.titleweight'] = 'medium'
    plt.rcParams['font.family'] = 'Lato'
    plt.rcParams['font.size'] = _fontsize

update_style()



def plot_heatmap(counts: np.ndarray, max_count: int, slices=None, shift=10, cmap=_cmap):
    hm = np.ones((max_count + shift , max_count + shift))
    counts = counts[(counts[:, 0] < max_count + shift) & (counts[:, 1] < max_count + shift)]
    hm[counts[:, 0], counts[:, 1]] += counts[:, 2]
    max_order = int(np.ceil(np.log10(counts[:, 2].max() + 1 )))
    hm = np.log10(hm)

    plt.imshow(hm, cmap=cmap, vmin=0, vmax=max_order)
    if slices:
        try:
            a, b = slices
            plt.vlines(a, 0, max_count, colors=_alt, linestyles='dashed', linewidth=3)
            plt.hlines(b, 0, max_count, colors=_ref, linestyles='dashed', linewidth=3)
        except TypeError:
            plt.plot([0, slices], [slices, 0], color='k', linestyle='dashed', linewidth=3)
    plt.xlim(0, max_count)
    plt.ylim(0, max_count)
    plt.xlabel('Reference allele read count')
    plt.ylabel('Alternative allele read count')
    cbar = plt.colorbar(fraction=0.046, pad=0.02,)
    ticks = list(); tick_labels = list()
    for i in range(max_order + 1):
        ticks.append(i)
        tick_labels.append(f'$10^{i}$')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    return hm, counts

def get_pdf_computer(m: ModelMixture, params: dict):
    return lambda x, y: np.exp(m.logprob(m.dict_to_vec(get_params_at_slice(params, y)), x))

    
def plot_histogram(counts: np.ndarray, max_count: int, slc: int, pdf_computer, s=0, c='r', slc_sum=False):
    counts = counts[counts[:, s] < max_count, :]
    if slc_sum:
        counts = counts[counts[:, 1 - s] + counts[:, s] == slc][:, [s, 2]]
    else:
        counts = counts[counts[:, 1 - s] == slc][:, [s, 2]]
    plt.bar(counts[:, 0], counts[:, 1] / counts[:, 1].sum(), width=1, color=_count)
    x = np.arange(0, max_count)
    y = pdf_computer(x, slc)
    if s:
        plt.xlabel('Alternative allele read counts')
    else:
        plt.xlabel('Reference allele read counts')
    ax = plt.plot(x, y, color=c, linewidth=2)[0].axes
    return ax

def sliceplot(counts: np.ndarray, max_count:int, ref: int, alt: int, m: ModelMixture, params_ref: dict,
              params_alt: dict, figsize=(20, 6), dpi=200):
    params_ref = dictify_params(params_ref)
    params_alt = dictify_params(params_alt)
    pdf_ref = get_pdf_computer(m, params_ref)
    pdf_alt = get_pdf_computer(m, params_alt)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.subplot(1, 3, 1)
    plot_heatmap(counts, max_count, (ref, alt))
    plt.subplot(1, 3, 2)
    ax1 = plot_histogram(counts, max_count, ref, pdf_ref, c=_ref)
    ylim1 = ax1.get_ylim()[1]
    plt.subplot(1, 3, 3)
    ax2 = plot_histogram(counts, max_count, alt, pdf_alt, 1, c=_alt)
    ylim2 = ax2.get_ylim()[1]
    ylim = max(ylim1, ylim2)
    ax1.set_ylim(0, ylim)
    ax2.set_ylim(0, ylim)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

def plot_gof(stats_ref: dict, stats_alt: dict, max_count: int, figsize=(6, 6), dpi=200, spline=False):
    x = set(stats_ref.keys()) & set(stats_alt.keys())
    x = np.array(list(filter(lambda x: x < max_count, sorted(x))))
    stats_ref = np.array([stats_ref[k]['rmsea'] for k in x])
    stats_alt = np.array([stats_alt[k]['rmsea'] for k in x])
    
    # ticks = np.arange(0, max(max(stats_ref), max(stats_alt)), 0.05)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(x, stats_ref, 'o', color=_ref, markersize=_markersize * 1.1)
    plt.plot(x, stats_alt, 'o', color=_alt, markersize=_markersize, alpha=0.85)
    if spline:
        nx = np.linspace(x[0], max_count, max_count * 10)
        inds = ~np.isnan(stats_ref)
        spline = UnivariateSpline(x[inds], stats_ref[inds], s=0.01, k=4)
        plt.plot(nx, spline(nx), 'b--')
        inds = ~np.isnan(stats_alt)
        spline = UnivariateSpline(x[inds], stats_alt[inds], s=0.01, k=4)
        plt.plot(nx, spline(nx), 'y--')
    # plt.yticks(ticks)
    plt.axhline(0.05, color='k', linestyle='dashed')
    plt.grid(True)
    plt.legend(['ref', 'alt'])
    plt.xlabel('Read count for the fixed allele')
    plt.ylabel('Goodness of fit, RMSEA')
    
def plot_stat(stats_ref: dict, stats_alt: dict, max_count: int, stat: str, figsize=(6, 6), dpi=200, spline=False, log=False):
    x = set(stats_ref.keys()) & set(stats_alt.keys())
    x = np.array(list(filter(lambda x: x < max_count, sorted(x))))
    stats_ref = np.array([stats_ref[k][stat] for k in x])
    stats_alt = np.array([stats_alt[k][stat] for k in x])
    if log:
        stats_ref = np.log10(stats_ref)
        stats_alt = np.log10(stats_alt)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(x, stats_ref, 'o', color=_ref, markersize=_markersize * 1.1)
    plt.plot(x, stats_alt, 'o', color=_alt, markersize=_markersize, alpha=0.85)
    if spline:
        nx = np.linspace(x[0], max_count, max_count * 10)
        inds = ~np.isnan(stats_ref)
        spline = UnivariateSpline(x[inds], stats_ref[inds], s=0.001, k=4)
        plt.plot(nx, spline(nx), 'b--')
        inds = ~np.isnan(stats_alt)
        spline = UnivariateSpline(x[inds], stats_alt[inds], s=0.001, k=4)
        plt.plot(nx, spline(nx), 'y--')
    plt.grid(True)
    plt.legend(['ref', 'alt'])
    plt.xlabel('Read count for the fixed allele')
    plt.ylabel(r'$log_{10}$' + stat if log else stat)

def plot_scorefiles_qc(covers: dict, biases: dict, scorefiles: list, bad=None, rule=3.0,
                       figsize=(6, 6), dpi=200,):
    common_prefix = 0
    common_postfix = 0
    for i in range(len(min(scorefiles, key=len))):
        c = scorefiles[0][i]
        common = True
        for f in scorefiles[1:]:
            if c != f[i]:
                common = False
                break
        if not common:
            break
        common_prefix += 1
    for i in range(1, len(min(scorefiles, key=len))):
        c = scorefiles[0][-i]
        common = True
        for f in scorefiles[1:]:
            if c != f[-i]:
                common = False
                break
        if not common:
            break
        common_postfix -= 1
    if not common_postfix:
        common_postfix = None
    scorefiles = [f[common_prefix:common_postfix] for f in scorefiles]
    labels = list()
    x = list()
    y = list()
    labels = list()
    for ind in sorted(covers):
        cover = covers[ind][bad]
        if cover:
            x.append(cover)
            y.append(biases[ind][bad])
            labels.append(scorefiles[ind])
    x = np.log10(x)
    y = np.array(y)
    labels = np.array(labels)
    inds = np.abs(y - np.mean(y)) > rule * np.std(y)
    plt.figure(figsize=(6, 6), dpi=200)
    plt.scatter(x, y, s=15, c=-np.array(y))
    for xc, yc, label in zip(x[inds], y[inds], labels[inds]):
        plt.annotate(label, (xc, yc), fontsize=8)
    plt.xlabel(r'$log_{10}(\mathrm{coverage})$')
    plt.ylabel(r'Fraction of SNVs when $ref > alt$')
    plt.grid(True)

def plot_params(params_ref: dict, params_alt: dict, max_count: int, param: str,
                figsize=(6, 6), dpi=200, inv=False,  diag=False, name=None, spline=False,
                hor_expected=None, std=True):
    x = np.arange(0, max_count)
    if std:
        try:
            stds_ref = dictify_params(params_ref, 'stds')
            stds_alt = dictify_params(params_alt, 'stds')
            sref = np.array([get_params_at_slice(stds_ref, i, clip_at_max_slice=False, nan_min=True, std=True).get(param, np.nan) for i in x]) 
            salt = np.array([get_params_at_slice(stds_alt, i, clip_at_max_slice=False, nan_min=True, std=True).get(param, np.nan) for i in x])
        except KeyError:
            std = False
    params_ref = dictify_params(params_ref)
    params_alt = dictify_params(params_alt)
    if type(param) is not str:
        if len(param) == 2:
            pref1 = np.array([get_params_at_slice(params_ref, i, clip_at_max_slice=False, nan_min=True).get(param[0], np.nan) for i in x])
            palt1 = np.array([get_params_at_slice(params_alt, i, clip_at_max_slice=False, nan_min=True).get(param[0], np.nan) for i in x])
            pref2 = np.array([get_params_at_slice(params_ref, i, clip_at_max_slice=False, nan_min=True).get(param[1], np.nan) for i in x])
            palt2 = np.array([get_params_at_slice(params_alt, i, clip_at_max_slice=False, nan_min=True).get(param[1], np.nan) for i in x])
            pref = pref1 + pref2 
            palt = palt1 + palt2
        else:
            pref1 = np.array([get_params_at_slice(params_ref, i, clip_at_max_slice=False, nan_min=True).get(param[0], np.nan) for i in x])
            palt1 = np.array([get_params_at_slice(params_alt, i, clip_at_max_slice=False, nan_min=True).get(param[0], np.nan) for i in x])
            pref2 = np.array([get_params_at_slice(params_ref, i, clip_at_max_slice=False, nan_min=True).get(param[1], np.nan) for i in x])
            palt2 = np.array([get_params_at_slice(params_alt, i, clip_at_max_slice=False, nan_min=True).get(param[1], np.nan) for i in x])
            pref3 = np.array([get_params_at_slice(params_ref, i, clip_at_max_slice=False, nan_min=True).get(param[2], np.nan) for i in x])
            palt3 = np.array([get_params_at_slice(params_alt, i, clip_at_max_slice=False, nan_min=True).get(param[2], np.nan) for i in x])
            pref = pref3 / (pref1 + pref2)
            palt = palt3 / (palt1 + palt2)
    else:
        pref = np.array([get_params_at_slice(params_ref, i, clip_at_max_slice=False, nan_min=True).get(param, np.nan) for i in x])
        palt = np.array([get_params_at_slice(params_alt, i, clip_at_max_slice=False, nan_min=True).get(param, np.nan) for i in x])
    if inv:
        pref = 1 / pref
        palt = 1 / palt
        if std:
            sref *= pref ** 2
            salt *= palt ** 2
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(x, pref, 'o', color=_ref, markersize=_markersize * 1.1)
    plt.plot(x, palt, 'o', color=_alt, markersize=_markersize, alpha=0.85)
    if spline:
        nx = np.linspace(0, max_count, max_count * 10)
        inds = ~np.isnan(pref)
        spline = UnivariateSpline(x[inds], pref[inds], s=0.001, k=4)
        plt.plot(nx, spline(nx), 'b--')
        inds = ~np.isnan(palt)
        spline = UnivariateSpline(x[inds], palt[inds], s=0.001, k=4)
        plt.plot(nx, spline(nx), 'y--')
    if std:
        sref *= 1.5
        salt *= 1.5
        plt.fill_between(x, pref - sref, pref + sref, alpha=0.2, color=_ref)
        plt.fill_between(x, palt - salt, palt + salt, alpha=0.2, color=_alt)
    if diag:
        plt.axline((1, 1), slope=1, linestyle='dashed', color='k')
    if hor_expected is not None:
        plt.axhline(hor_expected, linestyle='dashed', color='k')
    plt.grid(True)
    plt.legend(['ref', 'alt'])
    plt.xlabel('Read count for the fixed allele')
    plt.ylabel(name if name else param)

def plot_anova_snvs(name: str, snv_names=None, snvs=None, subname=None, plot_raw_es=True, plot_test_es=True,
                    plot_p_diff=True, color_significant=True, folder=str(), ext='png', 
                    figsize=(12, 4), dpi=200):
    
    def plot_barplot(groups, values, n_rows, i, variance=None, legend_colors=False):
        plt.subplot(n_rows, 1, i)
        plt.bar(groups, values, color='grey', width=0.9)
        fdr_bar = None
        pval_bar = None
        if color_significant:
            for name, pval, pval_fdr, v in zip(groups, pvals, pvals_fdr, values):
                if pval < 0.05:
                    pval_bar_ = plt.bar(name, v, width=0.9, color='orange')
                    if pval_fdr < 0.05:
                        fdr_bar = plt.bar(name, v, width=0.9, color='green')
                    else:
                        pval_bar = pval_bar_
        if variance is not None:
            plt.errorbar(groups, values, np.array(variance) ** 0.5,  fmt='.', color='Black', elinewidth=2, capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)
        if legend_colors:
            handles = list()
            labels = list()
            if fdr_bar is not None:
                labels.append('FDR < 0.05')
                handles.append(fdr_bar)
            if pval_bar is not None:
                labels.append('P-value < 0.05')
                handles.append(pval_bar)
            if handles:
                plt.legend(handles, labels)
        plt.grid(True, which='both')
        plt.margins(x=0)
        plt.gca().set_xticklabels([])
        
    if folder:
        os.makedirs(folder, exist_ok=True)
    if snvs is None:
        snvs = set()
    else:
        snvs = set(snvs)
    if snv_names is None:
        snv_names = set()
    filename = get_init_file(name)
    compressor = filename.split('.')[-1]
    open = openers[compressor]
    filename = f'{name}.anova.{compressor}'
    with open(filename, 'rb') as f:
        anova = dill.load(f)[subname]
        snvs_db = anova['snvs']
        anova = anova['tests']
    
    if plot_test_es:
        filename = f'{name}.test.{compressor}'
        with open(filename, 'rb') as f:
            test = dill.load(f)
    names_dict = dict()
    for pos in snvs:
        for g in snvs_db:
            if pos in g:
                names_dict[pos] = g[pos][0][0]
                break
    for snv_name in snv_names:
        stop = False
        snv_name, allele = snv_name
        for g in snvs_db:
            for pos, lt in g.items():
                if lt and (lt[0][0] == snv_name) and (pos[-1] == allele):
                    snvs.add(pos)
                    names_dict[pos] = snv_name
                    stop = True
                    break
            if stop:
                break
    
    for ind in snvs:
        snv_name = names_dict.get(ind, None)
        for  allele in ('ref', 'alt'):
            try:
                snv = anova[anova.ind == ind].iloc[0]
            except IndexError:
                raise IndexError(f'No SNV at {ind[0]}, {ind[1]} (allele {ind[2]}) found.')
            
            groups = ['_'.join(c.split('_')[1:]) for c in anova.columns if (c != 'n_all') and c.startswith('n_')]
            
            
            title = ', '.join(map(str, snv.ind))
            if snv_name:
                title += f', id: {snv_name}'
            title += f'. Allele: {allele}'
            es_cols = [f'{allele}_es_{name}' for name in groups]
            es = snv[es_cols]
            es = es.dropna()
            inds = list()
            for i, name in enumerate(groups):
                if f'{allele}_es_{name}' in es.index:
                    inds.append(i)
            es = es.values
            inds_s = np.argsort(es)
            inds = [inds[i] for i in inds_s]
            groups = [groups[i] for i in inds]
            es = es[inds_s]
            es_var_cols = [f'{allele}_es_var_{name}' for name in groups]
            es_var = snv[es_var_cols]
            p_cols = [f'{allele}_p_{name}' for name in groups]
            ps = snv[p_cols].values
            p_all = snv[f'{allele}_p_all']
            ps = np.log2(ps.astype(float)) - np.log2(p_all)
            
            pvals_fdr_cols = [f'{allele}_fdr_pval_{name}' for name in groups]
            pvals_fdr = snv[pvals_fdr_cols]
            
            pvals_cols = [f'{allele}_pval_{name}' for name in groups]
            pvals = snv[pvals_cols]
            
            if plot_test_es:
                test_res = test[allele][1]
            es_raw = list()
            es_raw_var = list()
            es_comb = list()
            es_comb_var = list()
            
            for i in inds:
                lt = snvs_db[i][snv.ind][1:]
                raws = list()
                raws_es = list()
                for t in lt:
                    ref, alt = t[1:3]
                    t = np.log2(ref) - np.log2(alt)
                    raws.append(t if allele == 'ref' else -t)
                    if plot_test_es:
                        raws_es.append(test_res[(ref, alt)][1])
                es_raw.append(np.mean(raws))
                es_raw_var.append(np.var(raws))
                if plot_test_es:
                    es_comb.append(np.mean(raws_es))
                    es_comb_var.append(np.var(raws_es))
                
            
            n_rows = 1 + plot_p_diff + plot_test_es + plot_raw_es
            
            plt.figure(dpi=dpi, figsize=(figsize[0], figsize[1] * n_rows + 2))            
            
            i = 1
            plot_barplot(groups, es, n_rows, i, es_var, legend_colors=True)
            plt.ylabel(f'$E\\left[log_2\\left( \\frac{{group_{{{allele}}}}}{{all_{{{allele}}}}}\\right)\\right]$')
            i += 1
            if plot_p_diff:
                plot_barplot(groups, ps, n_rows, i)
                plt.ylabel(r'$log_2\left(\frac{p_{group}}{p_{all}}\right)$')
                i += 1
            
            if plot_test_es:
                plot_barplot(groups, es_comb, n_rows, i, es_comb_var)
                plt.ylabel('Average of effect-sizes from test')
                i += 1
            
            if plot_raw_es:
                plot_barplot(groups, es_raw, n_rows, i, es_raw_var)
                plt.ylabel('Average of ' + r'$log_2\left(\frac{ref}{alt}\right)$' if allele == 'ref' else r'$log_2\left(\frac{alt}{ref}\right)$')
                i += 1
            plt.xticks(groups, rotation=90, fontsize=_fontsize - 3)
            plt.gca().set_xticklabels(groups, rotation=90, fontsize=_fontsize - 3)
            plt.suptitle((title + '\tWhiskers S.D.').expandtabs(), horizontalalignment='left', verticalalignment='top', x=0, fontsize=18)
            plt.tight_layout()
            name = snv_name if snv_name else '_'.join(map(str, snv.ind))
            plt.savefig(os.path.join(folder, f'{name}_{ind[-1]}_{allele}.{ext}'))

def visualize(name: str, output: str, what: str, fmt='png', slices=(5, 10, 15, 20, 30, 40, 50),
              max_count=100, slice_ref=True, fbad=None, show_bad=True, dpi=200):
    filename = get_init_file(name)
    compressor = filename.split('.')[-1]
    open = openers[compressor]
    with open(filename, 'rb') as f:
        counts = dill.load(f)
        covers, biases = scorefiles_qc(counts)
        scorefiles = counts['scorefiles']
        counts = counts['counts']
    filename = f'{name}.fit.{compressor}'
    with open(filename, 'rb') as f:
        fits = dill.load(f)
    bads = [fbad] if fbad else sorted(counts)
    if what == 'all' and fbad is None:
        os.makedirs(output, exist_ok=True)
        filename = os.path.join(output, f'scorefiles_qc.{fmt}')
        plot_scorefiles_qc(covers, biases, scorefiles)
        if show_bad:
            plt.title('All BADs')
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        
    for bad in bads:
        if not fbad:
            subfolder = os.path.join(output, f'BAD{bad:.2f}')
        else:
            subfolder = output
        inst_params = fits['ref'][bad]['inst_params']
        inst_params['name'] = 'slice'
        dist = inst_params['dist']
        m = get_model_creator(**inst_params)()
        if what == 'all':
            os.makedirs(subfolder, exist_ok=True)
            filename = os.path.join(subfolder, f'scorefiles_qc.{fmt}')
            plot_scorefiles_qc(covers, biases, scorefiles, bad=bad, dpi=dpi)
            plt.tight_layout()
            if show_bad:
                plt.title(f'BAD = {bad:.2f}')
            plt.savefig(filename, bbox_inches='tight')
            filename = os.path.join(subfolder, f'gof.{fmt}')
            plot_gof(fits['ref'][bad]['stats'], fits['alt'][bad]['stats'], max_count, dpi=dpi)
            if show_bad:
                plt.title(f'BAD = {bad:.2f}')
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight')
            filename = os.path.join(subfolder, f'r.{fmt}')
            try:
                plot_params(fits['ref'][bad]['params'], fits['alt'][bad]['params'], max_count, 'r', diag=True, dpi=dpi)
                if show_bad:
                    plt.title(f'BAD = {bad:.2f}')
                plt.tight_layout()
                plt.savefig(filename, bbox_inches='tight')
            except KeyError:
                pass
            if dist.startswith('Beta'):
                filename = os.path.join(subfolder, f'k.{fmt}')
                try:
                    plot_params(fits['ref'][bad]['params'], fits['alt'][bad]['params'], max_count, 'k', inv=True, name='$1/\kappa$', dpi=dpi)
                    if show_bad:
                        plt.title(f'BAD = {bad:.2f}')
                    plt.tight_layout()
                    plt.savefig(filename, bbox_inches='tight')
                except KeyError:
                    pass
            if bad != 1:
                filename = os.path.join(subfolder, f'w.{fmt}')
                try:
                    plot_params(fits['ref'][bad]['params'], fits['alt'][bad]['params'], max_count, 'w', dpi=dpi)
                    if show_bad:
                        plt.title(f'BAD = {bad:.2f}')
                    plt.tight_layout()
                    plt.savefig(filename, bbox_inches='tight')
                except KeyError:
                    pass
            filename = os.path.join(subfolder, f'p1.{fmt}')
            try:
                plot_params(fits['ref'][bad]['params'], fits['alt'][bad]['params'], max_count, 'p1', name='$p_1$', dpi=dpi,
                            hor_expected=bad/(bad + 1)
                            )
                if show_bad:
                    plt.title(f'BAD = {bad:.2f}')
                plt.tight_layout()
                plt.savefig(filename, bbox_inches='tight')
            except KeyError:
                pass
            if bad > 1:
                filename = os.path.join(subfolder, f'p2.{fmt}')
                try:
                    plot_params(fits['ref'][bad]['params'], fits['alt'][bad]['params'], max_count, 'p2', name='$p_2$', dpi=dpi,
                                hor_expected=1/(bad + 1))
                    if show_bad:
                        plt.title(f'BAD = {bad:.2f}')
                    plt.tight_layout()
                    plt.savefig(filename, bbox_inches='tight')
                except KeyError:
                    pass
                filename = os.path.join(subfolder, f'p0.{fmt}')
                try:
                    plot_params(fits['ref'][bad]['params'], fits['alt'][bad]['params'], max_count, ('p1', 'p2'), name='$p_0$', dpi=dpi,
                                hor_expected=1.0, std=False)
                    if show_bad:
                        plt.title(f'BAD = {bad:.2f}')
                    plt.tight_layout()
                    plt.savefig(filename, bbox_inches='tight')
                except KeyError:
                    pass
                filename = os.path.join(subfolder, f'p.{fmt}')
                try:
                    plot_params(fits['ref'][bad]['params'], fits['alt'][bad]['params'], max_count, ('p1', 'p2', 'p1'), name='$p$', dpi=dpi,
                                hor_expected=bad/(bad + 1), std=False)
                    if show_bad:
                        plt.title(f'BAD = {bad:.2f}')
                    plt.tight_layout()
                    plt.savefig(filename, bbox_inches='tight')
                except KeyError:
                    pass
            filename = os.path.join(subfolder, f'n.{fmt}')
            plot_stat(fits['ref'][bad]['stats'], fits['alt'][bad]['stats'], max_count, 'n', dpi=dpi, log=True)
            if show_bad:
                plt.title(f'BAD = {bad:.2f}')
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight')
            for slc in slices:
                if type(slc) is tuple:
                    ref, alt = slc
                    filename = os.path.join(subfolder, f'slices_{ref}_{alt}.{fmt}')
                else:
                    ref = alt = slc
                    filename = os.path.join(subfolder, f'slices_{ref}.{fmt}')
                sliceplot(counts[bad], max_count, ref, alt, m, fits['ref'][bad]['params'], fits['alt'][bad]['params'], dpi=dpi)
                if show_bad:
                    plt.suptitle(f'BAD = {bad:.2f}')
                plt.tight_layout()
                plt.savefig(filename, bbox_inches='tight')
        else:
            if what == 'gof':
                plot_gof(fits['ref'][bad]['stats'], fits['alt'][bad]['stats'], max_count, dpi=dpi)
            elif what == 'r':
                plot_params(fits['ref'][bad]['params'], fits['alt'][bad]['params'], max_count, 'r', diag=True, dpi=dpi)
            elif what == 'k':
                plot_params(fits['ref'][bad]['params'], fits['alt'][bad]['params'], max_count, 'k', inv=True, name='$1/\kappa$', dpi=dpi)
            elif what == 'w':
                plot_params(fits['ref'][bad]['params'], fits['alt'][bad]['params'], max_count, 'w', dpi=dpi)
            elif what == 'n':
                plot_stat(fits['ref'][bad]['stats'], fits['alt'][bad]['stats'], max_count, 'n', dpi=dpi)
            elif what == 'sliceplot':
                if type(slices) is tuple:
                    ref, alt = slices
                else:
                    ref = alt = slices
                sliceplot(counts[bad], max_count, ref, alt, m, fits['ref'][bad]['params'], fits['alt'][bad]['params'], dpi=dpi)
            elif what == 'counts':
                plt.figure(figsize=(6, 6), dpi=dpi)
                plot_heatmap(counts[bad], max_count)
            elif what == 'slice':
                plt.figure(figsize=(6, 6), dpi=dpi)
                pdf = get_pdf_computer(m, dictify_params(fits['ref' if slice_ref else 'alt'][bad]['params']))
                plot_histogram(counts[bad], max_count, slices[0], s=not slice_ref, pdf_computer=pdf)
            if show_bad:
                plt.suptitle(f'BAD = {bad:.2f}')
            plt.tight_layout()
            plt.savefig(f'{output}.{fmt}' if fbad else f'{output}.{bad:.2f}.{fmt}', bbox_inches='tight')
