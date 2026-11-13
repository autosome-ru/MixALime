#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from .utils import get_init_file, scorefiles_qc, load_pickle, closest_bad
from .params import SliceParams
from .stats import mixture_logpmf
from itertools import product
import pandas as pd
import logging
import os

_fontsize = 16
_ref = '#DC267F'
_alt = '#FFB000'
_count = '#648FFF'
_cmap = LinearSegmentedColormap.from_list("", ['white', _count])
_markersize = 8

class _DropFontWeightNotes(logging.Filter):
    """Drops the notes of matplotlib on a fallback font without the medium weight."""

    def filter(self, record):
        return 'Failed to find font weight' not in record.getMessage()


def update_style():
    logging.getLogger('matplotlib.font_manager').addFilter(_DropFontWeightNotes())
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
    # Rows of an image are its vertical axis, so the alt count indexes the rows.
    hm[counts[:, 1].astype(int), counts[:, 0].astype(int)] += counts[:, 2]
    # Counts all beyond the window leave a blank image with a colorbar of one order.
    max_order = int(np.ceil(np.log10(counts[:, 2].max() + 1))) if len(counts) else 1
    hm = np.log10(hm)

    plt.imshow(hm, cmap=cmap, vmin=0, vmax=max_order)
    if slices:
        # The ref model is shown at a fixed alt count and the alt model at a fixed ref count.
        a, b = slices
        plt.hlines(a, 0, max_count, colors=_ref, linestyles='dashed', linewidth=3)
        plt.vlines(b, 0, max_count, colors=_alt, linestyles='dashed', linewidth=3)
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

def get_pdf_computer(params: dict, inst_params: dict):
    """
    Fitted pmf of an allele as a function of (counts, slice).

    params is the 'params' entry of a fit and inst_params its 'inst_params' entry.
    """
    sp = SliceParams(params)
    bad, left, dist = inst_params['bad'], int(inst_params['left']), inst_params['dist']
    r_transform = inst_params.get('r_transform')

    def pdf(x, slc):
        x = np.asarray(x, dtype=float)
        s = np.full(len(x), float(slc))
        return np.exp(mixture_logpmf(x, sp.at(s), bad, left, dist, r_transform))
    return pdf


def param_curve(params: dict, x: np.ndarray, name: str, field='ests') -> np.ndarray:
    """
    Values of the parameter name at the slices x, nan below the first fitted slice.

    A KeyError is raised if the fit lacks the parameter or the field.
    """
    sp = SliceParams(params, field)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.array(sp.at(x, std=field == 'stds')[name], dtype=float)
    res[x < sp.slices[0]] = np.nan
    return res


def plot_histogram(counts: np.ndarray, max_count: int, slc: int, pdf_computer, s=0, c='r', left=-1):
    counts = counts[counts[:, 1 - s] == slc][:, [s, 2]]
    # The fit leaves out the counts up to left and the pmf sums to one over the rest.
    counts = counts[counts[:, 0] > left, :]
    # Bars are fractions of the whole slice, as the pmf is, and not of its part below max_count.
    total = counts[:, 1].sum()
    counts = counts[counts[:, 0] < max_count, :]
    plt.bar(counts[:, 0], counts[:, 1] / total, width=1, color=_count)
    x = np.arange(0, max_count)
    y = pdf_computer(x, slc)
    if s:
        plt.xlabel('Alternative allele read counts')
    else:
        plt.xlabel('Reference allele read counts')
    ax = plt.plot(x, y, color=c, linewidth=2)[0].axes
    return ax

def sliceplot(counts: np.ndarray, max_count:int, ref: int, alt: int, fit_ref: dict, fit_alt: dict,
              figsize=(20, 6), dpi=200):
    """
    Heatmap of the counts of a BAD beside the histograms of the ref and alt models with their fitted pmfs.

    ref is the slice of the ref model, that is the alt count at which its histogram is taken, and
    alt is the slice of the alt model, that is the ref count. fit_ref and fit_alt are the fits of the BAD.
    """
    pdf_ref = get_pdf_computer(fit_ref['params'], fit_ref['inst_params'])
    pdf_alt = get_pdf_computer(fit_alt['params'], fit_alt['inst_params'])
    plt.figure(figsize=figsize, dpi=dpi)
    plt.subplot(1, 3, 1)
    plot_heatmap(counts, max_count, (ref, alt))
    plt.subplot(1, 3, 2)
    ax1 = plot_histogram(counts, max_count, ref, pdf_ref, c=_ref, left=int(fit_ref['inst_params']['left']))
    ylim1 = ax1.get_ylim()[1]
    plt.subplot(1, 3, 3)
    ax2 = plot_histogram(counts, max_count, alt, pdf_alt, 1, c=_alt, left=int(fit_alt['inst_params']['left']))
    ylim2 = ax2.get_ylim()[1]
    ylim = max(ylim1, ylim2)
    ax1.set_ylim(0, ylim)
    ax2.set_ylim(0, ylim)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

def plot_gof(stats_ref: dict, stats_alt: dict, max_count: int, figsize=(6, 6), dpi=200):
    x = set(stats_ref.keys()) & set(stats_alt.keys())
    x = np.array(list(filter(lambda x: x < max_count, sorted(x))))
    stats_ref = np.array([stats_ref[k]['rmsea'] for k in x])
    stats_alt = np.array([stats_alt[k]['rmsea'] for k in x])
    
    # ticks = np.arange(0, max(max(stats_ref), max(stats_alt)), 0.05)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(x, stats_ref, 'o', color=_ref, markersize=_markersize * 1.1)
    plt.plot(x, stats_alt, 'o', color=_alt, markersize=_markersize, alpha=0.85)
    # plt.yticks(ticks)
    plt.axhline(0.05, color='k', linestyle='dashed')
    plt.grid(True)
    plt.legend(['ref', 'alt'])
    plt.xlabel('Read count for the fixed allele')
    plt.ylabel('Goodness of fit, RMSEA')
    
def plot_stat(stats_ref: dict, stats_alt: dict, max_count: int, stat: str, figsize=(6, 6), dpi=200, log=False):
    x = set(stats_ref.keys()) & set(stats_alt.keys())
    x = np.array(list(filter(lambda x: x < max_count, sorted(x))))
    stats_ref = np.array([stats_ref[k][stat] for k in x])
    stats_alt = np.array([stats_alt[k][stat] for k in x])
    if log:
        # A zero count has no logarithm and is left out of the plot.
        stats_ref = np.log10(np.where(stats_ref > 0, stats_ref, np.nan))
        stats_alt = np.log10(np.where(stats_alt > 0, stats_alt, np.nan))
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(x, stats_ref, 'o', color=_ref, markersize=_markersize * 1.1)
    plt.plot(x, stats_alt, 'o', color=_alt, markersize=_markersize, alpha=0.85)
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
    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(x, y, s=15, c=-np.array(y))
    for xc, yc, label in zip(x[inds], y[inds], labels[inds]):
        plt.annotate(label, (xc, yc), fontsize=8)
    plt.xlabel(r'$log_{10}(\mathrm{coverage})$')
    plt.ylabel(r'Fraction of SNVs when $ref > alt$')
    plt.grid(True)

def plot_params(params_ref: dict, params_alt: dict, max_count: int, param: str,
                figsize=(6, 6), dpi=200, inv=False,  diag=False, name=None,
                hor_expected=None, std=True):
    x = np.arange(0, max_count)
    names = (param,) if type(param) is str else tuple(param)
    cols_ref = [param_curve(params_ref, x, n) for n in names]
    cols_alt = [param_curve(params_alt, x, n) for n in names]
    if len(names) == 1:
        pref, palt = cols_ref[0], cols_alt[0]
    elif len(names) == 2:
        pref, palt = cols_ref[0] + cols_ref[1], cols_alt[0] + cols_alt[1]
    else:
        pref, palt = cols_ref[2] / (cols_ref[0] + cols_ref[1]), cols_alt[2] / (cols_alt[0] + cols_alt[1])
    std = std and len(names) == 1 and 'stds' in params_ref and 'stds' in params_alt
    if std:
        sref = param_curve(params_ref, x, param, 'stds')
        salt = param_curve(params_alt, x, param, 'stds')
    if inv:
        pref = 1 / pref
        palt = 1 / palt
        if std:
            sref *= pref ** 2
            salt *= palt ** 2
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(x, pref, 'o', color=_ref, markersize=_markersize * 1.1)
    plt.plot(x, palt, 'o', color=_alt, markersize=_markersize, alpha=0.85)
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

def _bad_digits(bads) -> int:
    """Decimal places of a BAD in names and titles: two, or more when two of the bads would read the same."""
    bads = set(bads)
    for digits in range(2, 5):
        if len({f'{bad:.{digits}f}' for bad in bads}) == len(bads):
            return digits
    return 5


def _file_stem(snv_name, ind) -> str:
    """The id of an SNV as a file name. A missing id ('.'), a leading dot and path separators are not kept."""
    stem = str(snv_name).translate({ord('/'): '_', ord('\\'): '_'}).lstrip('.') if snv_name else str()
    return stem or '_'.join(map(str, ind))


def plot_anova_snvs(name: str, snv_names=None, snvs=None, subname=None, plot_raw_es=True, plot_test_es=False,
                    plot_p_diff=True, color_significant=True, folder=str(), ext='png', 
                    figsize=(12, 4), dpi=200):
    """
    Draws the effect-sizes of the groups of an ANOVA run: a figure for each SNV, BAD and allele.

    snv_names are pairs (id, alt allele) and snvs are triples (chromosome, position, alt allele), drawn in the
    order given. A KeyError naming every SNV that was not tested is raised before anything is drawn.
    plot_test_es needs the test file of the project.
    """
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
        
    filename = get_init_file(name)
    compressor = filename.split('.')[-1]
    anova = load_pickle(f'{name}.anova.{compressor}')
    if subname not in anova:
        raise KeyError(f'No ANOVA results under the subname {subname}, available: {", ".join(map(str, anova))}.')
    anova = anova[subname]
    snvs_db = anova['snvs']
    anova = anova['tests']
    if plot_test_es:
        test = load_pickle(f'{name}.test.{compressor}')
    ids = dict()
    for g in snvs_db:
        for ind, lt in g.items():
            if lt:
                ids.setdefault(ind, lt[0][0])
    wanted = [tuple(ind) for ind in snvs or ()]
    unknown = list()
    for snv_name, allele in snv_names or ():
        found = sorted(ind for ind, i in ids.items() if i == snv_name and ind[-1] == allele)
        if len(found) > 1:
            logging.warning(f'{len(found)} SNVs have the id {snv_name} and alt allele {allele}, the first by position is drawn.')
        if found:
            wanted.append(found[0])
        else:
            unknown.append(f'id {snv_name} with alt allele {allele}')
    # An SNV is drawn once, and it is known only if the ANOVA has a row for it.
    rows_of = dict()
    for ind in dict.fromkeys(wanted):
        rows = anova[anova.ind == ind]
        if len(rows):
            rows_of[ind] = rows
        else:
            unknown.append(f'SNV {", ".join(map(str, ind))}')
    if unknown:
        raise KeyError('Not among the tested SNVs: ' + '; '.join(unknown) + '.')
    if folder:
        os.makedirs(folder, exist_ok=True)
    digits = _bad_digits(anova['bad']) if 'bad' in anova else 2
    all_groups = ['_'.join(c.split('_')[1:]) for c in anova.columns if (c != 'n_all') and c.startswith('n_')]
    for ind, rows in rows_of.items():
        snv_name = ids.get(ind)
        # An SNV has a row for each BAD it was tested at.
        for (_, snv), allele in product(rows.iterrows(), ('ref', 'alt')):
            bad = snv['bad'] if 'bad' in snv.index else None
            title = ', '.join(map(str, snv.ind))
            if snv_name:
                title += f', id: {snv_name}'
            if bad is not None and len(rows) > 1:
                title += f', BAD: {bad:.{digits}f}'
            title += f'. Allele: {allele}'
            es = pd.to_numeric(snv[[f'{allele}_es_{g}' for g in all_groups]], errors='coerce')
            inds = [i for i, v in enumerate(es.values) if not np.isnan(v)]
            if not inds:
                continue
            inds = [inds[i] for i in np.argsort(es.values[inds])]
            groups = [all_groups[i] for i in inds]
            get = lambda col: pd.to_numeric(snv[[f'{allele}_{col}_{g}' for g in groups]], errors='coerce').values
            es, es_var = get('es'), get('es_var')
            ps = np.log2(get('p')) - np.log2(float(snv[f'{allele}_p_all']))
            pvals_fdr, pvals = get('fdr_pval'), get('pval')
            es_raw, es_raw_var, es_comb, es_comb_var = list(), list(), list(), list()
            for i in inds:
                lt = snvs_db[i][snv.ind][1:]
                if bad is not None:
                    lt = [t for t in lt if t[-1] == bad]
                raws = list()
                raws_es = list()
                for t in lt:
                    ref, alt = t[1:3]
                    v = np.log2(ref) - np.log2(alt)
                    raws.append(v if allele == 'ref' else -v)
                    if plot_test_es:
                        raws_es.append(test[allele][t[-1]][(ref, alt)][1])
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
                plt.ylabel('Average of ' + (r'$log_2\left(\frac{ref}{alt}\right)$' if allele == 'ref'
                                            else r'$log_2\left(\frac{alt}{ref}\right)$'))
                i += 1
            plt.xticks(groups, rotation=90, fontsize=_fontsize - 3)
            plt.gca().set_xticklabels(groups, rotation=90, fontsize=_fontsize - 3)
            plt.suptitle((title + '\tWhiskers S.D.').expandtabs(), horizontalalignment='left', verticalalignment='top', x=0, fontsize=18)
            plt.tight_layout()
            file = _file_stem(snv_name, ind) + f'_{ind[-1]}_{allele}'
            if bad is not None and len(rows) > 1:
                file += f'_BAD{bad:.{digits}f}'
            plt.savefig(os.path.join(folder, f'{file}.{ext}'))
            plt.close()


def _save(filename: str, title=None, suptitle=False):
    """Saves the current figure under filename and closes it."""
    if title:
        (plt.suptitle if suptitle else plt.title)(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def _slice_pairs(slices) -> list:
    """The slices as pairs (ref, alt), an int s standing for (s, s)."""
    if not np.iterable(slices):
        slices = [slices]
    return [tuple(s) if np.iterable(s) else (s, s) for s in slices]


def _one_slice(slices) -> tuple:
    """The (ref, alt) of a single figure: an int s, a pair (ref, alt), or the first of a longer list of slices."""
    if np.iterable(slices) and len(slices) == 2 and not np.iterable(slices[0]):
        return tuple(slices)
    return _slice_pairs(slices)[0]


def _slice_problem(counts: np.ndarray, slc: int, s: int, params: dict):
    """Why the model of allele s (0 for ref, 1 for alt) cannot be set against the data at slice slc, None if it can."""
    first = SliceParams(params).slices[0]
    if slc < first:
        return f'the {("ref", "alt")[s]} model is fitted from slice {first}'
    if not (counts[:, 1 - s] == slc).any():
        return f'no SNV has {("alt", "ref")[s]} count {slc}'


def visualize(name: str, output: str, what: str, fmt='png', slices=(5, 10, 15, 20, 30, 40, 50),
              max_count=100, slice_ref=True, fbad=None, show_bad=True, dpi=200):
    """
    Draws the figures of a fitted project into the folder output, or one figure into the file output.fmt.

    what is 'all' for every figure of every BAD, or 'gof', 'r', 'k', 'w', 'n', 'counts', 'sliceplot' or 'slice'.
    slices lists the slices of sliceplots, each an int s for (s, s) or a pair (ref, alt) as in sliceplot. 'all' skips
    with a warning a slice that has no data or lies below the first fitted one. 'sliceplot' and 'slice' draw a single
    slice: an int, a pair (ref, alt) or the first of a longer list. 'slice' draws only the ref model, or the alt model
    if slice_ref is False. fbad selects a BAD.
    """
    filename = get_init_file(name)
    compressor = filename.split('.')[-1]
    counts = load_pickle(filename)
    covers, biases = scorefiles_qc(counts)
    scorefiles = counts['scorefiles']
    counts = counts['counts']
    fits = load_pickle(f'{name}.fit.{compressor}')
    bads = [closest_bad(fits['ref'], fbad)] if fbad is not None else sorted(fits['ref'])
    digits = _bad_digits(fits['ref'])
    if what == 'all' and fbad is None:
        os.makedirs(output, exist_ok=True)
        plot_scorefiles_qc(covers, biases, scorefiles, dpi=dpi)
        _save(os.path.join(output, f'scorefiles_qc.{fmt}'), 'All BADs' if show_bad else None)

    for bad in bads:
        if fbad is None:
            subfolder = os.path.join(output, f'BAD{bad:.{digits}f}')
        else:
            subfolder = output
        fit_ref, fit_alt = fits['ref'][bad], fits['alt'][bad]
        params_ref, params_alt = fit_ref['params'], fit_alt['params']
        stats_ref, stats_alt = fit_ref['stats'], fit_alt['stats']
        inst = fit_ref['inst_params']
        dist = inst['dist']
        # The distribution of a BAD can differ from the requested one, as small datasets get a simpler model.
        model_postfix = f', {dist}'
        if dist.startswith('Beta') and inst.get('regul_alpha'):
            model_postfix += ', alpha = {:.2f}'.format(inst['regul_alpha'])
        title = f'BAD = {bad:.{digits}f}' if show_bad else None
        title_model = title + model_postfix if show_bad else None
        if what == 'all':
            os.makedirs(subfolder, exist_ok=True)
            plot_scorefiles_qc(covers, biases, scorefiles, bad=bad, dpi=dpi)
            _save(os.path.join(subfolder, f'scorefiles_qc.{fmt}'), title)
            plot_gof(stats_ref, stats_alt, max_count, dpi=dpi)
            _save(os.path.join(subfolder, f'gof.{fmt}'), title_model)
            # File name, parameter, whether the plot applies and the options of plot_params.
            plots = [('r', 'r', True, dict(diag=True)),
                     ('k', 'k', dist.startswith('Beta'), dict(inv=True, name=r'$1/\kappa$')),
                     ('w', 'w', bad != 1, dict()),
                     ('p1', 'p1', True, dict(name='$p_1$', hor_expected=bad / (bad + 1))),
                     ('p2', 'p2', bad > 1, dict(name='$p_2$', hor_expected=1 / (bad + 1))),
                     ('p0', ('p1', 'p2'), bad > 1, dict(name='$p_0$', hor_expected=1.0, std=False)),
                     ('p', ('p1', 'p2', 'p1'), bad > 1, dict(name='$p$', hor_expected=bad / (bad + 1), std=False))]
            for file, param, applies, kwargs in plots:
                if not applies:
                    continue
                try:
                    plot_params(params_ref, params_alt, max_count, param, dpi=dpi, **kwargs)
                except KeyError:
                    continue
                _save(os.path.join(subfolder, f'{file}.{fmt}'), title_model)
            plot_stat(stats_ref, stats_alt, max_count, 'n', dpi=dpi, log=True)
            _save(os.path.join(subfolder, f'n.{fmt}'), title)
            for ref, alt in _slice_pairs(slices):
                problem = _slice_problem(counts[bad], ref, 0, params_ref) or _slice_problem(counts[bad], alt, 1, params_alt)
                if problem:
                    which = f'slice {ref}' if ref == alt else f'slices {ref} and {alt}'
                    logging.warning(f'BAD {bad:.{digits}f}: skipping {which}, {problem}.')
                    continue
                filename = os.path.join(subfolder, f'slices_{ref}.{fmt}' if ref == alt else f'slices_{ref}_{alt}.{fmt}')
                sliceplot(counts[bad], max_count, ref, alt, fit_ref, fit_alt, dpi=dpi)
                _save(filename, title_model, suptitle=True)
        else:
            if what == 'gof':
                plot_gof(stats_ref, stats_alt, max_count, dpi=dpi)
            elif what == 'r':
                plot_params(params_ref, params_alt, max_count, 'r', diag=True, dpi=dpi)
            elif what == 'k':
                plot_params(params_ref, params_alt, max_count, 'k', inv=True, name=r'$1/\kappa$', dpi=dpi)
            elif what == 'w':
                plot_params(params_ref, params_alt, max_count, 'w', dpi=dpi)
            elif what == 'n':
                plot_stat(stats_ref, stats_alt, max_count, 'n', dpi=dpi)
            elif what == 'sliceplot':
                ref, alt = _one_slice(slices)
                sliceplot(counts[bad], max_count, ref, alt, fit_ref, fit_alt, dpi=dpi)
            elif what == 'counts':
                plt.figure(figsize=(6, 6), dpi=dpi)
                plot_heatmap(counts[bad], max_count)
            elif what == 'slice':
                plt.figure(figsize=(6, 6), dpi=dpi)
                fit = fit_ref if slice_ref else fit_alt
                pdf = get_pdf_computer(fit['params'], fit['inst_params'])
                ref, alt = _one_slice(slices)
                plot_histogram(counts[bad], max_count, ref if slice_ref else alt, s=int(not slice_ref), pdf_computer=pdf,
                               left=int(fit['inst_params']['left']))
            else:
                raise ValueError(f'Unknown plot: {what}.')
            _save(f'{output}.{fmt}' if fbad is not None else f'{output}.{bad:.{digits}f}.{fmt}', title, suptitle=True)
