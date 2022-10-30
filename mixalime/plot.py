#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
from betanegbinfit import ModelMixture
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from .utils import openers, get_init_file, get_model_creator, dictify_params
from betanegbinfit.utils import get_params_at_slice
from scipy.interpolate import UnivariateSpline
import dill
import os


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
    plt.rcParams['font.size'] = 16

update_style()
_ref = '#DC267F'
_alt = '#FFB000'
_count = '#648FFF'
_cmap = LinearSegmentedColormap.from_list("", ['white', _count])
_markersize = 8


def plot_heatmap(counts: np.ndarray, max_count: int, slices=None, shift=10, cmap=_cmap):
    hm = np.ones((max_count + shift , max_count + shift))
    counts = counts[(counts[:, 0] < max_count + shift) & (counts[:, 1] < max_count + shift)]
    m = counts[:, [0,1]].min()
    hm[counts[:, 0] - m, counts[:, 1] - m] += counts[:, 2]
    max_order = int(np.ceil(np.log10(counts[:, 2].max() + 1 )))
    hm = np.log10(hm)

    plt.imshow(hm, cmap=cmap, vmin=0, vmax=max_order)
    if slices:
        a, b = slices
        plt.vlines(a, 0, max_count, colors=_alt, linestyles='dashed', linewidth=3)
        plt.hlines(b, 0, max_count, colors=_ref, linestyles='dashed', linewidth=3)
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

    
def plot_histogram(counts: np.ndarray, max_count: int, slc: int, pdf_computer, s=0, c='r'):
    
    counts = counts[counts[:, s] < max_count, :]
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
    plt.plot(x, stats_ref, 'o', color=_ref, markersize=_markersize)
    plt.plot(x, stats_alt, 'o', color=_alt, markersize=_markersize)
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
    
def plot_stat(stats_ref: dict, stats_alt: dict, max_count: int, stat: str, figsize=(6, 6), dpi=200, spline=False):
    x = set(stats_ref.keys()) & set(stats_alt.keys())
    x = np.array(list(filter(lambda x: x < max_count, sorted(x))))
    stats_ref = np.array([stats_ref[k][stat] for k in x])
    stats_alt = np.array([stats_alt[k][stat] for k in x])
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(x, stats_ref, 'o', color=_ref, markersize=_markersize)
    plt.plot(x, stats_alt, 'o', color=_alt, markersize=_markersize)
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
    plt.ylabel(stat)

def plot_params(params_ref: dict, params_alt: dict, max_count: int, param: str,
                figsize=(6, 6), dpi=200, inv=False, diag=False, name=None, spline=False):
    params_ref = dictify_params(params_ref)
    params_alt = dictify_params(params_alt)
    x = np.arange(0, max_count)
    pref = np.array([get_params_at_slice(params_ref, i, clip_at_max_slice=False).get(param, np.nan) for i in x])
    palt = np.array([get_params_at_slice(params_alt, i, clip_at_max_slice=False).get(param, np.nan) for i in x])
    if inv:
        pref = 1 / pref
        palt = 1 / palt
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(x, pref, 'o', color=_ref, markersize=_markersize)
    plt.plot(x, palt, 'o', color=_alt, markersize=_markersize)
    if spline:
        nx = np.linspace(0, max_count, max_count * 10)
        inds = ~np.isnan(pref)
        spline = UnivariateSpline(x[inds], pref[inds], s=0.001, k=4)
        plt.plot(nx, spline(nx), 'b--')
        inds = ~np.isnan(palt)
        spline = UnivariateSpline(x[inds], palt[inds], s=0.001, k=4)
        plt.plot(nx, spline(nx), 'y--')
    if diag:
        plt.axline((1, 1), slope=1, linestyle='dashed', color='k')
    plt.grid(True)
    plt.legend(['ref', 'alt'])
    plt.xlabel('Read count for the fixed allele')
    plt.ylabel(name if name else param)
    

def visualize(name: str, output: str, what: str, fmt='png', slices=(5, 10, 15, 20, 30, 40, 50),
              max_count=100, slice_ref=True, fbad=None, show_bad=True, dpi=200):
    filename = get_init_file(name)
    compressor = filename.split('.')[-1]
    open = openers[compressor]
    with open(filename, 'rb') as f:
        counts = dill.load(f)['counts']
    filename = f'{name}.fit.{compressor}'
    with open(filename, 'rb') as f:
        fits = dill.load(f)
    bads = [fbad] if fbad else sorted(counts) 
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
            filename = os.path.join(subfolder, f'gof.{fmt}')
            plot_gof(fits['ref'][bad]['stats'], fits['alt'][bad]['stats'], max_count, dpi=dpi)
            if show_bad:
                plt.title(f'BAD = {bad:.2f}')
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight')
            filename = os.path.join(subfolder, f'r.{fmt}')
            plot_params(fits['ref'][bad]['params'], fits['alt'][bad]['params'], max_count, 'r', diag=True, dpi=dpi)
            if show_bad:
                plt.title(f'BAD = {bad:.2f}')
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight')
            if dist == 'BetaNB':
                filename = os.path.join(subfolder, f'k.{fmt}')
                plot_params(fits['ref'][bad]['params'], fits['alt'][bad]['params'], max_count, 'k', inv=True, name='$1/\kappa$', dpi=dpi)
                if show_bad:
                    plt.title(f'BAD = {bad:.2f}')
                plt.tight_layout()
                plt.savefig(filename, bbox_inches='tight')
            if bad != 1:
                filename = os.path.join(subfolder, f'w.{fmt}')
                plot_params(fits['ref'][bad]['params'], fits['alt'][bad]['params'], max_count, 'w', dpi=dpi)
                if show_bad:
                    plt.title(f'BAD = {bad:.2f}')
                plt.tight_layout()
                plt.savefig(filename, bbox_inches='tight')
            filename = os.path.join(subfolder, f'n.{fmt}')
            plot_stat(fits['ref'][bad]['stats'], fits['alt'][bad]['stats'], max_count, 'n', dpi=dpi)
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
