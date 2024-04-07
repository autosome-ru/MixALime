# -*- coding: utf-8 -*-
from . import check_packages
from enum import Enum
from click import Context
from typer import Typer, Option, Argument
from typer.core import TyperGroup
from typing import List, Tuple
from rich import print as rprint
from betanegbinfit import __version__ as bnb_version
from jax import __version__ as jax_version
from scipy import __version__ as scipy_version
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from .diff import differential_test, anova_test
from .create import create_project
from .combine import combine
from .tests import test, binom_test
from pathlib import Path
from .fit import fit
from time import time
from dill import __version__ as dill_version
import importlib
import logging
from . import export
from . import __version__ as mixalime_version
import json
from . import plot

logging.getLogger("jax._src.xla_bridge").addFilter(logging.Filter("No GPU/TPU found, falling back to CPU."))
logging.getLogger("jax._src.xla_bridge").addFilter(logging.Filter("An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu."))

__all__ = ['main']

class Compression(str, Enum):
    lzma = 'lzma'
    gzip = 'gzip'
    bz2 = 'bz2'
    raw = 'raw'

class Model(str, Enum):
    window = 'window'
    line = 'line'
    slices = 'slices'

class Dist(str, Enum):
    nb = 'NB'
    betanb = 'BetaNB'
    cdnb = 'MCNB'
    betacdnb = 'BetaMCNB'

class WindowBehavior(str, Enum):
    both = 'both'
    right = 'right'

class Correction(str, Enum):
    none = 'none'
    hard = 'hard'
    single = 'single'

class Allele(str, Enum):
    ref = 'ref'
    alt = 'alt'

class Prior(str, Enum):
    laplace = 'laplace'
    normal = 'normal'

class DiffTest(str, Enum):
    lrt = 'lrt'
    wald = 'wald'
    
class RTransforms(str, Enum):
    none = 'none'
    NB = 'NB'
    mean = 'mean'

class SmallDatasetStrategy(str, Enum):
    none = 'none'
    conservative = 'conservative'
    fixed_r = 'fixed_r'
   
class ESMode(str, Enum):
    entropy = 'entropy'
    mean = 'mean'


class OrderCommands(TyperGroup):
  def list_commands(self, ctx: Context):
    """Return list of commands in the order appear."""
    return list(self.commands)    # get commands using self.commands

_DO_NOT_UPDATE_HISTORY = False

def update_history(name: str, command: str, **kwargs):
    if _DO_NOT_UPDATE_HISTORY:
        return
    try:
        with open(f'{name}.json', 'r') as f:
            d = json.load(f)
    except FileNotFoundError:
        d = dict()
    if command == 'create':
        d.clear()
        d['betanegbinfit'] = bnb_version
        d['jax'] = jax_version
        d['mixalime'] = mixalime_version
        d['scipy'] = scipy_version
        d['dill'] = dill_version
        d['name'] = name
    elif command == 'fit':
        for k in ('test', 'test_binom', 'difftest', 'combine', 'export', 'plot'):
            if k in d:
                del d[k]
        for k in list(d):
            if k.startswith('export'):
                del d[k]
    elif command.startswith('test'):
        for k in ('test', 'test_binom', 'combine', 'export', 'export pvalues', 'export raw_pvalues'):
            if k in d:
                del d[k]
    elif command == 'combine':
        if 'export' in d:
            del d['export']
        if 'export pvalues' in d:
            del d['export pvalues']
        subname = kwargs['subname']
        command = f'{command} {subname}' if subname else command
        for k in list(d):
            if k in ('export', command):
                del d[k]
            elif k.startswith('export pvalues '):
                t = k.split('export pvalues ')[-1]
                if subname == t:
                    del d[k]
        del kwargs['subname']
    elif command == 'difftest':
        if 'export' in d:
            del d['export']
        if 'export difftest' in d:
            del d['export pvalues']
        subname = kwargs['subname']
        command = f'{command} {subname}' if subname else command
        for k in list(d):
            if k in ('export', command):
                del d[k]
            elif k.startswith('export difftest '):
                t = k.split('export difftest ')[-1]
                if subname == t:
                    del d[k]
        del kwargs['subname']
    d[command] = kwargs
    with open(f'{name}.json', 'w') as f:
        json.dump(d, f, indent=4)

def reproduce(filename: str, pretty: bool = True, check_results: bool = True):
    global _DO_NOT_UPDATE_HISTORY
    _DO_NOT_UPDATE_HISTORY = True
    with open(filename, 'r') as f:
        d = json.load(f)
    name = d['name']
    for package in ('betanegbinfit', 'jax', 'scipy', 'mixalime', 'dill'):
        ver = d.get(package, None)
        if ver:
            curver = importlib.import_module(package).__version__
            if curver != ver:
                if pretty:
                    rprint(f'[yellow]Warning:[/yellow] Current [bold]{package}[/bold] version is {curver}, but the project was created with'
                          f' {ver}.')
                else:
                    print(f'Warning: Current {package} version is {curver}, but the project was created with {ver}.')
        else:
            if pretty:
                rprint(f'[yellow]Warning:[/yellow] No information on [bold]{package}[/bold] found in the history file.')
            else:
                print(f'Warning: No information on {package} found in the history file.')

    for command, args in d.items():
        if type(args) is str:
            continue
        r_old = args.get('expected_result', None)
        if r_old is not None:
            del args['expected_result']
        args['pretty'] = pretty
        if command == 'create':
            r = _create(name, **args)
        elif command == 'fit':
            r = _fit(name, **args)
        elif command == 'test':
            r = _test(name, **args)
        elif command == 'test_binom':
            r = _test_binom(name, **args)
        elif command.startswith('combine'):
            lt = command.split()
            if len(lt) != 1:
                command, subname = lt
                args['subname'] = subname
            else:
                args['subname'] = None
            r = _combine(name, **args)
        elif command.startswith('difftest'):
            lt = command.split()
            if len(lt) != 1:
                command, subname = lt
                args['subname'] = subname
            else:
                args['subname'] = None
            r = _difftest(name, **args)
        elif command == 'plot':
            r = _plot_all(name, **args)
        elif command.startswith('export'):
            lt = command.split()
            if len(lt) != 1:
                command, what = lt
                if what == 'params':
                    _params(name, **args)
                elif what == 'indices':
                    _indices(name, **args)
                elif what == 'counts':
                    _counts(name, **args)
                elif what == 'indices':
                    _indices(name, **args)
                elif what == 'pvalues':
                    _combined_pvalues(name, **args)
                elif what == 'raw_pvalues':
                    _raw_pvalues(name, **args)
                elif what == 'diftest':
                    _difftest(name, **args)
            else:
                r = _export_all(name, **args)
        if check_results and r_old and r != r_old:
            _DO_NOT_UPDATE_HISTORY = False
            raise Exception(f'"{name}" produced different result from what was recorded at the history file.')   
    _DO_NOT_UPDATE_HISTORY = False
    
    
doc = f'''
[bold]MixALime[/bold] version {mixalime_version}: [bold]M[/bold]ixture models for [bold]A[/bold]llelic [bold]I[/bold]mbalance [bold]E[/bold]stimation
\b\n
\b\n
A typical [bold]MixALime[/bold] session consists of sequential runs of [bold cyan]create[/bold cyan], [bold cyan]fit[/bold cyan], [bold cyan]test[/bold cyan], \
[bold cyan]combine[/bold cyan] and, finally, [bold cyan]export all[/bold cyan], [bold cyan]plot[/bold cyan] commands. For instance, we provide a demo \
dataset that consists of a bunch of BED-like files with allele counts at SNVs (just for the record, [bold]MixALime[/bold] can work with most vcf and \
BED-like file formats):
[magenta]>[/magenta] [cyan]mixalime export demo[/cyan]
A [i]scorefiles[/i] folder should appear now in a working directory with plenty of BED-like files.\
First, we'd like to parse those files into a [bold]MixALime[/bold]-friendly and efficient data structures for further usage, as well as perform some \
basic filtering if necessary:
[magenta]>[/magenta] [cyan]mixalime create myprojectname [i]scorefiles[/i] --no-snp-bad-check[/cyan]
Here, --no-snp-bad-check is necessarily for our demo: we used cancer data from the K562 cell line, and SNVs can indeed come from different BAD regions.
Then we fit model parameters to the data with Negative Binomial distribution:
[magenta]>[/magenta] [cyan]mixalime fit myprojectname NB[/cyan]
Next we obtain raw p-values:
[magenta]>[/magenta] [cyan]mixalime test myprojectname[/cyan]
Usually we'd want to combine p-values across samples and apply a FDR correction:
[magenta]>[/magenta] [cyan]mixalime combine myprojectname[/cyan]
Finally, we obtain fancy plots fir diagnostic purposes and easy-to-work with tabular data:
[magenta]>[/magenta] [cyan]mixalime export all myprojectname [i]results_folder[/i][/cyan]
[magenta]>[/magenta] [cyan]mixalime plot myprojectname [i]results_folder[/i][/cyan]
You'll find everything of interest in [i]results_folder[/i].\b\n
All commands have plenty of optional arguments, a much more detailed review on them can be obtained by further invoking [cyan]--help[/cyan] on them.
\b\n
If you found a bug or have any questions, feel free to contact us via
a) e-mail: [blue]iam@georgy.top[/blue] b) issue-tracker at [blue]github.com/autosome-ru/mixalime[/blue]
'''
app = Typer(rich_markup_mode='rich', cls=OrderCommands, add_completion=False, help=doc)

app_export = Typer(rich_markup_mode='rich', cls=OrderCommands, add_completion=False)
app.add_typer(app_export, name='export', help='Export tabulars obtained at previous steps (parameter estimates, fit indices, '
                                              ' count data, p-values).')

app_plot = Typer(rich_markup_mode='rich', cls=OrderCommands, add_completion=False)
app.add_typer(app_plot, name='plot', help='Commands to draw various diagnostic plots.')

help_str = 'Initialize [bold]MixALime[/bold] projects initial files: do parsing and filtering of VCFs/BEDs.'

@app.command('create', help=help_str)
def _create(name: str = Argument(..., help='Project name. [bold]MixALime[/bold] will produce files for internal usage that start with [cyan]'
                                            'name[/cyan].'),
            files: List[Path] = Argument(..., help='A list (if applicable, separated by space) of either filenames in VCF or BED-like format, '
                                                    'paths to folder with those files, paths to files that contain list of paths to those '
                                                    'files or  mask (masks should start with "[yellow]m:[/yellow]"'
                                                    'prefix, e.g. "m:vcfs/*_M_*.vcf.gz")  '),
            bad_maps: Path = Option(None, help='A path to an interval file that separates genome into BADs. The file must be tabular with '
                                              'columns "[bold]chr[/bold]", "[bold]start[/bold]", "[bold]end[/bold]", "[bold]bad[/bold]".'
                                              ' The "[bold]end[/bold]" column is optional, if it is not present, then the file is '
                                              'assumed to store precise BADs for each SNV.'),
            default_bad: float = Option(1.0, help='Those SNVs that are not present in the [cyan]bad_maps[/cyan]/in BED-like file will assume this'
                                                  ' BAD value.'),
            drop_bad: List[float] = Option(None, '--drop-bad', '-d', help='Those BADs and their respective SNVs will be ommited.'),
            snp_bad_check: bool = Option(True, help='Require SNVs to come from the same BAD.'),
            min_qual: int = Option(10, help='Minimal SNV quality'),
            min_cnt: int = Option(5, help='Minimal allowed number of counts at an allele.'),
            max_cover: int = Option(None, help='Maximal allowed total counts (ref + alt) for an SNV.'),
            symmetrify: bool = Option(False, help='Counts are symmetrified, i.e. (ref, alt) = (alt, ref) for each pair of ref, alt. It is done by'
                                                  ' summing (ref, alt) + (alt, ref) and dividing the result by 2.'),
            filter_db: bool = Option(False, help='Omit SNVs that are not present in DB.'),
            filter_rs: bool = Option(False, help='Omit SNVs whose IDs don''t start with "[bold]rs[/bold]".'),
            filter_name: str = Option(None, help='Custom regex-compatible pattern: SNVs whose IDs are not in an agreement with the pattern, '
                                                  'shall be omitted.'), 
            filter_chr: str = Option(None, help='Custom regex-compatible pattern for chr filtering. Possible applicaitons include omitting scaffold '
                                                'data.'),
            compression: Compression = Option(Compression.lzma.value, help='Compression method used to store results.'), 
            pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    if max_cover is None:
        max_cover = float('inf')
    if type(compression) is Compression:
        compression = str(compression.value)
    files = list(map(str, files))
    if bad_maps:
        bad_maps = str(bad_maps)
    t0 = time()
    if not pretty:
        print('Processing files...')
    _, samples, snvs = create_project(name=name, snvs=files, bad_maps=bad_maps, default_bad=default_bad, drop_bads=drop_bad, min_qual=min_qual,
                                      min_cnt=min_cnt, max_cover=max_cover, filter_db=filter_db, filter_rs=filter_rs, symmetrify=symmetrify,
                                      filter_name=filter_name, filter_chr=filter_chr, compression=compression, count_snvs=True, 
                                      snp_bad_check=snp_bad_check, progress_bar=pretty)
    rows = list()
    tot_snv = 0
    tot_samples = 0
    for bad in sorted(samples):
        a = len(snvs[bad])
        b = samples[bad]
        rows.append(['{:.2f}'.format(bad), str(a), str(b)])
        tot_snv += a
        tot_samples += b
    if 0 in samples:
        if pretty:
            rprint('[yellow]Warning:[/yellow] BAD=0 is present in the data. Please remove BAD=0 occurrences or substitute them with BAD=1.')
        else:
            print('Warning: BAD=0 is present in the data. Please remove BAD=0 occurrences or substitute them with BAD=1.')
    rows = rows
    if pretty:
        table = Table('BAD', 'SNVs', 'Obvservations')
        for row in rows:
            table.add_row(*row)
        rprint(table)
        rprint(f'Total unique SNVs: {tot_snv}, total observations: {tot_samples}')
    else:
        print('BAD\tSNVs\tObvservations')
        print('\n'.join(['\t'.join(row) for row in rows]))     
        print(f'Total unique SNVs: {tot_snv}, total observations: {tot_samples}')    
    update_history(name, 'create', files=files, bad_maps=bad_maps, default_bad=default_bad, drop_bad=drop_bad, min_qual=min_qual,
                   min_cnt=min_cnt, max_cover=max_cover, filter_db=filter_db, filter_rs=filter_rs, symmetrify=symmetrify,
                   filter_name=filter_name, filter_chr=filter_chr, compression=compression, snp_bad_check=snp_bad_check,
                   expected_result=rows)
    dt = time() - t0
    if pretty:
        rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')
    else:
        print(f'✔️ Done!\t time: {dt:.2f} s.')
    return rows

@app.command('fit')
def _fit(name: str = Argument(..., help='Project name.'),
         dist: Dist = Argument(..., help='Name of distribution that will be used in the mixture model.'),
         left: int = Option(None, help='Left-truncation bound. If None, then it will be estimated from the counts data as the minimal present count minus 1.'),
         estimate_p: bool = Option(False, help='If True, then p will be estimated instead of assuming it to be fixed to bad / (bad + 1)'),
         window_size: int = Option(10000, help='Sets the required minimal window size.'),
         window_behavior: WindowBehavior = Option(WindowBehavior.both.value, help='If "[bold]both[/bold]", then window is expanded in 2 directions. '
                                                                                  'If "[right]right[/right]", then the window is expanded only to'
                                                                                  ' the right (except for cases when such an expansion is not'
                                                                                  ' possible but the minimal slice or window size requirement is not '
                                                                                  'met).'),
         min_slices: int = Option(3, help='Minimal number of slices per window.'),
         stop_slice_n: int = Option(10, help='Parameters are estimated as long as windows center slice has more than [cyan]stop-slice-n[/cyan]'
                                              ' number of samples.'),
         adjust_line: bool = Option(False, help='Line parameter beta and mu will be reestimated without a loss of likelihood so they differ'
                                                ' as little as possible from the previous b and mu estimates.'),
         k_left_bound: int = Option(0, help='Minimal allowed value for concentration parameter [italic]k[/italic].'),
         max_count: int = Option(None, help='Maximal number of counts for an allele (be it ref or alt).'),
         max_cover: int = Option(None, help='Maximal sum of ref + alt.'),
         regul: bool = Option(False, help='Apply regularization/encourage high kappa parameter estimates. Valid only for [cyan]dist[/cyan]='
                                               '[yellow]BetaNB[/yellow].'),
         regul_alpha: float = Option(1.0, help='Regularization/prior strength hyperparameter alpha. Valid only for [cyan]dist[/cyan]='
                                               '[yellow]BetaNB[/yellow].'),
         regul_n: bool = Option(True, help='Multiply alpha by a number of SNV records captured by a particular window. Valid only for [cyan]dist[/cyan]='
                                               '[yellow]BetaNB[/yellow].'),
         regul_slice: bool = Option(True, help='Multiply alpha by a an average slice captured by a particular window. Valid only for [cyan]dist[/cyan]='
                                               '[yellow]BetaNB[/yellow].'),
         regul_prior: Prior = Option('laplace', help='Prior distribution used to penalize concentration parameter kappa. Valid only for [cyan]dist[/cyan]='
                                               '[yellow]BetaNB[/yellow].'),
         std : bool = Option(False, help='Compute standard errors for parameter estimates. Note that it may significantly increase computation'
                                         ' time.'),
         fix_params : str = Option(None, help='Parameters that are not estimated, but fixed to a constant float argument instead. This argument '
                                             'accepts parameters in the form of "[cyan]param_name_1[/cyan]=[cyan]val1[/cyan];'
                                             '[cyan]param_name_2[/cyan]=[cyan]val2[/cyan]. For instance, one might be interested in the so-called'
                                             ' conservative scoring "w=1;mu=0;b=1" that is of interest when data is scarce.'),
         adjusted_loglik: bool = Option(False, help='Calculate adjusted loglikelihood alongside other statistics.'),
         optimizer: str = Option('SLSQP', help='Name of [bold]scipy[/bold]"s optimization method'),
         r_transform: RTransforms = Option('NB', help='[red]r[/red] parameter will be reparameterized so the mean of a [cyan]DIST[/cyan] will '
                                                      'agree with either a Negative Binomial mean, or the mean itself will be equal to the '
                                                      '[red]r[/red] itself.'),
         symmetrify: bool = Option(False, help='Symmetrifies count data before fitting model to it. Might be helpful in cases when you know'
                                                ' that ref|alt model should be equal to alt|ref, e.g. when fitting the model to post-WASP data.'),
         small_dataset_n: int = Option(10000, help='If a number of SNV records at BAD is below this threshold, [yellow]small_dataset_strategy[/yellow]'
                                                   ' will be applied.'),
         small_dataset_strategy: SmallDatasetStrategy = Option('conservative', help='[yellow]fixed_r[/yellow] constraints r as it is constrained in'
                                                                                    ' [yellow]conservative[/yellow] mode.'),
         kappa_right_boundary: float = Option(10000, help='Right boundary for the kappa parameter. Applicable only for the BetaNB distribution.'),
         n_jobs: int = Option(-1, help='Number of jobs to be run at parallel, -1 will use all available threads.'),
         pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Fit a a mixture model parameters to data for the given project.
    """
    
    start_est = True
    apply_weights = False
    model = Model.window.value
    if max_count is None:
        max_count = float('inf')
    if max_cover is None:
        max_cover = float('inf')
    if type(dist) is Dist:
        dist = dist.value
    if type(model) is Model:
        model = model.value
    if type(window_behavior) is WindowBehavior:
        window_behavior = window_behavior.value
    if type(regul_prior) is Prior:
        regul_prior = regul_prior.value
    if type(r_transform) is RTransforms:
        r_transform = r_transform.value
    if type(small_dataset_strategy) is SmallDatasetStrategy:
        small_dataset_strategy = small_dataset_strategy.value
    if fix_params is None:
        fix_params = str()
    t0 = time()
    if pretty:
        p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
        p.add_task(description="Optimizing model parameters...", total=None)
        p.start()
    else:
        print('Optimizing model parameters...')
    fit(name, dist=dist, model=model, left=left, estimate_p=estimate_p, window_size=window_size, 
        window_behavior=window_behavior, min_slices=min_slices, adjust_line=adjust_line, k_left_bound=k_left_bound,
        max_count=max_count, max_cover=max_cover, adjusted_loglik=adjusted_loglik, n_jobs=n_jobs, start_est=start_est,
        apply_weights=apply_weights, regul_alpha=regul_alpha if regul else 0, regul_n=regul_n, regul_slice=regul_slice, regul_prior=regul_prior,
        fix_params=fix_params, std=std, optimizer=optimizer, r_transform=None if r_transform == 'none' else r_transform,
        symmetrify=symmetrify, small_dataset_strategy=small_dataset_strategy, small_dataset_n=small_dataset_n,
        stop_slice_n=stop_slice_n)
    if pretty:
        p.stop()
    update_history(name, 'fit', dist=dist, left=left, estimate_p=estimate_p, window_size=window_size, 
                   window_behavior=window_behavior, min_slices=min_slices, adjust_line=adjust_line, k_left_bound=k_left_bound,
                   max_count=max_count, max_cover=max_cover, adjusted_loglik=adjusted_loglik, n_jobs=n_jobs, 
                   regul=regul, regul_alpha=regul_alpha, regul_n=regul_n, regul_slice=regul_slice, regul_prior=regul_prior,
                   fix_params=fix_params, std=std, optimizer=optimizer, r_transform=r_transform, symmetrify=symmetrify,
                   small_dataset_strategy=small_dataset_strategy, small_dataset_n=small_dataset_n, stop_slice_n=stop_slice_n)
    dt = time() - t0
    if pretty:
        rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')
    else:
        print(f'✔️ Done!\t time: {dt:.2f} s.')

@app.command('test')
def _test(name: str = Argument(..., help='Project name.'),
          fit: str = Option(None, help='Path to a fit file from a different project. If not provided, fit from the same project is used.'),
          correction: Correction = Option(Correction.single.value, help='Posterior weight correction method. It effectively helps to choose'
                                                                      ' a particular component of a distribution for further tests, neglecting '
                                                                      ' an impact of more distant component.'),
          gof_thr: float = Option(None, help='Conservative scoring will be used if goodness-of-fit statistic (RMSEA) exceeds [cyan]gof-thr[/cyan] '
                                                'for a particular slice.'),
          stop_slice_n_thr: int = Option(-1, help='Conservative scoring will be used if number of samples at a slice is below '
                                                  '[cyan]dataset-n-thr[/cyan].'),
          n_jobs: int = Option(1, help='Number of jobs to be run at parallel, -1 will use all available threads.'),
          pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Calculate p-values using parameter estimates obtained after [cyan bold]fit[/cyan bold] command.
    """
    if type(correction) is Correction:
        correction = correction.value
    t0 = time()
    if pretty:
        p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
        p.add_task(description="Comptuing p-values and effect sizes...", total=None)
        p.start()
    else:
        print('Computing p-values and effect sizes...')
    test(name, correction=correction, gof_tr=gof_thr, fit=fit, dataset_n_thr=stop_slice_n_thr, n_jobs=n_jobs)
    if pretty:
        p.stop()
    update_history(name, 'test', correction=correction, gof_thr=gof_thr, fit=fit, 
                   stop_slice_n_thr=stop_slice_n_thr, n_jobs=n_jobs)
    dt = time() - t0
    if pretty:
        rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')
    else:
        print(f'✔️ Done!\t time: {dt:.2f} s.')

@app.command('combine')
def _combine(name: str = Argument(..., help='Project name.'),
             group: List[Path] = Option(None, '--group', '-g', help='A list of either filenames (vcf or BAM-like tabulars) or folders that contain '
                                                                      'those filenames or file(s) that contain a list of paths to files, or a mask'
                                                                      ' (masks should start with "[yellow]m:[/yellow]"'
                                                                      ' prefix, e.g. "m:vcfs/*_M_*.vcf.gz"). '
                                                                      ' SNV p-values from those files shall be combined via logit method.'),
             alpha: float = Option(0.05, help='FWER, family-wise error rate.'),
             min_cover: int = Option(20, help='If none of combined p-values is associated with a sample whose ref + alt exceeds'
                                              ' [cyan]min_cover[/cyan], the SNV is omitted.'),
             adaptive_min_cover: bool = Option(False, help='Use adaptive [cyan]min_cover[/cyan] for each BAD. The algorithm has two hyperparameters:'
                                               '[cyan]adaptive_es[/cyan] and [cyan]adaptive_pval[/cyan]. The minimal coverage where effect-size of'
                                               'at least [cyan]adative_es[/cyan] is achievable for a p-value of [cyan]adaptive_pval[/cyan].'),
             adaptive_es: float = Option(1.0, help='Minimal required effect-size for the adaptive coverage algorithm.'),
             adaptive_pval: float = Option(0.05, help='Minimal required p-value for the adaptive coverage algorithm.'),
             uniform_weights: bool = Option(True, help='Uniform weighing for effect-sizes.'),
             filter_id: str = Option(None, help='Only SNVs whose IDs agree with this regex pattern are tested (e.g. "rs\w+").'),
             filter_chr: str = Option(None, help='SNVs with chr that does not align with this regex pattern are filtered (e.g. "chr\d+").'),
             subname: str = Option(None, help='You may give a result a subname in case you plan to use multiple groups.'),
             n_jobs: int = Option(1, help='Number of jobs to be run at parallel, -1 will use all available threads.'),
             pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Combine p-values obtained at [cyan bold]test[/cyan bold] stage with a Mudholkar-George method.
    """
    if group:
        group = list(map(str, group))
    t0 = time()
    if pretty:
        p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
        p.add_task(description="Combining p-values with respect to sample groups...", total=None)
        p.start()
    else:
        print('Combining p-values with respect to sample groups...')
    if subname:
        subname = str(subname)
    else:
        subname = None
    r, adaptive_coverage = combine(name, group_files=group if group else [r'm:*'], alpha=alpha, filter_id=filter_id, filter_chr=filter_chr,
                                   subname=subname, min_cnt_sum=min_cover, adaptive_min_cover=adaptive_min_cover,
                                   adaptive_es=adaptive_es, adaptive_pval=adaptive_pval,
                                   uniform_weights=uniform_weights, n_jobs=n_jobs)
    r = r[subname]['snvs']
    if pretty:
        p.stop()
    ref = alt = both = total = 0
    for t in r.values():
        pv_ref, pv_alt = t[-1]
        ref += pv_ref < alpha
        alt += pv_alt < alpha
        total += (pv_ref < alpha) | (pv_alt < alpha)
        both += (pv_ref < alpha) & (pv_alt < alpha)
    expected_res = [int(ref), int(alt), int(both)]
    if pretty:
        if adaptive_min_cover:
            rprint('Minimal required coverages for each BAD:')
            table = Table('BAD', 'Ref', 'Alt', 'Min')
            for bad in sorted(adaptive_coverage['ref']):
                cov_ref = adaptive_coverage['ref'][bad]
                cov_alt = adaptive_coverage['alt'][bad]
                table.add_row(*map(str, (bad, cov_ref, cov_alt, min(cov_ref, cov_alt))))
            rprint(table)
        rprint('Number of significantly imbalanced SNVs after FDR correction:')
        table = Table('Ref', 'Alt', 'Both', 'Total significant\n(Percentage of total SNVs)')
        table.add_row(str(ref), str(alt), str(both), f'{total} ({total/len(r) * 100:.2f}%)')
        rprint(table)
        rprint(f'Total SNVs tested: {len(r)}')
    else:
        if adaptive_min_cover:
            print('Minimal required coverages for each BAD:')
            table = print('\t'.join(('BAD', 'Ref', 'Alt', 'Min')))
            
            for bad in sorted(adaptive_coverage['ref']):
                cov_ref = adaptive_coverage['ref'][bad]
                cov_alt = adaptive_coverage['alt'][bad]
                print('\t'.join(map(str, (bad, cov_ref, cov_alt, min(cov_ref, cov_alt)))))
        print('Number of significantly imbalanced SNVs after FDR correction:')
        print('\t'.join(('Ref', 'Alt', 'Both', 'Total significant (Percentage of total SNVs)')))
        print('\t'.join((str(ref), str(alt), str(both), f'{total} ({total/len(r) * 100:.2f}%)')))
        print(f'Total SNVs tested: {len(r)}')
    update_history(name, 'combine', group=group, alpha=alpha, min_cover=min_cover, adaptive_min_cover=adaptive_min_cover,
                   adaptive_es=adaptive_es, adaptive_pval=adaptive_pval, filter_id=filter_id, subname=subname, filter_chr=filter_chr,
                   uniform_weights=uniform_weights, n_jobs=n_jobs, expected_result=expected_res)
    dt = time() - t0
    if pretty:
        rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')
    else:
        print(f'✔️ Done!\t time: {dt:.2f} s.')
        
    return expected_res


@app.command('difftest')
def _difftest(name: str = Argument(..., help='Project name.'),
              fit: str = Option(None, help='Path to a fit file from a different project. If not provided, fit from the same project is used.'),
              group_control: Path = Argument(..., help='A file with a list of filenames, folder or a mask (masks should start with "[yellow]m:[/yellow]"'
                                                 'prefix, e.g. "m:vcfs/*_M_*.vcf.gz") for the first group.'),
              group_test: Path = Argument(..., help='A file with a list of filenames, folder or a mask (masks should start with "[yellow]m:[/yellow]"'
                                                 'prefix, e.g. "m:vcfs/*_M_*.vcf.gz") for the second group.'),
              mode: DiffTest = Option(DiffTest.wald.value, help='Test method.'),
              param_window: bool = Option(True, help='If disabled, parameters will be taken from a line with respect to the mean window for given'
                                                     ' reps/samples.'),
              robust_se: bool = Option(False, help='Use robust standard errors (Huber-White Sandwich correction). Applicable only if '
                                                   '[cyan]--mode[/cyan]=[yellow]wald[/yellow].'),
              n_bootstrap: int = Option(0, help='Boostrap iterations used in stochastic bias correction. Applicable only if [cyan]--mode[/cyan]=='
                                                '[yellow]wald[/yellow].'),
              logit_transform: bool = Option(False, help='Apply logit transform to [bold]p[/bold] and its variance with Delta method. Applicable '
                                                         'only if [cyan]--mode[/cyan]=[yellow]wald[/yellow].'),
              test_groups: bool = Option(False, help='Whole groups will be tested against each other first. Note that this will take'
                                                    ' the same time as [cyan]fit[/cyan] stage.'),
              contrasts: Tuple[float, float, float] = Option((1, -1, 0), help='Contrasts vector where 1st, 2nd positions are for groups A and B '
                                                                              'respectively and the 3rd stand for the free term, i.e. the default'
                                                                              ' value will test for difference between p_a and p_b, but '
                                                                              '[bold]1 0 -0.5[/bold] will test only for p_a being equal to 0.5. '),
              alpha: float = Option(0.05, help='FWER, family-wise error rate.'),
              min_samples: int = Option(2, help='Minimal number of samples/reps per an SNV to be considered for the analysis.'),
              min_cover: int = Option(None, help='Minimal required cover (ref + alt) for an SNV to be considered.'),
              max_cover: int = Option(None, help='Maximal allowed cover (ref + alt) for an SNV to be considered.'),
              max_cover_group_test: int = Option(None, help='Maximal allowed cover (ref + alt) for an SNV to be considered. Used only to trim'
                                                            ' for the whole group test if applicable (i.e. if [cyan]group_test[/cyan]) to'
                                                            'avoid long waiting times similarily to [cyan]fit[/cyan].'),
              filter_id: str = Option(None, help='Only SNVs whose IDs agree with this regex pattern are tested (e.g. "rs\w+").'),
              filter_chr: str = Option(None, help='SNVs with chr that does not align with this regex pattern are filtered (e.g. "chr\d+").'),
              subname: str = Option(None, help='You may give a result a subname in case you plan to draw multiple comparisons.'),
              n_jobs: int = Option(1, help='Number of jobs to be run at parallel, -1 will use all available threads.'),
              pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Differential expression tests via Wald/LR tests.
    """
    t0 = time()
    group_a = str(group_control)
    group_b = str(group_test)
    contrasts = list(contrasts)
    if type(mode) is DiffTest:
        mode = mode.value
    if pretty:
        p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
        p.add_task(description='Performing {} tests...'.format('LRT' if mode == 'lrt' else 'Wald'), total=None)
        p.start()
    else:
        print('Performing {} tests...'.format('LRT' if mode == 'lrt' else 'Wald'))
    if subname:
        subname = str(subname)
    else:
        subname = None
    r = differential_test(name, group_a=group_a, group_b=group_b, mode=mode, min_samples=min_samples, min_cover=min_cover,
                          max_cover=max_cover, group_test=test_groups, subname=subname,  filter_id=filter_id,
                          max_cover_group_test=max_cover_group_test, filter_chr=filter_chr, alpha=alpha, n_jobs=n_jobs,
                          param_mode='window' if param_window else 'line', logit_transform=logit_transform,
                          robust_se=robust_se, contrasts=contrasts, n_bootstrap=n_bootstrap, fit=fit)[subname]
    if pretty:
        p.stop()
    if test_groups:
        if pretty:
            rprint('Group A vs Group B:')
            rprint(r['whole'])
        else:
            print('Group A vs Group B:')
            print(r['whole'])
    
    r = r['tests']
    ref = r['ref_fdr_pval'] < alpha
    alt = r['alt_fdr_pval'] < alpha
    both = (ref & alt).sum()
    total = (ref | alt).sum()
    ref = ref.sum()
    alt = alt.sum()
    if pretty:
        if not total:
            rprint('No SNVs passed filters for differential test.')
        else:
            rprint('Number of significantly differentially expressed SNVs after FDR correction:')
            table = Table('Ref', 'Alt', 'Both', 'Total\nPercentage of total SNVs ')
            table.add_row(str(ref), str(alt), str(both), f'{total} ({total/len(r) * 100:.2f}%)')
            rprint(table)
            rprint('Total SNVs tested:', len(r))
    else:
        if not total:
            print('No SNVs passed filters for differential test.')
        else:
            print('Number of significantly differentially expressed SNVs after FDR correction:')
            print('\t'.join('Ref', 'Alt', 'Both', 'Total/Percentage of total SNVs'))
            print('\t'.join((str(ref), str(alt), str(both), f'{total} ({total/len(r) * 100:.2f}%)')))
            print('Total SNVs tested:', len(r))
    expected_res = [int(ref), int(alt), int(total)]
    update_history(name, 'difftest', group_control=group_a, group_test=group_b, alpha=alpha, min_samples=min_samples, min_cover=min_cover,
                   mode=mode, subname=subname, test_groups=test_groups, max_cover=max_cover, filter_id=filter_id,
                   filter_chr=filter_chr, max_cover_group_test=max_cover_group_test, n_jobs=n_jobs,
                   param_window=param_window, logit_transform=logit_transform, robust_se=robust_se,
                   contrasts=contrasts, n_bootstrap=n_bootstrap, expected_result=expected_res, fit=fit)
    dt = time() - t0
    if pretty:
        rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')
    else:
        print(f'✔️ Done!\t time: {dt:.2f} s.')
    return expected_res

@app.command('anova')
def _anova(name: str = Argument(..., help='Project name.'),
           groups: str = Option(str(), help='Files with lists of filenames, folder or a mask (masks should start with "[yellow]m:[/yellow]"'
                                             'prefix, e.g. "m:vcfs/*_M_*.vcf.gz"), each for a group. Entries must be separated with a semicolon ";" character.'
                                             'If not provided, then it is assumed that [b]combine[/b] was called previously with '
                                             '[yellow]subname[/yellow]s corresponding to group names (recommended).'),
           fit: str = Option(None, help='Path to a fit file from a different project. If not provided, fit from the same project is used.'),
           param_window: bool = Option(True, help='If disabled, parameters will be taken from a line with respect to the mean window for given'
                                                     ' reps/samples.'),
           es_mode: ESMode = Option(ESMode.entropy.value, help='Effect-size calculation method.'),
           alpha: float = Option(0.05, help='FWER, family-wise error rate.'),
           min_samples: int = Option(2, help='Minimal number of samples/reps per an SNV to be considered for the analysis.'),
           min_groups: int = Option(2, help='Minimal number of groups present for an SNV to be considered for the analysis.'),
           min_cover: int = Option(None, help='Minimal required cover (ref + alt) for an SNV to be considered.'),
           max_cover: int = Option(None, help='Maximal allowed cover (ref + alt) for an SNV to be considered.'),
           subname: str = Option(None, help='You may give a result a subname in case you plan to draw multiple comparisons.'),
           n_jobs: int = Option(1, help='Number of jobs to be run at parallel, -1 will use all available threads.'),
           pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    ANOVA test via likelihood ratio.
    """
    t0 = time()
    groups = str(groups)
    if type(es_mode) is ESMode:
        es_mode = es_mode.value
    if pretty:
        p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
        p.add_task(description='Performing ANOVA tests...', total=None)
        p.start()
    else:
        print('Performing ANOVA tests...')
    if subname:
        subname = str(subname)
    else:
        subname = None
    r = anova_test(name, groups=groups.split(';') if groups else None, min_samples=min_samples, min_cover=min_cover,
                   max_cover=max_cover, alpha=alpha, n_jobs=n_jobs, min_groups=min_groups, es_mode=es_mode,
                   param_mode='window' if param_window else 'line', fit=fit)[subname]
    if pretty:
        p.stop()
    
    r = r['tests']
    ref = r['ref_fdr_pval'] < alpha
    alt = r['alt_fdr_pval'] < alpha
    both = (ref & alt).sum()
    total = (ref | alt).sum()
    ref = ref.sum()
    alt = alt.sum()
    if pretty:
        if not total:
            rprint('No SNVs passed filters for ANOVA test.')
        else:
            rprint('Number of significantly differentially expressed SNVs after FDR correction:')
            table = Table('Ref', 'Alt', 'Both', 'Total\nPercentage of total SNVs ')
            table.add_row(str(ref), str(alt), str(both), f'{total} ({total/len(r) * 100:.2f}%)')
            rprint(table)
            rprint('Total SNVs tested:', len(r))
    else:
        if not total:
            print('No SNVs passed filters for differential test.')
        else:
            print('Number of significantly differentially expressed SNVs after FDR correction:')
            print('\t'.join('Ref', 'Alt', 'Both', 'Total/Percentage of total SNVs'))
            print('\t'.join((str(ref), str(alt), str(both), f'{total} ({total/len(r) * 100:.2f}%)')))
            print('Total SNVs tested:', len(r))
    expected_res = [int(ref), int(alt), int(total)]
    update_history(name, 'anova', groups=groups, min_samples=min_samples, min_cover=min_cover,
                          max_cover=max_cover, alpha=alpha, n_jobs=n_jobs, min_groups=min_groups,
                          param_window=param_window, es_mode=es_mode, fit=fit)
    dt = time() - t0
    if pretty:
        rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')
    else:
        print(f'✔️ Done!\t time: {dt:.2f} s.')
    return expected_res

@app_plot.command('anova', help='Visualize effect-sizes for some SNVs given ANOVA results.')
def _plot_anova(name: str = Argument(..., help='Project name.'),
                folder: Path = Argument(..., help='Output folder where figures will be stored'),
                ids: str = Option(None, help='IDs+alt alleles of SNVs to draw separated by ";" character, e.g. "rs123,C;rs124,A".'),
                snvs: str = Option(None, help='Chromosome+pos+alt allele of SNVs to draw separated by "," character. Multiple SNVs can be'
                                              ' provided by separating them with ";" character, e.g.: "chr_1,10,A;chr_3,5,G".'),
                subname: str = Option(None, help='Subname to use.'),
                plot_raw_es: bool = Option(True, help='Draw effect-sizes estimated as a mean of ESes from the raw data.'),
                plot_test_es: bool = Option(False, help='Draw effect-sizes estimated as a mean of ESes from the test run.'),
                plot_p_diff: bool = Option(True, help='Draw difference between estimated p between nested models and the joint model.'),
                color_significant: bool = Option(True, help='Bars corresponding to groups that have passed the 0.05 threshold for the'
                                                             ' pairwise LRT test will be colored in green.'),
                dpi: int = Option(200, help='DPI.'),
                fmt: str = Option('png', help='Image format.'),
                pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Plots effect-sizes for give SNVs given previously ran ANOVA call.
    """
    t0 = time()
    if pretty:
        p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
        p.add_task(description='Plotting SNVs ESes...', total=None)
        p.start()
    else:
        print('Plotting SNVs ESes..')
    if ids is None and snvs is None:
        raise SyntaxError("At least ids or snvs should be provided.")
    if ids:
        ids = [tuple(t.split(',')) for t in ids.split(';')]
    if snvs:
        snvs_t = list()
        for t in snvs.split(';'):
            t = t.split(',')
            snvs_t.append((t[0], int(t[1]), t[2]))
        snvs = snvs_t
    plot.plot_anova_snvs(name, snv_names=ids, snvs=snvs, subname=subname, plot_raw_es=plot_raw_es, plot_test_es=plot_test_es,
                         plot_p_diff=plot_p_diff, color_significant=color_significant, folder=folder, ext=fmt, 
                         dpi=dpi)
    dt = time() - t0
    if pretty:
        rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')
    else:
        print(f'✔️ Done!\t time: {dt:.2f} s.')

@app_export.command('all', help='Export everything.')
def _export_all(name: str = Argument(..., help='Project name.'), out: Path = Argument(..., help='Output filename/path.'),
                sample_info: bool = Option(False, help='Include raw p-values and names of sample scorefiles to the tabular. '
                                                       'Note that this may bloat output if number of samples is high.'),
                pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    out = str(out)
    t0 = time()
    if pretty:
        p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
        p.add_task(description="Exporting tabular data...", total=None)
        p.start()
    export.export_all(name, out, sample_info=sample_info)
    if pretty:
        p.stop()
    update_history(name, 'export', out=out, sample_info=sample_info)
    dt = time() - t0
    if pretty:
        rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')
    else:
        print(f'✔️ Done!\t time: {dt:.2f} s.')

@app_export.command('counts')
def _counts(name: str = Argument(..., help='Project name.'), out: str = Argument(..., help='Output filename/path.'),
            bad: float = Option(None, help='If provided, then only this particular BAD will be exported. In that case, [cyan]out[/cyan] is a '
                                           'filename, not a name of directory. Otherwise, counts from all bads will be exported to the'
                                           ' [cyan]out[/cyan] path.'),
            pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Export counts data.
    """
    export.export_counts(name, out, bad=bad)
    if pretty:
        rprint('[green][bold]✔️[/bold] Done![/green]')
    else:
        print('✔️ Done!')

@app_export.command('params')
def _params(name: str = Argument(..., help='Project name.'), out: str = Argument(..., help='Output filename/path.'),
            bad: float = Option(None, help='If provided, then only this particular BAD will be exported. In that case, [cyan]out[/cyan] is a '
                                           'filename, not a name of directory. Otherwise, params obtained from all BADs will be exported to the'
                                           ' [cyan]out[/cyan] path.'),
            allele: Allele = Option(None, help='Allele name (remember that we fit 2 separate models with respect to [yellow]ref[/yellow]'
                                               ' and [yellow]alt[/yellow] alleles). Must be provided as well if [cyan]bad[/cyan] is provided.'),
            pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Export parameters estimates.
    """
    if type(allele) is Allele:
        allele = allele.value
    export.export_params(name, out, bad=bad, allele=allele)
    update_history(name, 'export params', out=out, bad=bad, allele=allele)
    if pretty:
        rprint('[green][bold]✔️[/bold] Done![/green]')
    else:
        print('✔️ Done!')


@app_export.command('indices')
def _indices(name: str = Argument(..., help='Project name.'), out: Path = Argument(..., help='Output filename/path.'),
             bad: float = Option(None, help='If provided, then only this particular BAD will be exported. In that case, [cyan]out[/cyan] is a '
                                            'filename, not a name of directory. Otherwise, indices obtained from all BADs will be exported to the'
                                            ' [cyan]out[/cyan] path.'),
             allele: Allele = Option(None, help='Allele name (remember that we fit 2 separate models with respect to [yellow]ref[/yellow]'
                                                ' and [yellow]alt[/yellow] alleles). Must be provided as well if [cyan]bad[/cyan] is provided.'),
             pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Export fit indices.
    """
    out = str(out)
    if type(allele) is Allele:
        allele = allele.value
    export.export_stats(name, out, bad=bad, allele=allele)
    update_history(name, 'export indices', out=out, bad=bad, allele=allele)
    if pretty:
        rprint('[green][bold]✔️[/bold] Done![/green]')
    else:
        print('✔️ Done!')


@app_export.command('pvalues')
def _combined_pvalues(name: str = Argument(..., help='Project name.'), out: Path = Argument(..., help='Output filename/path.'),
                      sample_info: bool = Option(False, help='Include raw p-values and names of sample scorefiles to the tabular. '
                                                          'Note that this may bloat output if number of samples is high.'),
                      subname: str = Option(None, help='A subname that can be used to reference a set of combined p-values in case if you'
                                                       ' provided one at [cyan bold]combine[/cyan bold] step.'),
                      pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    '''
    Export combined across samples and FDR-corrected p-values.
    '''
    out = str(out)
    export.export_combined_pvalues(name, out, sample_info=sample_info, subname=subname)
    update_history(name, 'export pvalues', out=out, sample_info=sample_info, subname=subname)
    if pretty:
        rprint('[green][bold]✔️[/bold] Done![/green]')
    else:
        print('✔️ Done!')


help_str = 'Export "raw" pvalues (i.e. prior to combining them across samples and FDR-corrections). '\
           'Note that for each vcf/BED-like file submitted at [cyan bold]create[/cyan bold] stage, this command will '\
           'spawn a tabular file with similar name (including a relative path).'
@app_export.command('raw_pvalues', help=help_str)
def _raw_pvalues(name: str, out: Path,
                 pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    out = str(out)
    export.export_pvalues(name, out)
    update_history(name, 'export raw-pvalues', out=out)
    if pretty:
        rprint('[green][bold]✔️[/bold] Done![/green]')
    else:
        print('✔️ Done!')


@app_export.command('difftest')
def _difftests(name: str = Argument(..., help='Project name.'), out: Path = Argument(..., help='Output filename/path.'),
                      subname: str = Option(None, help='A subname that can be used to reference a set of combined p-values in case if you'
                                                       ' provided one at [cyan bold]difftest[/cyan bold] step.'),
                      sample_info: bool = Option(False, help='Include ref, alt counts and names of sample scorefiles to the tabular. '
                                                          'Note that this may bloat output if number of samples is high.'),
                      pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    '''
    Export FDR-corrected p-values for differential test.
    '''
    out = str(out)
    export.export_difftests(name, out, subname=subname, sample_info=sample_info)
    update_history(name, 'export difftest', out=out, subname=subname)
    if pretty:
        rprint('[green][bold]✔️[/bold] Done![/green]')
    else:
        print('✔️ Done!')

@app_export.command('demo')
def _demo(export_path: Path = Option(str(), help='Path where the demo data will be extracted.'),
          pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    '''
    Extract demonstration data from DNase-seq experiment for the cell line K562.
    '''
    export_path = str(export_path)
    export.export_demo(export_path)
    if pretty:
        rprint('[green][bold]✔️[/bold] Done![/green]')
    else:
        print('✔️ Done!')
    

@app_plot.command('all', help='Visualize model fits and plot model fit indices for diagnostic purposes.')
def _plot_all(name: str = Argument(..., help='Project name.'), out: Path = Argument(..., help='Output filename/path.'),
              max_count: int = Option(60, help='Maximal read counts at an allele.'),
              slices: List[int] = Option([5, 10, 20, 30, 40], '--slices', '-s', help='List of slices for plotting sliceplots.'),
              bad: float = Option(None, help='Draw plots only for this particular BAD.'),
              show_bad: bool = Option(True, help='Show BADs in figures'' titles.'),
              dpi: int = Option(200, help='DPI.'),
              fmt: str = Option('png', help='Image format.'),
              pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    out = str(out)
    t0 = time()
    if pretty:
        p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
        p.add_task(description="Plotting figures...", total=None)
        p.start()
    else:
        print('Plotting figures...')
    plot.visualize(name, out, what='all', max_count=max_count, slices=slices, fbad=bad, show_bad=show_bad, fmt=fmt, dpi=dpi)
    if pretty:
        p.stop()
    update_history(name, 'plot', out=out, max_count=max_count, slices=slices, bad=bad, show_bad=show_bad, fmt=fmt, dpi=dpi)
    dt = time() - t0
    if pretty:
        rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')
    else:
        print(f'✔️ Done!\t time: {dt:.2f} s.')

@app.command('test_binom')
def _test_binom(name: str = Argument(..., help='Project name.'),
                beta: bool = Option(False, help='Use beta-binomial model instead of binomial. In that case, concentration paremeter will be'
                                                ' estimated from the data for each BAD.'),
                w: str = Option(str(), help='Left mode weight. If None, then p=1/2, w=1 will be used everywhere.'),
                estimate_p: bool = Option(False, help='Estimate p, useful for negating reference bias.'),
                max_cover: int = Option(None, help='Maximal cover to be used for parameter estimation.'),
                n_jobs: int = Option(1, help='Number of jobs to be run at parallel, -1 will use all available threads.'),
                pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Calculate p-values with a non-bias adjusted left truncated binomial model.
    """
    t0 = time()
    if pretty:
        p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
        p.add_task(description="Comptuing p-values and effect sizes...", total=None)
        p.start()
    else:
        print('Computing p-values and effect sizes...')
    _, params = binom_test(name, w=None if not w else float(w), beta=beta, estimate_p=estimate_p, max_cover=max_cover,
                           n_jobs=n_jobs)
    if beta or estimate_p:
        if pretty:
            items = ['BAD']
            if estimate_p:
                items.extend(['p_ref', 'p_alt'])
            if beta:
                items.extend(['k_ref', 'k_alt'])
            table = Table(*items)
            for bad in sorted(params):
                d = params[bad]
                items = ['{:.2f}'.format(bad)]
                if estimate_p:
                    for allele in ('ref', 'alt'):
                        pt = '{:.3f}'.format(d[allele][0] if beta else d[allele])
                        items.append(pt)
                if beta:
                    for allele in ('ref', 'alt'):
                        k = '{:.3f}'.format(d[allele][1] if estimate_p else d[allele])
                        items.append(k)
                table.add_row(*items)
            p.stop()
            rprint(table)
            
        else:
            print('Estimated parameters:')
            print(params)
    update_history(name, 'test_binom', w=w, beta=beta, estimate_p=estimate_p, max_cover=max_cover, n_jobs=n_jobs)
    dt = time() - t0
    if pretty:
        rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')
    else:
        print(f'✔️ Done!\t time: {dt:.2f} s.')

@app.command('reproduce', help='Reproduce a project as was recorded by a history file.')
def _reproduce(filename: Path = Argument(..., help='Path to the history JSON file.'),
               pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    filename = str(filename)
    reproduce(filename, pretty=pretty)

def main():
    check_packages()
    app()
