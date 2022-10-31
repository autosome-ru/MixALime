# -*- coding: utf-8 -*-
from enum import Enum
from click import Context
from typer import Typer, Option, Argument
from typer.core import TyperGroup
from typing import List
from rich import print as rprint
from betanegbinfit import __version__ as bnb_version
from jax import __version__ as jax_version
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from .create import create_project
from .combine import combine
from .tests import test
from pathlib import Path
from .fit import fit
from time import time
from dill import __version__ as dill_version
from . import export
from . import __version__ as mixalime_version
import json
from . import plot
import os


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
    


class OrderCommands(TyperGroup):
  def list_commands(self, ctx: Context):
    """Return list of commands in the order appear."""
    return list(self.commands)    # get commands using self.commands

def update_history(name: str, command: str, **kwargs):
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
        d['dill'] = dill_version
        d['name'] = name
    elif command == 'fit':
        for k in ('test', 'combine', 'export', 'plot'):
            if k in d:
                del d[k]
        for k in list(d):
            if k.startswith('export'):
                del d[k]
    elif command == 'test':
        for k in ('test', 'combine', 'export', 'export pvalues', 'export raw_pvalues'):
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
    d[command] = kwargs
    with open(f'{name}.json', 'w') as f:
        json.dump(d, f, indent=4)

def reproduce(filename: str, pretty: bool = True):
    with open(filename, 'r') as f:
        d = json.load(f)
    name = d['name']
    if d['betanegbinfit'] != bnb_version:
        rprint(f'[yellow]Warning:[/yellow] Current [bold]betanegbinfit[/bold] version is {bnb_version}, but the project was created with'
              f' {d["betanegbinfit"]}.')
    if d['jax'] != jax_version:
        rprint(f'[yellow]Warning:[/yellow] Current [bold]JAX[/bold] version is {jax_version}, but the project was created with'
              f' {d["jax"]}.')
    if d['mixalime'] != mixalime_version:
        rprint(f'[yellow]Warning:[/yellow] Current [bold]MixALime[/bold] version is {mixalime_version}, but the project was created with'
              f' {d["mixalime"]}.')
    if d['dill'] != dill_version:
        rprint(f'[yellow]Warning:[/yellow] Current [bold]dill[/bold] version is {dill_version}, but the project was created with'
              f' {d["dill"]}.')
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
        elif command.startswith('combine'):
            lt = command.split()
            if len(lt) != 1:
                command, subname = lt
                args['subname'] = subname
            if 'subname' in args:
                del args['subname']
            r = _combine(name, **args)
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
            else:
                r = _export_all(name, **args)
        if r_old and r != r_old:
            raise Exception(f'"{name}" produced different result from what was recorded at the history file.')
    os.remove(f'{name}.json')
    if f'{name}.json' == filename:
        with open(filename, 'w') as f:
            json.dump(d, f, indent=4)       
    
    
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
[magenta]>[/magenta] [cyan]mixalime create myprojectname [i]scorefiles[/i][/cyan]
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

help_str = 'Initialize [bold]MixALime[/bold] projects initial files: do parsing and filtering of VCFs/BEDs.'

@app.command('create', help=help_str)
def _create(name: str = Argument(..., help='Project name. [bold]MixALime[/bold] will produce files for internal usage that start with [cyan]'
                                            'name[/cyan].'),
            files: List[Path] = Argument(..., help='A list (if applicable, separated by space) of either filenames in VCF or BED-like format, '
                                                    'paths to folder with those files or paths to files that contain list of paths to those '
                                                    'files. '),
            bad_maps: Path = Option(None, help='A path to an interval file that separates genome into BADs. The file must be tabular with '
                                              'columns "[bold]chr[/bold]", "[bold]start[/bold]", "[bold]end[/bold]", "[bold]bad[/bold]".'
                                              ' The "[bold]end[/bold]" column is optional, if it is not present, then the file is '
                                              'assumed to store precise BADs for each SNV.'),
            default_bad: float = Option(1.0, help='Those SNVs that are not present in the [cyan]bad_maps[/cyan]/in BED-like file will assume this'
                                                  ' BAD value.'),
            drop_bad: List[float] = Option(None, '--drop-bad', '-d', help='Those BADs and their respective SNVs will be ommited.'),
            min_qual: int = Option(10, help='Minimal SNV quality'),
            min_cnt: int = Option(5, help='Minimal allowed number of counts at an allele.'),
            cnt_max_sum: int = Option(None, help='Maximal allowed total counts (ref + alt) for an SNV.'),
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
    if cnt_max_sum is None:
        cnt_max_sum = float('inf')
    files = list(map(str, files))
    if bad_maps:
        bad_maps = str(bad_maps)
    t0 = time()
    if not pretty:
        print('Processing files...')
    _, samples, snvs = create_project(name=name, snvs=files, bad_maps=bad_maps, default_bad=default_bad, drop_bads=drop_bad, min_qual=min_qual,
                                      min_cnt=min_cnt, cnt_max_sum=cnt_max_sum, filter_db=filter_db, filter_rs=filter_rs, symmetrify=symmetrify,
                                      filter_name=filter_name, filter_chr=filter_chr, compression=compression, count_snvs=True, 
                                      progress_bar=pretty)
    rows = list()
    tot_snv = 0
    tot_samples = 0
    for bad in sorted(samples):
        a = len(snvs[bad])
        b = samples[bad]
        rows.append(['{:.2f}'.format(bad), str(a), str(b)])
        tot_snv += a
        tot_samples += b
    rows = rows
    if pretty:
        table = Table('BAD', 'SNVs', 'Samples/reps')
        for row in rows:
            table.add_row(*row)
        rprint(table)
        rprint(f'Total SNVs: {tot_snv}, total samples/reps: {tot_samples}')
    else:
        print('BAD\tSNVs\tSamples/reps')
        print('\n'.join(['\t'.join(row) for row in rows]))     
        print(f'Total SNVs: {tot_snv}, total samples/reps: {tot_samples}')    
    update_history(name, 'create', files=files, bad_maps=bad_maps, default_bad=default_bad, drop_bad=drop_bad, min_qual=min_qual,
                   min_cnt=min_cnt, cnt_max_sum=cnt_max_sum, filter_db=filter_db, filter_rs=filter_rs, symmetrify=symmetrify,
                   filter_name=filter_name, filter_chr=filter_chr, compression=compression, expected_result=rows)
    dt = time() - t0
    if pretty:
        rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')
    else:
        print(f'✔️ Done!\t time: {dt:.2f} s.')
    return rows

@app.command('fit')
def _fit(name: str = Argument(..., help='Project name.'),
         dist: Dist = Argument(..., help='Name of distribution that will be used in the mixture model.'),
         model: Model = Option(Model.window.value, help='Model to be used.'),
         left: int = Option(4, help='Left-truncation bound. If None, then it will be estimated from the counts data as the minimal present count minus 1.'),
         estimate_p: bool = Option(False, help='If True, then p will be estimated instead of assuming it to be fixed to bad / (bad + 1)'),
         window_size: int = Option(10000, help='Has effect only if [cyan]model[/cyan] = "[bold]window[/bold]", sets the required minimal window size.'),
         window_behavior: WindowBehavior = Option(WindowBehavior.both.value, help='If "[bold]both[/bold]", then window is expanded in 2 directions. '
                                                                                  'If "[right]right[/right]", then the window is expanded only to'
                                                                                  ' the right (except for cases when such an expansion is not'
                                                                                  ' possible but the minimal slice or window size requirement is not '
                                                                                  'met).'),
         min_slices: int = Option(1, help='Minimal number of slices per window.'),
         adjust_line: bool = Option(False, help='Line parameter beta and mu will be reestimated without a loss of likelihood so they differ'
                                                ' as little as possible from the previous b and mu estimates.'),
         k_left_bound: int = Option(1, help='Minimal allowed value for concentration parameter [italic]k[/italic].'),
         max_count: int = Option(None, help='Maximal number of counts for an allele (be it ref or alt).'),
         max_cover: int = Option(None, help='Maximal sum of ref + alt.'), 
         regul_alpha: float = Option(0.0, help='Regularization/prior strength hyperparameter alpha. Valid only for [cyan]model[/cyan]='
                                               '[yellow]window[/yellow].'),
         regul_n: bool = Option(True, help='Multiply alpha by a number of observations captured by a particular window. Valid only for '
                                           '[cyan]model[/cyan]=[yellow]window[/yellow].'),
         regul_slice: bool = Option(True, help='Multiply alpha by a an average slice captured by a particular window. Valid only for '
                                           '[cyan]model[/cyan]=[yellow]window[/yellow].'),
         regul_prior: Prior = Option('laplace', help='Prior distribution used to penalize concentration parameter kappa. Valid only for '
                                                     '[cyan]model[/cyan]=[yellow]window[/yellow].'),
         adjusted_loglik: bool = Option(False, help='Calculate adjusted loglikelihood alongside other statistics.'),
         n_jobs: int = Option(-1, help='Number of jobs to be run at parallel, -1 will use all available threads.'),
         pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Fit a a mixture model parameters to data for the given project.
    """
    
    start_est = True
    apply_weights = False
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
        apply_weights=apply_weights, regul_alpha=regul_alpha, regul_n=regul_n, regul_slice=regul_slice, regul_prior=regul_prior)
    if pretty:
        p.stop()
    update_history(name, 'fit', dist=dist, model=model, left=left, estimate_p=estimate_p, window_size=window_size, 
                   window_behavior=window_behavior, min_slices=min_slices, adjust_line=adjust_line, k_left_bound=k_left_bound,
                   max_count=max_count, max_cover=max_cover, adjusted_loglik=adjusted_loglik, n_jobs=n_jobs, 
                   regul_alpha=regul_alpha, regul_n=regul_n, regul_slice=regul_slice, regul_prior=regul_prior)
    dt = time() - t0
    if pretty:
        rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')
    else:
        print(f'✔️ Done!\t time: {dt:.2f} s.')

@app.command('test')
def _test(name: str = Argument(..., help='Project name.'),
          correction: Correction = Option(Correction.none.value, help='Posterior weight correction method. It effectively helps to choose'
                                                                      ' a particular component of a distribution for further tests, neglecting '
                                                                      ' an impact of more distant component.'),
          gof_tr: float = Option(None, help='Conservative scoring will be used if goodness-of-fit statistic (RMSEA) exceeds [cyan]gof-tr[/cyan] '
                                            'for a particular slice.'),
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
    test(name, correction=correction, gof_tr=gof_tr, n_jobs=n_jobs)
    if pretty:
        p.stop()
    update_history(name, 'test', correction=correction, gof_tr=gof_tr, n_jobs=n_jobs)
    dt = time() - t0
    if pretty:
        rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')
    else:
        print(f'✔️ Done!\t time: {dt:.2f} s.')
        

@app.command('combine')
def _combine(name: str = Argument(..., help='Project name.'),
             group: List[Path] = Option(None, '--group', '-g', help='A list of either filenames (vcf or BAM-like tabulars) or folders that contain '
                                                                      'those filenames or file(s) that contain a list of paths to files.'
                                                                      ' SNV p-values from those files shall be combined via logit method.'),
             alpha: float = Option(0.05, help='FWER, family-wise error rate.'),
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
    r = combine(name, group_files=group, alpha=alpha, filter_id=filter_id, subname=subname, n_jobs=n_jobs)[subname]
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
        rprint('Number of significant SNVs after FDR correction:')
        table = Table('Ref', 'Alt', 'Both', 'Total significant\n(Percentage of total SNVs)')
        table.add_row(str(ref), str(alt), str(both), f'{total} ({total/len(r) * 100:.2f}%)')
        rprint(table)
    else:
        print('Number of significant SNVs after FDR correction:')
        print('\t'.join(('Ref', 'Alt', 'Both', 'Total significant (Percentage of total SNVs)')))
        print('\t'.join((str(ref), str(alt), str(both), f'{total} ({total/len(r) * 100:.2f}%)')))
    update_history(name, 'combine', group=group, alpha=alpha, filter_id=filter_id, subname=subname, filter_chr=filter_chr, n_jobs=n_jobs,
                   expected_result=expected_res)
    dt = time() - t0
    if pretty:
        rprint(f'[green][bold]✔️[/bold] Done![/green]\t time: {dt:.2f} s.')
    else:
        print(f'✔️ Done!\t time: {dt:.2f} s.')
        
    return expected_res


@app_export.command('all', help='Export everything.')
def _export_all(name: str = Argument(..., help='Project name.'), out: Path = Argument(..., help='Output filename/path.'),
                rep_info: bool = Option(False, help='Include raw p-values and names of sample scorefiles to the tabular. '
                                                    'Note that this may bloat output if number of samples is high.'),
                pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    out = str(out)
    t0 = time()
    if pretty:
        p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
        p.add_task(description="Exporting tabular data...", total=None)
        p.start()
    export.export_all(name, out, rep_info=rep_info)
    if pretty:
        p.stop()
    update_history(name, 'export', out=out, rep_info=rep_info)
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
                      rep_info: bool = Option(False, help='Include raw p-values and names of sample scorefiles to the tabular. '
                                                          'Note that this may bloat output if number of samples is high.'),
                      subname: str = Option(None, help='A subname that can be used to reference a set of combined p-values in case if you'
                                                       ' provided one at [cyan bold]combine[/cyan bold] step.'),
                      pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    '''
    Export combined across samples and FDR-corrected p-values.
    '''
    out = str(out)
    export.export_combined_pvalues(name, out, rep_info=rep_info, subname=subname)
    update_history(name, 'export pvalues', out=out, rep_info=rep_info, subname=subname)
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

@app_export.command('demo')
def _demo(export_path: Path = Option(str(), help='Path where the demo data will be extracted.'),
          pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    '''
    Extract demonstration data from ChIP-seq experiment.
    '''
    export_path = str(export_path)
    export.export_demo(export_path)
    if pretty:
        rprint('[green][bold]✔️[/bold] Done![/green]')
    else:
        print('✔️ Done!')
    

@app.command('plot', help='Visualize model fits and plot model fit indices for diagnostic purposes.')
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

@app.command('reproduce', help='Reproduce a project as was recorded by a history file.')
def _reproduce(filename: Path = Argument(..., help='Path to the history JSON file.'),
               pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    filename = str(filename)
    reproduce(filename, pretty=pretty)

def main():
    app()
