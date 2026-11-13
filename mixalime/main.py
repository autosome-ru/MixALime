# -*- coding: utf-8 -*-
from . import check_packages
from .utils import get_init_file
from enum import Enum
from contextlib import contextmanager
from typer import Typer, Option, Argument, Context
from typer.core import TyperGroup
from typer.models import ParameterInfo
from typing import List, Tuple
from rich import print as rprint
from rich.text import Text
from jacobin import __version__ as jacobin_version
from jax import __version__ as jax_version
from scipy import __version__ as scipy_version
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from .diff import differential_test, anova_test
from .create import create_project
from .combine import combine, polymbine
from .tests import test, binom_test
from pathlib import Path
from .fit import fit
from time import time
from dill import __version__ as dill_version
import inspect
from . import export
from . import __version__ as mixalime_version
import json
import os
from . import plot
import typer


__all__ = ['main']

class Compression(str, Enum):
    lzma = 'lzma'
    gzip = 'gzip'
    bz2 = 'bz2'
    raw = 'raw'

class Dist(str, Enum):
    nb = 'NB'
    betanb = 'BetaNB'
    cdnb = 'MCNB'

class WindowBehavior(str, Enum):
    both = 'both'
    right = 'right'

class Correction(str, Enum):
    none = 'none'
    hard = 'hard'
    single = 'single'

class Unsupported(str, Enum):
    conservative = 'conservative'
    nan = 'nan'

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


def echo(text: str, pretty: bool):
    """Prints text with rich markup. Without pretty the markup is removed."""
    if pretty:
        rprint(text)
    else:
        print(Text.from_markup(text).plain)


def echo_table(header: tuple, rows: list, pretty: bool):
    """Prints a table: a rich one, or tab-separated lines without pretty."""
    if pretty:
        table = Table(*header)
        for row in rows:
            table.add_row(*map(str, row))
        rprint(table)
    else:
        print('\t'.join(h.replace('\n', ' ') for h in header))
        for row in rows:
            print('\t'.join(map(str, row)))


def done(pretty: bool, t0: float = None):
    """Prints the closing line of a command, with the time since t0 if it is given."""
    s = '[green][bold]✔️[/bold] Done![/green]'
    if t0 is not None:
        s += f'\t time: {time() - t0:.2f} s.'
    echo(s, pretty)


@contextmanager
def status(description: str, pretty: bool):
    """Shows a spinner with a description while the body runs. Without pretty the description is printed once."""
    if not pretty:
        print(description)
        yield
        return
    with Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True) as p:
        p.add_task(description=description, total=None)
        yield


def _value(x):
    return x.value if isinstance(x, Enum) else x


_DO_NOT_UPDATE_HISTORY = False

# Commands whose history entries are kept per subname.
_SUBNAMED = ('combine', 'multiple_combine', 'difftest', 'anova', 'export pvalues', 'export difftest', 'export anova')
_PACKAGES = ('jacobin', 'jax', 'scipy', 'mixalime', 'dill')


def _versions() -> dict:
    return {'jacobin': jacobin_version, 'jax': jax_version, 'mixalime': mixalime_version, 'scipy': scipy_version,
            'dill': dill_version}


def _drop(d: dict, *prefixes):
    """Removes the entries of the commands in prefixes from a history, with all their subnames."""
    for k in list(d):
        if type(d[k]) is dict and any(k == p or k.startswith(p + ' ') for p in prefixes):
            del d[k]


def _combined_anovas(d: dict) -> list:
    """Keys of the anova entries that take their groups from the combined p-values, each followed by the key of its export."""
    keys = list()
    for k, v in d.items():
        if (k == 'anova' or k.startswith('anova ')) and type(v) is dict and not v.get('groups'):
            keys += [k, f'export {k}']
    return [k for k in keys if k in d]


def update_history(name: str, command: str, **kwargs):
    """
    Records a command and its arguments in the history file of a project.

    Entries of the commands whose results the command invalidates are
    removed. The entry itself goes to the end, so the file lists commands
    in an order they can be replayed in. The file is replaced atomically,
    and an unreadable one is moved to name.json.bad before a new one is started.
    """
    if _DO_NOT_UPDATE_HISTORY:
        return
    path = f'{name}.json'
    try:
        with open(path, 'r') as f:
            d = json.load(f)
        if type(d) is not dict:
            raise ValueError('A history is a JSON object.')
    except FileNotFoundError:
        d = dict()
    except ValueError:
        os.replace(path, f'{path}.bad')
        echo(f'Warning: {path} is not a valid history, it is moved to {path}.bad and a new one is started.', False)
        d = dict()
    subname = kwargs.pop('subname', None) if command in _SUBNAMED else None
    key = f'{command} {subname}' if subname else command
    if command == 'create' or 'name' not in d:
        if command == 'create':
            d.clear()
        d = {**_versions(), 'name': name, **d}
    if command == 'fit':
        _drop(d, 'fit', 'test', 'test_binom', 'combine', 'multiple_combine', 'difftest', 'anova', 'export', 'plot')
    elif command in ('test', 'test_binom'):
        # The anovas without groups can not be replayed once the combine entries they read the groups from are gone.
        _drop(d, 'test', 'test_binom', 'combine', 'multiple_combine', 'export pvalues', 'export raw_pvalues')
        d.pop('export', None)
        for k in _combined_anovas(d):
            del d[k]
    elif command in ('combine', 'difftest', 'anova'):
        # The export of everything and the export of this result are out of date, other exports are not.
        export = 'export pvalues' if command == 'combine' else f'export {command}'
        d.pop('export', None)
        d.pop(f'{export} {subname}' if subname else export, None)
    d.pop(key, None)
    d[key] = kwargs
    if command == 'combine':
        # An anova without groups reads them from the combine entries, so it follows them.
        for k in _combined_anovas(d):
            d[k] = d.pop(k)
    tmp = f'{path}.{os.getpid()}.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(d, f, indent=4, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _require(name: str, kind: str, command: str):
    """Raises a FileNotFoundError naming the command to run first if the project has no results of the stage kind."""
    filename = f'{name}.{kind}.{get_init_file(name).split(".")[-1]}'
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'No {filename}: the project has no {kind} results, the {command} command makes them.')


def _split_records(text: str, option: str, shape: str, n: int) -> list:
    """Splits 'a,b;c,d' into lists of n stripped fields. A BadParameter naming the option and the shape is raised for any other form."""
    records = [[field.strip() for field in record.split(',')] for record in text.split(';')]
    for record in records:
        if len(record) != n or not all(record):
            raise typer.BadParameter(f'"{",".join(record)}" is not of the form {shape}.', param_hint=option)
    return records


def _call(fun, *args, **kwargs):
    """Calls a command as a function: options absent from kwargs take their declared defaults."""
    for pname, param in inspect.signature(fun).parameters.items():
        if pname not in kwargs and isinstance(param.default, ParameterInfo) and param.default.default is not ...:
            kwargs[pname] = param.default.default
    return fun(*args, **kwargs)


def reproduce(filename: str, pretty: bool = True, check_results: bool = True):
    """Runs the commands recorded in a history file again and compares their summaries with the recorded ones."""
    global _DO_NOT_UPDATE_HISTORY
    with open(filename, 'r') as f:
        d = json.load(f)
    name = d['name']
    if 'betanegbinfit' in d:
        echo('[yellow]Warning:[/yellow] The project was created by a [bold]MixALime[/bold] version built on '
             '[bold]betanegbinfit[/bold], small numerical differences are expected.', pretty)
    for package in _PACKAGES:
        ver = d.get(package, None)
        if ver:
            curver = _versions()[package]
            if curver != ver:
                echo(f'[yellow]Warning:[/yellow] Current [bold]{package}[/bold] version is {curver}, but the project was created with'
                     f' {ver}.', pretty)
        else:
            echo(f'[yellow]Warning:[/yellow] No information on [bold]{package}[/bold] found in the history file.', pretty)
    commands = {'create': _create, 'fit': _fit, 'test': _test, 'test_binom': _test_binom, 'combine': _combine,
                'multiple_combine': _multiple_combine, 'difftest': _difftest, 'anova': _anova, 'plot': _plot_all,
                'export': _export_all, 'export params': _params, 'export indices': _indices, 'export counts': _counts,
                'export pvalues': _combined_pvalues, 'export raw_pvalues': _raw_pvalues, 'export raw-pvalues': _raw_pvalues,
                'export difftest': _difftests, 'export anova': _anovas}
    _DO_NOT_UPDATE_HISTORY = True
    try:
        for key, args in d.items():
            if type(args) is not dict:
                continue
            args = dict(args)
            r_old = args.pop('expected_result', None)
            command = max((c for c in commands if key == c or (c in _SUBNAMED and key.startswith(c + ' '))), key=len,
                          default=None)
            if command is None:
                echo(f'[yellow]Warning:[/yellow] The entry [bold]{key}[/bold] of the history file is not replayed.', pretty)
                continue
            if command in _SUBNAMED:
                args['subname'] = key[len(command) + 1:] or args.get('subname')
            dropped = sorted(set(args) - set(inspect.signature(commands[command]).parameters))
            if dropped:
                echo(f'[yellow]Warning:[/yellow] Options {", ".join(dropped)} of [bold]{key}[/bold] are not supported by this '
                     'version and are ignored.', pretty)
                args = {k: v for k, v in args.items() if k not in dropped}
            r = _call(commands[command], name, **args, pretty=pretty)
            if check_results and r_old is not None and r is not None and json.loads(json.dumps(r, default=str)) != r_old:
                raise Exception(f'"{key}" of "{name}" produced a result different from the one recorded in the history file.')
    finally:
        _DO_NOT_UPDATE_HISTORY = False


doc = f'''
[bold]MixALime[/bold] version {mixalime_version}: [bold]M[/bold]ixture models for [bold]A[/bold]llelic [bold]I[/bold]mbalance [bold]E[/bold]stimation
\b\n
\b\n
A typical [bold]MixALime[/bold] session consists of sequential runs of [bold cyan]create[/bold cyan], [bold cyan]fit[/bold cyan], [bold cyan]test[/bold cyan], \
[bold cyan]combine[/bold cyan] and, finally, [bold cyan]export all[/bold cyan], [bold cyan]plot all[/bold cyan] commands. For instance, we provide a demo \
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
Finally, we obtain fancy plots for diagnostic purposes and easy-to-work with tabular data:
[magenta]>[/magenta] [cyan]mixalime export all myprojectname [i]results_folder[/i][/cyan]
[magenta]>[/magenta] [cyan]mixalime plot all myprojectname [i]results_folder[/i][/cyan]
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

def version_callback(v: bool):
    if v:
        print(mixalime_version)
        raise typer.Exit()
@app.callback()
def main(version: bool = Option(None, "--version", callback=version_callback)):
    pass

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
            max_cover: int = Option(None, help='Observations whose total counts (ref + alt) are not below this value are omitted.'),
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
    compression = str(_value(compression))
    files = list(map(str, files))
    if bad_maps:
        bad_maps = str(bad_maps)
    t0 = time()
    if not pretty:
        print('Processing files...')
    _, samples, snvs = create_project(name=name, snvs=files, bad_maps=bad_maps, default_bad=default_bad, drop_bads=drop_bad, min_qual=min_qual,
                                      min_cnt=min_cnt, max_cover=float('inf') if max_cover is None else max_cover,
                                      filter_db=filter_db, filter_rs=filter_rs, symmetrify=symmetrify,
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
        echo('[yellow]Warning:[/yellow] BAD=0 is present in the data. Please remove BAD=0 occurrences or substitute them with BAD=1.', pretty)
    echo_table(('BAD', 'SNVs', 'Observations'), rows, pretty)
    echo(f'Total unique SNVs: {tot_snv}, total observations: {tot_samples}', pretty)
    update_history(name, 'create', files=files, bad_maps=bad_maps, default_bad=default_bad, drop_bad=drop_bad, min_qual=min_qual,
                   min_cnt=min_cnt, max_cover=max_cover, filter_db=filter_db, filter_rs=filter_rs, symmetrify=symmetrify,
                   filter_name=filter_name, filter_chr=filter_chr, compression=compression, snp_bad_check=snp_bad_check,
                   expected_result=rows)
    done(pretty, t0)
    return rows

@app.command('fit')
def _fit(name: str = Argument(..., help='Project name.'),
         dist: Dist = Argument(..., help='Name of distribution that will be used in the mixture model.'),
         left: int = Option(None, help='Left-truncation bound. If None, then it will be estimated from the counts data as the minimal present count minus 1.'),
         estimate_p: bool = Option(False, help='If True, then p will be estimated instead of assuming it to be fixed to bad / (bad + 1)'),
         window_size: int = Option(10000, help='Sets the required minimal window size.'),
         window_behavior: WindowBehavior = Option(WindowBehavior.both.value, help='If "[bold]both[/bold]", then window is expanded in 2 directions. '
                                                                                  'If "[bold]right[/bold]", then the window is expanded only to'
                                                                                  ' the right (except for cases when such an expansion is not'
                                                                                  ' possible but the minimal slice or window size requirement is not '
                                                                                  'met).'),
         min_slices: int = Option(3, help='Minimal number of slices per window.'),
         stop_slice_n: int = Option(10, help='Parameters are estimated at a slice with at least [cyan]stop-slice-n[/cyan] samples. A slice'
                                              ' with fewer samples takes the estimates of the last fitted window.'),
         window_ratio: float = Option(None, help='A window around a slice [italic]s[/italic] holds no slice below [italic]s[/italic] / '
                                                 '[cyan]window-ratio[/cyan] or above [italic]s[/italic] * [cyan]window-ratio[/cyan], '
                                                 'even if it has fewer samples than [cyan]window-size[/cyan] then. By default, a window '
                                                 'has no such bound.'),
         adjust_line: bool = Option(False, help='Line parameter beta and mu will be reestimated without a loss of likelihood so they differ'
                                                ' as little as possible from the previous b and mu estimates.'),
         k_left_bound: int = Option(0, help='Minimal allowed value for concentration parameter [italic]k[/italic]. For '
                                            '[yellow]BetaNB[/yellow] with an r transform and a fixed p it is at least BAD + 1.001.'),
         max_count: int = Option(None, help='Maximal number of counts for an allele (be it ref or alt).'),
         max_cover: int = Option(None, help='Maximal sum of ref + alt.'),
         regul: bool = Option(False, help='Apply regularization/encourage high kappa parameter estimates. Valid only for [cyan]dist[/cyan]='
                                               '[yellow]BetaNB[/yellow].'),
         regul_alpha: float = Option(1.0, help='Regularization/prior strength hyperparameter alpha. Takes effect only together with '
                                               '[cyan]--regul[/cyan]. Valid only for [cyan]dist[/cyan]=[yellow]BetaNB[/yellow].'),
         regul_n: bool = Option(True, help='Multiply alpha by a number of SNV records captured by a particular window. Valid only for [cyan]dist[/cyan]='
                                               '[yellow]BetaNB[/yellow].'),
         regul_slice: bool = Option(True, help='Multiply alpha by a an average slice captured by a particular window. Valid only for [cyan]dist[/cyan]='
                                               '[yellow]BetaNB[/yellow].'),
         regul_prior: Prior = Option('laplace', help='Prior distribution used to penalize concentration parameter kappa. Valid only for [cyan]dist[/cyan]='
                                               '[yellow]BetaNB[/yellow].'),
         std : bool = Option(False, help='Compute standard errors for parameter estimates, and r and kappa at each slice with their errors.'
                                         ' It may increase computation time significantly.'),
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
         n_jobs: int = Option(-1, help='Number of jobs to be run at parallel, -1 will use all available cores.'),
         pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Fit a a mixture model parameters to data for the given project.
    """
    dist, window_behavior, regul_prior = _value(dist), _value(window_behavior), _value(regul_prior)
    r_transform, small_dataset_strategy = _value(r_transform), _value(small_dataset_strategy)
    if fix_params is None:
        fix_params = str()
    if not regul and regul_alpha != 1.0:
        echo('[yellow]Warning:[/yellow] [cyan]--regul-alpha[/cyan] has no effect without [cyan]--regul[/cyan].', pretty)
    t0 = time()
    with status('Optimizing model parameters...', pretty):
        fit(name, dist=dist, left=left, estimate_p=estimate_p, window_size=window_size,
            window_behavior=window_behavior, min_slices=min_slices, adjust_line=adjust_line, k_left_bound=k_left_bound,
            max_count=float('inf') if max_count is None else max_count,
            max_cover=float('inf') if max_cover is None else max_cover, adjusted_loglik=adjusted_loglik, n_jobs=n_jobs,
            regul_alpha=regul_alpha if regul else 0, regul_n=regul_n, regul_slice=regul_slice, regul_prior=regul_prior,
            fix_params=fix_params, std=std, optimizer=optimizer, r_transform=None if r_transform == 'none' else r_transform,
            symmetrify=symmetrify, small_dataset_strategy=small_dataset_strategy, small_dataset_n=small_dataset_n,
            stop_slice_n=stop_slice_n, kappa_right=kappa_right_boundary, window_ratio=window_ratio)
    update_history(name, 'fit', dist=dist, left=left, estimate_p=estimate_p, window_size=window_size,
                   window_behavior=window_behavior, min_slices=min_slices, adjust_line=adjust_line, k_left_bound=k_left_bound,
                   max_count=max_count, max_cover=max_cover, adjusted_loglik=adjusted_loglik, n_jobs=n_jobs,
                   regul=regul, regul_alpha=regul_alpha, regul_n=regul_n, regul_slice=regul_slice, regul_prior=regul_prior,
                   fix_params=fix_params, std=std, optimizer=optimizer, r_transform=r_transform, symmetrify=symmetrify,
                   small_dataset_strategy=small_dataset_strategy, small_dataset_n=small_dataset_n, stop_slice_n=stop_slice_n,
                   kappa_right_boundary=kappa_right_boundary, window_ratio=window_ratio)
    done(pretty, t0)

@app.command('test')
def _test(name: str = Argument(..., help='Project name.'),
          fit: str = Option(None, help='Path to a fit file from a different project. If not provided, fit from the same project is used.'),
          correction: Correction = Option(Correction.single.value, help='Posterior weight correction method. It effectively helps to choose'
                                                                      ' a particular component of a distribution for further tests, neglecting '
                                                                      ' an impact of more distant component.'),
          gof_thr: float = Option(None, help='Conservative scoring will be used if goodness-of-fit statistic (RMSEA) exceeds [cyan]gof-thr[/cyan] '
                                                'for a particular slice.'),
          stop_slice_n_thr: int = Option(-1, help='Conservative scoring will be used if number of samples at a slice is below '
                                                  '[cyan]stop-slice-n-thr[/cyan].'),
          min_support: float = Option(0.5, help='A slice is unsupported if less than this share of the samples of the window that'
                                                ' gave its parameters lies within a factor [cyan]support-ratio[/cyan] of the slice.'
                                                ' Its parameters are then an extrapolation from distant slices. 0 turns the check off.'),
          support_ratio: float = Option(2.0, help='Slices within this factor of a slice count as near it in [cyan]min-support[/cyan].'),
          unsupported: Unsupported = Option(Unsupported.conservative.value, help='Unsupported slices are scored conservatively'
                                                                                 ' ([yellow]conservative[/yellow]) or get no p-values'
                                                                                 ' ([yellow]nan[/yellow]).'),
          min_posterior_prob: float = Option(0.0, help='Both p-values of an SNV are set to 1 if the probability that the ref and'
                                                       ' the alt tests put the same allele on more copies is below'
                                                       ' [cyan]min-posterior-prob[/cyan]. 0 turns this off.'),
          n_jobs: int = Option(1, help='Number of jobs to be run at parallel, -1 will use all available cores.'),
          pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Calculate p-values using parameter estimates obtained after [cyan bold]fit[/cyan bold] command.
    """
    correction, unsupported = _value(correction), _value(unsupported)
    if not fit:
        _require(name, 'fit', 'fit')
    t0 = time()
    with status('Computing p-values and effect sizes...', pretty):
        test(name, correction=None if correction == 'none' else correction, gof_tr=gof_thr, fit=fit,
             dataset_n_thr=stop_slice_n_thr, min_support=min_support, support_ratio=support_ratio,
             unsupported=unsupported, min_posterior_prob=min_posterior_prob, n_jobs=n_jobs)
    update_history(name, 'test', correction=correction, gof_thr=gof_thr, fit=fit,
                   stop_slice_n_thr=stop_slice_n_thr, min_support=min_support, support_ratio=support_ratio,
                   unsupported=unsupported, min_posterior_prob=min_posterior_prob, n_jobs=n_jobs)
    done(pretty, t0)


def _report_combine(r: dict, alpha: float, adaptive_coverage: dict, pretty: bool) -> list:
    """Prints the summary of combined p-values r. Returns the numbers of SNVs significant by ref, by alt and by both."""
    ref = alt = both = total = 0
    for t in r.values():
        pv_ref, pv_alt = t[-1]
        ref += pv_ref < alpha
        alt += pv_alt < alpha
        total += (pv_ref < alpha) | (pv_alt < alpha)
        both += (pv_ref < alpha) & (pv_alt < alpha)
    if adaptive_coverage:
        echo('Minimal required coverages for each BAD:', pretty)
        rows = list()
        for bad in sorted(adaptive_coverage['ref']):
            cov_ref = adaptive_coverage['ref'][bad]
            cov_alt = adaptive_coverage['alt'][bad]
            rows.append((bad, cov_ref, cov_alt, min(cov_ref, cov_alt)))
        echo_table(('BAD', 'Ref', 'Alt', 'Min'), rows, pretty)
    if not r:
        echo('No SNVs passed the filters.', pretty)
    else:
        echo('Number of significantly imbalanced SNVs after FDR correction:', pretty)
        echo_table(('Ref', 'Alt', 'Both', 'Total significant\n(Percentage of total SNVs)'),
                   [(ref, alt, both, f'{total} ({total / len(r) * 100:.2f}%)')], pretty)
        echo(f'Total SNVs tested: {len(r)}', pretty)
    return [int(ref), int(alt), int(both)]


@app.command('combine')
def _combine(name: str = Argument(..., help='Project name.'),
             group: List[Path] = Option(None, '--group', '-g', help='A list of either filenames (vcf or BAM-like tabulars) or folders that contain '
                                                                      'those filenames or file(s) that contain a list of paths to files, or a mask'
                                                                      ' (masks should start with "[yellow]m:[/yellow]"'
                                                                      ' prefix, e.g. "m:vcfs/*_M_*.vcf.gz"). '
                                                                      ' SNV p-values from those files shall be combined via logit method.'),
             alpha: float = Option(0.05, help='FDR level (Benjamini-Hochberg). Sets the threshold of the printed summary;'
                                              ' the adjusted p-values in the tables do not depend on it.'),
             min_cover: int = Option(20, help='If none of combined p-values is associated with a sample whose ref + alt is at least'
                                              ' [cyan]min_cover[/cyan], the SNV is omitted.'),
             adaptive_min_cover: bool = Option(False, help='Use adaptive [cyan]min_cover[/cyan] for each BAD. The algorithm has two hyperparameters: '
                                               '[cyan]adaptive_es[/cyan] and [cyan]adaptive_pval[/cyan]. The minimal coverage where effect-size of '
                                               'at least [cyan]adaptive_es[/cyan] is achievable for a p-value of [cyan]adaptive_pval[/cyan].'),
             adaptive_es: float = Option(1.0, help='Minimal required effect-size for the adaptive coverage algorithm.'),
             adaptive_pval: float = Option(0.05, help='Minimal required p-value for the adaptive coverage algorithm.'),
             uniform_weights: bool = Option(True, help='Uniform weighing for effect-sizes.'),
             filter_id: str = Option(None, help=r'Only SNVs whose IDs agree with this regex pattern are tested (e.g. "rs\w+").'),
             filter_chr: str = Option(None, help=r'SNVs with chr that does not align with this regex pattern are filtered (e.g. "chr\d+").'),
             subname: str = Option(None, help='You may give a result a subname in case you plan to use multiple groups.'),
             n_jobs: int = Option(1, help='Number of jobs to be run at parallel, -1 will use all available cores.'),
             pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Combine p-values obtained at [cyan bold]test[/cyan bold] stage with a Mudholkar-George method.
    """
    if group:
        group = list(map(str, group))
    subname = str(subname) if subname else None
    _require(name, 'test', 'test or test_binom')
    t0 = time()
    with status('Combining p-values with respect to sample groups...', pretty):
        r, adaptive_coverage = combine(name, group_files=group if group else [r'm:*'], alpha=alpha, filter_id=filter_id,
                                       filter_chr=filter_chr, subname=subname, min_cnt_sum=min_cover,
                                       adaptive_min_cover=adaptive_min_cover, adaptive_es=adaptive_es,
                                       adaptive_pval=adaptive_pval, uniform_weights=uniform_weights, n_jobs=n_jobs)
    expected_res = _report_combine(r[subname]['snvs'], alpha, adaptive_coverage if adaptive_min_cover else None, pretty)
    update_history(name, 'combine', group=group, alpha=alpha, min_cover=min_cover, adaptive_min_cover=adaptive_min_cover,
                   adaptive_es=adaptive_es, adaptive_pval=adaptive_pval, filter_id=filter_id, subname=subname, filter_chr=filter_chr,
                   uniform_weights=uniform_weights, n_jobs=n_jobs, expected_result=expected_res)
    done(pretty, t0)
    return expected_res

@app.command('multiple_combine')
def _multiple_combine(
                      new_project_name: str = Argument(..., help='Output quasi-project name.'),
                      names: List[str] = Argument(..., help='List of project names'),
                      alpha: float = Option(0.05, help='FDR level (Benjamini-Hochberg). Sets the threshold of the printed summary;'
                                                       ' the adjusted p-values in the tables do not depend on it.'),
                      group: List[Path] = Option(None, '--group', '-g', help='A list of either filenames (vcf or BAM-like tabulars) or folders that contain '
                                                                               'those filenames or file(s) that contain a list of paths to files, or a mask'
                                                                               ' (masks should start with "[yellow]m:[/yellow]"'
                                                                               ' prefix, e.g. "m:vcfs/*_M_*.vcf.gz"). '
                                                                               ' SNV p-values from those files shall be combined via logit method.'
                                                                               ' With as many [cyan]--group[/cyan] options as projects, the i-th one selects the'
                                                                               ' files of the i-th project, otherwise each one is applied to every project.'),
                      min_cover: int = Option(20, help='If none of combined p-values is associated with a sample whose ref + alt is at least'
                                                      ' [cyan]min_cover[/cyan], the SNV is omitted.'),
                      adaptive_min_cover: bool = Option(False, help='Use adaptive [cyan]min_cover[/cyan] for each BAD. The algorithm has two hyperparameters: '
                                                       '[cyan]adaptive_es[/cyan] and [cyan]adaptive_pval[/cyan]. The minimal coverage where effect-size of '
                                                       'at least [cyan]adaptive_es[/cyan] is achievable for a p-value of [cyan]adaptive_pval[/cyan].'),
                      adaptive_es: float = Option(1.0, help='Minimal required effect-size for the adaptive coverage algorithm.'),
                      adaptive_pval: float = Option(0.05, help='Minimal required p-value for the adaptive coverage algorithm.'),
                      uniform_weights: bool = Option(True, help='Uniform weighing for effect-sizes.'),
                      filter_id: str = Option(None, help=r'Only SNVs whose IDs agree with this regex pattern are tested (e.g. "rs\w+").'),
                      filter_chr: str = Option(None, help=r'SNVs with chr that does not align with this regex pattern are filtered (e.g. "chr\d+").'),
                      subname: str = Option(None, help='You may give a result a subname in case you plan to use multiple groups.'),
                      n_jobs: int = Option(1, help='Number of jobs to be run at parallel, -1 will use all available cores.'),
                      pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Combine p-values obtained at [cyan bold]test[/cyan bold] stage with a Mudholkar-George method across different projects.
    Treats p-values from different projects as extra replicates.
    """
    if group:
        group = list(map(str, group))
    subname = str(subname) if subname else None
    t0 = time()
    with status('Combining p-values with respect to sample groups...', pretty):
        r, coverages = polymbine(new_project_name, names, group_files=group if group else [r'm:*'], alpha=alpha,
                                 filter_id=filter_id, filter_chr=filter_chr,
                                 subname=subname, min_cnt_sum=min_cover, adaptive_min_cover=adaptive_min_cover,
                                 adaptive_es=adaptive_es, adaptive_pval=adaptive_pval,
                                 uniform_weights=uniform_weights, n_jobs=n_jobs)
    for name, coverage in zip(names, coverages or ()):
        echo(f'Minimal required coverages for each BAD of the project {name}:'.replace('[', r'\['), pretty)
        ref, alt = coverage['ref'], coverage['alt']
        echo_table(('BAD', 'Ref', 'Alt', 'Min'), [(bad, ref[bad], alt[bad], min(ref[bad], alt[bad])) for bad in sorted(ref)], pretty)
    expected_res = _report_combine(r[subname]['snvs'], alpha, None, pretty)
    update_history(new_project_name, 'multiple_combine', names=names, group=group, alpha=alpha, min_cover=min_cover,
                   adaptive_min_cover=adaptive_min_cover, adaptive_es=adaptive_es, adaptive_pval=adaptive_pval, filter_id=filter_id,
                   subname=subname, filter_chr=filter_chr, uniform_weights=uniform_weights, n_jobs=n_jobs,
                   expected_result=expected_res)
    done(pretty, t0)
    return expected_res


def _report_diff(r, alpha: float, pretty: bool) -> list:
    """Prints the summary of a table of differential tests. Returns the numbers of SNVs significant by ref, by alt and by any."""
    ref = r['ref_fdr_pval'] < alpha
    alt = r['alt_fdr_pval'] < alpha
    both = (ref & alt).sum()
    total = (ref | alt).sum()
    ref = ref.sum()
    alt = alt.sum()
    if not len(r):
        echo('No SNVs passed the filters.', pretty)
    else:
        echo('Number of significantly differentially expressed SNVs after FDR correction:', pretty)
        echo_table(('Ref', 'Alt', 'Both', 'Total\n(Percentage of tests)'),
                   [(ref, alt, both, f'{total} ({total / len(r) * 100:.2f}%)')], pretty)
        echo(f'Total tests (an SNV is tested at each of its BADs): {len(r)}', pretty)
    return [int(ref), int(alt), int(total)]


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
                                                   '[cyan]--mode[/cyan]=[yellow]wald[/yellow]. The sandwich estimate is '
                                                   'anti-conservative in groups of few observations.'),
              n_bootstrap: int = Option(0, help='Bootstrap iterations of the bias correction of p. The draws are seeded by the SNV, so a run is reproducible.'),
              logit_transform: bool = Option(False, help='Apply logit transform to [bold]p[/bold] and its variance with Delta method. Applicable '
                                                         'only if [cyan]--mode[/cyan]=[yellow]wald[/yellow].'),
              test_groups: bool = Option(False, help='Whole groups will be tested against each other first. Note that this will take'
                                                    ' the same time as [cyan]fit[/cyan] stage.'),
              contrasts: Tuple[float, float, float] = Option((1, -1, 0), help='Contrasts vector where 1st, 2nd positions are for groups A and B '
                                                                              'respectively and the 3rd stand for the free term, i.e. the default'
                                                                              ' value will test for difference between p_a and p_b, but '
                                                                              '[bold]1 0 -0.5[/bold] will test only for p_a being equal to 0.5. '),
              alpha: float = Option(0.05, help='FDR level (Benjamini-Hochberg). Sets the threshold of the printed summary;'
                                               ' the adjusted p-values in the tables do not depend on it.'),
              min_samples: int = Option(2, help='Minimal number of samples/reps per an SNV to be considered for the analysis.'),
              min_cover: int = Option(None, help='Minimal required cover (ref + alt) for an SNV to be considered.'),
              max_cover: int = Option(None, help='Maximal allowed cover (ref + alt) for an SNV to be considered.'),
              max_cover_group_test: int = Option(None, help='Maximal allowed cover (ref + alt) for an SNV to be considered. Used only to trim'
                                                            ' for the whole group test if applicable (i.e. if [cyan]test_groups[/cyan]) to '
                                                            'avoid long waiting times similarily to [cyan]fit[/cyan].'),
              filter_id: str = Option(None, help=r'Only SNVs whose IDs agree with this regex pattern are tested (e.g. "rs\w+").'),
              filter_chr: str = Option(None, help=r'SNVs with chr that does not align with this regex pattern are filtered (e.g. "chr\d+").'),
              subname: str = Option(None, help='You may give a result a subname in case you plan to draw multiple comparisons.'),
              n_jobs: int = Option(1, help='Number of jobs to be run at parallel, -1 will use all available cores.'),
              pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Differential expression tests via Wald/LR tests.
    """
    if not fit:
        _require(name, 'fit', 'fit')
    t0 = time()
    group_a = str(group_control)
    group_b = str(group_test)
    contrasts = list(contrasts)
    mode = _value(mode)
    subname = str(subname) if subname else None
    with status('Performing {} tests...'.format('LRT' if mode == 'lrt' else 'Wald'), pretty):
        r = differential_test(name, group_a=group_a, group_b=group_b, mode=mode, min_samples=min_samples, min_cover=min_cover,
                              max_cover=max_cover, group_test=test_groups, subname=subname,  filter_id=filter_id,
                              max_cover_group_test=max_cover_group_test, filter_chr=filter_chr, alpha=alpha, n_jobs=n_jobs,
                              param_mode='window' if param_window else 'line', logit_transform=logit_transform,
                              robust_se=robust_se, contrasts=contrasts, n_bootstrap=n_bootstrap, fit=fit)[subname]
    if test_groups:
        echo('Group A vs Group B:', pretty)
        print(r['whole'].to_string(index=False))
    expected_res = _report_diff(r['tests'], alpha, pretty)
    update_history(name, 'difftest', group_control=group_a, group_test=group_b, alpha=alpha, min_samples=min_samples, min_cover=min_cover,
                   mode=mode, subname=subname, test_groups=test_groups, max_cover=max_cover, filter_id=filter_id,
                   filter_chr=filter_chr, max_cover_group_test=max_cover_group_test, n_jobs=n_jobs,
                   param_window=param_window, logit_transform=logit_transform, robust_se=robust_se,
                   contrasts=contrasts, n_bootstrap=n_bootstrap, expected_result=expected_res, fit=fit)
    done(pretty, t0)
    return expected_res

@app.command('anova')
def _anova(name: str = Argument(..., help='Project name.'),
           groups: str = Option(str(), help='Files with lists of filenames, folder or a mask (masks should start with "[yellow]m:[/yellow]"'
                                             'prefix, e.g. "m:vcfs/*_M_*.vcf.gz"), each for a group. Entries must be separated with a semicolon ";" character. '
                                             'If not provided, then it is assumed that [b]combine[/b] was called previously with '
                                             '[yellow]subname[/yellow]s corresponding to group names (recommended).'),
           fit: str = Option(None, help='Path to a fit file from a different project. If not provided, fit from the same project is used.'),
           param_window: bool = Option(True, help='If disabled, parameters will be taken from a line with respect to the mean window for given'
                                                     ' reps/samples.'),
           es_mode: ESMode = Option(ESMode.entropy.value, help='Effect-size calculation method.'),
           alpha: float = Option(0.05, help='FDR level (Benjamini-Hochberg). Sets the threshold of the printed summary;'
                                            ' the adjusted p-values in the tables do not depend on it.'),
           min_samples: int = Option(2, help='Minimal number of samples/reps per an SNV to be considered for the analysis.'),
           min_groups: int = Option(2, help='Minimal number of groups present for an SNV to be considered for the analysis.'),
           min_cover: int = Option(None, help='Minimal required cover (ref + alt) for an SNV to be considered.'),
           max_cover: int = Option(None, help='Maximal allowed cover (ref + alt) for an SNV to be considered.'),
           subname: str = Option(None, help='You may give a result a subname in case you plan to draw multiple comparisons.'),
           n_jobs: int = Option(1, help='Number of jobs to be run at parallel, -1 will use all available cores.'),
           pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    ANOVA test via likelihood ratio.
    """
    groups = str(groups)
    es_mode = _value(es_mode)
    subname = str(subname) if subname else None
    if not fit:
        _require(name, 'fit', 'fit')
    t0 = time()
    with status('Performing ANOVA tests...', pretty):
        r = anova_test(name, groups=groups.split(';') if groups else None, min_samples=min_samples, min_cover=min_cover,
                       max_cover=max_cover, alpha=alpha, n_jobs=n_jobs, min_groups=min_groups, es_mode=es_mode,
                       param_mode='window' if param_window else 'line', fit=fit, subname=subname)[subname]
    expected_res = _report_diff(r['tests'], alpha, pretty)
    update_history(name, 'anova', groups=groups, min_samples=min_samples, min_cover=min_cover,
                   max_cover=max_cover, alpha=alpha, n_jobs=n_jobs, min_groups=min_groups,
                   param_window=param_window, es_mode=es_mode, fit=fit, subname=subname, expected_result=expected_res)
    done(pretty, t0)
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
                color_significant: bool = Option(True, help='Bars of groups with a p-value below 0.05 in the pairwise LRT test are colored'
                                                             ' orange, and green if the FDR is below 0.05 too.'),
                dpi: int = Option(200, help='DPI.'),
                fmt: str = Option('png', help='Image format.'),
                pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Plots effect sizes of the given SNVs from the results of the anova command.
    """
    if not ids and not snvs:
        raise typer.BadParameter('At least one of --ids and --snvs should be provided.')
    if ids:
        ids = list(map(tuple, _split_records(ids, '--ids', 'id,allele', 2)))
    if snvs:
        snvs_t = list()
        for chrom, pos, alt in _split_records(snvs, '--snvs', 'chromosome,position,allele', 3):
            try:
                snvs_t.append((chrom, int(pos), alt))
            except ValueError:
                raise typer.BadParameter(f'The position "{pos}" of "{chrom},{pos},{alt}" is not an integer.', param_hint='--snvs')
        snvs = snvs_t
    _require(name, 'anova', 'anova')
    if plot_test_es:
        _require(name, 'test', 'test or test_binom')
    t0 = time()
    with status('Plotting SNVs ESes...', pretty):
        plot.plot_anova_snvs(name, snv_names=ids, snvs=snvs, subname=subname, plot_raw_es=plot_raw_es, plot_test_es=plot_test_es,
                             plot_p_diff=plot_p_diff, color_significant=color_significant, folder=str(folder), ext=fmt,
                             dpi=dpi)
    done(pretty, t0)

@app_export.command('all', help='Export everything.')
def _export_all(name: str = Argument(..., help='Project name.'), out: Path = Argument(..., help='Output filename/path.'),
                sample_info: bool = Option(False, help='Include raw p-values and names of sample scorefiles to the tabular. '
                                                       'Note that this may bloat output if number of samples is high.'),
                pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    out = str(out)
    t0 = time()
    with status('Exporting tabular data...', pretty):
        export.export_all(name, out, sample_info=sample_info)
    update_history(name, 'export', out=out, sample_info=sample_info)
    done(pretty, t0)

@app_export.command('counts')
def _counts(name: str = Argument(..., help='Project name.'), out: str = Argument(..., help='Output filename/path.'),
            bad: float = Option(None, help='If provided, then only this particular BAD will be exported. In that case, [cyan]out[/cyan] is a '
                                           'filename, not a name of directory. Otherwise, counts from all bads will be exported to the'
                                           ' [cyan]out[/cyan] path.'),
            pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Export counts data.
    """
    out = str(out)
    export.export_counts(name, out, bad=bad)
    update_history(name, 'export counts', out=out, bad=bad)
    done(pretty)


def _check_bad_allele(bad, allele):
    if (bad is None) != (allele is None):
        raise typer.BadParameter('--bad and --allele should be provided together.')


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
    out = str(out)
    allele = _value(allele)
    _check_bad_allele(bad, allele)
    export.export_params(name, out, bad=bad, allele=allele)
    update_history(name, 'export params', out=out, bad=bad, allele=allele)
    done(pretty)


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
    allele = _value(allele)
    _check_bad_allele(bad, allele)
    export.export_stats(name, out, bad=bad, allele=allele)
    update_history(name, 'export indices', out=out, bad=bad, allele=allele)
    done(pretty)


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
    done(pretty)


@app_export.command('multiple_pvalues')
def _combined_multiple_pvalues(
                      project_name: str = Argument(..., help='Output project name.'),
                      names: List[str] = Argument(..., help='Project name.'),
                      out: Path = Argument(..., help='Output filename/path.'),
                      sample_info: bool = Option(False, help='Include raw p-values and names of sample scorefiles to the tabular. '
                                                          'Note that this may bloat output if number of samples is high.'),
                      subname: str = Option(None, help='A subname that can be used to reference a set of combined p-values in case if you'
                                                       ' provided one at [cyan bold]combine[/cyan bold] step.'),
                      pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    '''
    Export combined across samples and FDR-corrected p-values, aggregated across multiple projects.
    '''
    out = str(out)
    export.export_polycombined_pvalues(project_name, names, out, sample_info=sample_info, subname=subname)
    done(pretty)

help_str = 'Export "raw" pvalues (i.e. prior to combining them across samples and FDR-corrections). '\
           'Note that for each vcf/BED-like file submitted at [cyan bold]create[/cyan bold] stage, this command will '\
           'spawn a tabular file with similar name (including a relative path).'
@app_export.command('raw_pvalues', help=help_str)
def _raw_pvalues(name: str = Argument(..., help='Project name.'), out: Path = Argument(..., help='Output path.'),
                 pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    out = str(out)
    export.export_pvalues(name, out)
    update_history(name, 'export raw_pvalues', out=out)
    done(pretty)


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
    update_history(name, 'export difftest', out=out, subname=subname, sample_info=sample_info)
    done(pretty)

@app_export.command('anova')
def _anovas(name: str = Argument(..., help='Project name.'), out: Path = Argument(..., help='Output filename/path.'),
            subname: str = Option(None, help='A subname that can be used to reference a set of tests in case if you'
                                             ' provided one at [cyan bold]anova[/cyan bold] step.'),
            sample_info: bool = Option(False, help='Include names of sample scorefiles to the tabular. '
                                                   'Note that this may bloat output if number of samples is high.'),
            pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    '''
    Export FDR-corrected p-values for ANOVA.
    '''
    out = str(out)
    export.export_anova(name, out, subname=subname, sample_info=sample_info)
    update_history(name, 'export anova', out=out, subname=subname, sample_info=sample_info)
    done(pretty)

@app_export.command('demo')
def _demo(export_path: Path = Option(str(), help='Path where the demo data will be extracted.'),
          pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    '''
    Extract demonstration data from DNase-seq experiment for the cell line K562.
    '''
    export_path = str(export_path)
    export.export_demo(export_path)
    done(pretty)


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
    _require(name, 'fit', 'fit')
    t0 = time()
    with status('Plotting figures...', pretty):
        plot.visualize(name, out, what='all', max_count=max_count, slices=slices, fbad=bad, show_bad=show_bad, fmt=fmt, dpi=dpi)
    update_history(name, 'plot', out=out, max_count=max_count, slices=slices, bad=bad, show_bad=show_bad, fmt=fmt, dpi=dpi)
    done(pretty, t0)

@app.command('test_binom')
def _test_binom(name: str = Argument(..., help='Project name.'),
                beta: bool = Option(False, help='Use beta-binomial model instead of binomial. In that case, concentration paremeter will be'
                                                ' estimated from the data for each BAD.'),
                w: float = Option(None, help='Weight of the mixture component with the success probability 1 - p, p being BAD / (BAD + 1)'
                                             ' unless estimated. If not set, a single component with p = 1/2 is used everywhere.'),
                estimate_p: bool = Option(False, help='Estimate p, useful for negating reference bias.'),
                max_cover: int = Option(None, help='Maximal cover to be used for parameter estimation.'),
                n_jobs: int = Option(1, help='Number of jobs to be run at parallel, -1 will use all available cores.'),
                pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    """
    Calculate p-values with a non-bias adjusted left truncated binomial model.
    """
    # A history may hold w as a string, empty when unset.
    w = None if w in (None, '') else float(w)
    t0 = time()
    with status('Computing p-values and effect sizes...', pretty):
        _, params = binom_test(name, w=w, beta=beta, estimate_p=estimate_p, max_cover=max_cover,
                               n_jobs=n_jobs)
    if beta or estimate_p:
        # params[bad][allele] is (p, k), with k None for the binomial model.
        header = ['BAD'] + ['p_ref', 'p_alt'] * estimate_p + ['k_ref', 'k_alt'] * beta
        rows = list()
        for bad in sorted(params):
            row = ['{:.2f}'.format(bad)]
            for i, shown in enumerate((estimate_p, beta)):
                if shown:
                    row.extend('{:.3f}'.format(params[bad][allele][i]) for allele in ('ref', 'alt'))
            rows.append(row)
        echo('Estimated parameters:', pretty)
        echo_table(header, rows, pretty)
    update_history(name, 'test_binom', w=w, beta=beta, estimate_p=estimate_p, max_cover=max_cover, n_jobs=n_jobs)
    done(pretty, t0)

@app.command('reproduce', help='Reproduce a project as was recorded by a history file.')
def _reproduce(filename: Path = Argument(..., help='Path to the history JSON file.'),
               pretty: bool = Option(True, help='Use "rich" package to produce eye-candy output.')):
    filename = str(filename)
    reproduce(filename, pretty=pretty)

def main():
    check_packages()
    app()
