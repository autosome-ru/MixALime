#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import Counter, defaultdict
from fnmatch import fnmatch
from glob import glob
from multiprocessing import get_context
import numpy as np
import polars as pl
import logging
import pickle
import dill
import lzma
import gzip
import bz2
import csv
import os
import re
import zipfile
from .params import dictify_params  # noqa: F401


class _DropBackendNotes(logging.Filter):
    """Drops the notes of JAX on the absence of a GPU."""

    notes = ('No GPU/TPU found, falling back to CPU.', 'a CUDA-enabled jaxlib is not installed')

    def filter(self, record):
        msg = record.getMessage()
        return not any(note in msg for note in self.notes)


# Installed at import, so that the worker processes have it as well.
logging.getLogger("jax._src.xla_bridge").addFilter(_DropBackendNotes())


def process_pool(n_jobs: int, **kwargs):
    """
    A pool of n_jobs processes started by a forkserver, as a fork of a process
    with a live JAX runtime is not safe. Unless the environment says otherwise,
    the workers run JAX on the CPU: several of them do not share a GPU well.
    kwargs go to Pool. A script that starts a pool needs a main guard.

    A worker spends about a second on its imports, so a pool is best kept for
    all the work of a command. The modules are not preloaded by the
    forkserver: in Python 3.11 it imports them before it takes the sys.path
    of the parent, and may find another copy of MixALime.
    """
    os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
    return get_context('forkserver').Pool(n_jobs, **kwargs)


openers = {'lzma': lzma.open,
           'gzip': gzip.open,
           'bz2': bz2.open,
           'raw': open}


def save_pickle(obj, filename: str):
    """
    Writes obj to filename with the compression named by the extension of filename.

    The data goes to a temporary file that replaces filename at the end, so
    an interrupted write leaves the previous file intact. Plain containers
    are written by pickle, anything it refuses by dill. Both are read by
    load_pickle.
    """
    filename = os.fspath(filename)
    compression = filename.split('.')[-1]
    # A low preset costs some file size and saves most of the compression time.
    kwargs = {'preset': 1} if compression == 'lzma' else dict()
    tmp = f'{filename}.{os.getpid()}.tmp'
    try:
        with openers[compression](tmp, 'wb', **kwargs) as f:
            try:
                data = pickle.dumps(obj, protocol=4)
            except (pickle.PicklingError, AttributeError, TypeError):
                data = dill.dumps(obj)
            f.write(data)
        os.replace(tmp, filename)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_pickle(filename: str):
    """Reads an object written by save_pickle or by dill.dump into a file opened by one of the openers."""
    filename = os.fspath(filename)
    with openers[filename.split('.')[-1]](filename, 'rb') as f:
        return dill.load(f)


def get_init_file(path: str) -> str:
    """Path to the file with the parsed data of the project at path. A FileNotFoundError is raised if there is none."""
    folder, name = os.path.split(path)
    if not folder or os.path.isdir(folder):
        for file in sorted(os.listdir(folder if folder else None)):
            if file.startswith(f'{name}.init.') and file.endswith(tuple(openers.keys())) and os.path.isfile(os.path.join(folder, file)):
                return os.path.join(folder, file)
    raise FileNotFoundError(f'No project {path} is found: the create command makes one.')


def closest_bad(bads, bad: float, tol=0.01) -> float:
    """
    The element of bads equal to bad up to tol.

    A BAD like 4/3 is stored with more digits than are typed or shown in
    folder names. A KeyError is raised if no element is that close.
    """
    bads = list(bads)
    if bad in bads:
        return bad
    best = min(bads, key=lambda x: abs(x - bad), default=None)
    if best is None or abs(best - bad) > tol:
        raise KeyError(f'No BAD {bad} in the project, the BADs are: ' + ', '.join(map(str, sorted(bads))))
    return best


def dictify_fix(s: str) -> dict:
    """Parses fixed parameters given as 'name1=value1;name2=value2' into a dict of floats."""
    if type(s) is not str:
        return s
    res = dict()
    for t in s.replace(' ', '').split(';'):
        if not t:
            continue
        try:
            name, value = t.split('=')
            res[name] = float(value)
        except ValueError:
            raise ValueError(f'Can not parse "{t}": fixed parameters are given as name=number, separated by semicolons.')
    return res


def scorefiles_qc(init) -> tuple:
    covers = defaultdict(lambda: defaultdict(int))
    biases = defaultdict(lambda: defaultdict(list))
    for _, its in init['snvs'].items():
        its = its[1:]
        for it in its:
            ind, ref, alt = it[:3]
            bad = it[-1]
            covers[ind][bad] += ref + alt
            covers[ind][None] += ref + alt
            biases[ind][bad].append(ref > alt)
            biases[ind][None].append(ref > alt)
    for ind, bads in biases.items():
        for bad in bads:
            biases[ind][bad] = np.mean(biases[ind][bad])
    return covers, biases

# Index files that go with variant files.
_INDEX_SUFFIXES = ('.tbi', '.csi')
# File names that are taken for data files without looking into them.
_DATA_SUFFIXES = ('.gz', '.bgz', '.vcf', '.bcf', '.bam')
# Output files of a MixALime project: name.kind.compression.
PROJECT_FILE = re.compile(r'(?P<name>.+)\.(?:init|fit|test|comb|difftest|anova)\.(?:' + '|'.join(openers) + ')')


def select_filenames(patterns: list, files) -> list:
    """
    Selects from files (as stored in a project) those that agree with any of the patterns.

    A pattern is a mask with the prefix "m:" (fnmatch), a file, a folder, a file that lists files or a folder
    on the disk. Paths are compared after os.path.normpath on both sides, so './data/ctrl', 'data//ctrl' and
    'data/ctrl/' agree. The files are returned as they are spelled in files.
    """
    if isinstance(patterns, (str, os.PathLike)):
        patterns = [patterns]
    files = list(files)
    norm = [os.path.normpath(str(f)) for f in files]
    spelled = dict()
    for f, n in zip(files, norm):
        spelled.setdefault(n, f)
    res = list()
    for pattern in patterns:
        pattern = str(pattern)
        if not pattern:
            subres = []
        elif pattern.startswith('m:'):
            t = os.path.normpath(pattern[2:])
            subres = [f for f, n in zip(files, norm) if fnmatch(n, t)]
        else:
            p = os.path.normpath(pattern)
            if p in spelled:
                subres = [spelled[p]]
            elif p == '.':
                subres = [f for f, n in zip(files, norm) if not os.path.isabs(n) and n != '..' and not n.startswith('..' + os.sep)]
            else:
                t = p if p.endswith(os.sep) else p + os.sep
                subres = [f for f, n in zip(files, norm) if n.startswith(t)]
        if not subres:
            for r in parse_filenames(pattern, files_list=files, ignore_errors=True):
                f = spelled.get(os.path.normpath(r))
                if f is not None:
                    subres.append(f)
            if not subres:
                logging.error(f'No files agree with the pattern {pattern}.')
        res.extend(subres)
    return res


def _is_variant_file(file: str) -> bool:
    """Whether pysam opens file as a VCF/BCF."""
    try:
        import pysam
    except ImportError:
        return False
    save = pysam.set_verbosity(0)
    try:
        with pysam.VariantFile(file):
            return True
    except Exception:
        return False
    finally:
        pysam.set_verbosity(save)


# The part of a file that is looked at to tell the separator.
_HEAD = 1 << 20


def _read_bytes(filename: str, size=-1) -> bytes:
    """The first size bytes (all by default) of a file, unpacked by its suffix: .gz, .bz2, .xz or .zip (its first member)."""
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename) as z, z.open(z.namelist()[0]) as f:
            return f.read(size)
    opener = {'.gz': gzip.open, '.bz2': bz2.open, '.xz': lzma.open}.get(os.path.splitext(filename)[1], open)
    with opener(filename, 'rb') as f:
        return f.read(size)


def _split(line: str, sep: str) -> list:
    """Fields of a line, quotes respected. A run of spaces is one separator if sep is a space."""
    if sep == ' ':
        return line.split()
    return next(csv.reader([line], delimiter=sep)) if '"' in line else line.split(sep)


def _sniff(lines: list) -> tuple:
    """
    The separator that splits most of the lines into the same number of fields (above one), and that number.

    The separator is None and the number is 1 if there is no such one. A tie goes to the earlier of comma, pipe,
    semicolon and tab, as a space is tried only if none of them splits, and a tie of counts to the later line.
    """
    best, top = None, (0, 0)
    for sep in (',', '|', ';', '\t', ' '):
        if sep == ' ' and best is not None:
            break
        n, agree = Counter(len(_split(line, sep)) for line in reversed(lines)).most_common(1)[0]
        if n > 1 and (agree, n) > top:
            best, top = sep, (agree, n)
    return best, top[1] or 1


def read_table(filename: str, header=True, n_rows=None) -> pl.DataFrame:
    """
    Reads a delimited table into a polars DataFrame of strings, where a missing value ('NA') is None.

    The separator is told from the first 100 lines by _sniff, a table without one has a single column. Blank lines
    and lines that start with "# " or "; " are skipped on top. A header one name short gets the name 'index' first,
    and without a header the columns are C0, C1 and so on. A file is unpacked by its suffix. With n_rows only the
    beginning of the file is read. An empty file gives an empty DataFrame.
    """
    data = _read_bytes(filename, _HEAD if n_rows else -1).removeprefix(b'\xef\xbb\xbf')
    if b'\n' not in data[:_HEAD]:
        data = data.replace(b'\r', b'\n')
    text = data[:_HEAD].decode('utf-8', 'replace').replace('\r', '').split('\n')
    if len(data) > _HEAD:
        text.pop()
    skip = next((i for i, line in enumerate(text) if line.strip() and not re.match(r'[#;](\s|$)', line)), None)
    if skip is None:
        return pl.DataFrame()
    lines = [line for line in text[skip:] if line.strip()][:100]
    sep, n = _sniff(lines)
    sep = sep or '\x1f'
    names = [x.strip() for x in _split(lines[0], sep)] if header else []
    if header and len(names) == n - 1:
        names.insert(0, 'index')
    names = [x or f'C{i}' for i, x in enumerate((names + [''] * n)[:n])]
    first = {x: i for i, x in reversed(list(enumerate(names)))}
    names = [x if first[x] == i else f'{x}.{i}' for i, x in enumerate(names)]
    if sep == ' ':
        data = b'\n'.join(b' '.join(line.split()) for line in data.split(b'\n'))
    rows = data.rstrip(b' \r\n').split(b'\n', skip + header)[skip + header:]
    df = pl.read_csv(rows[0] if rows else b'', separator=sep, has_header=False, n_rows=n_rows, schema=dict.fromkeys(names, pl.String),
                     encoding='utf8-lossy', raise_if_empty=False)
    return df.select(pl.all().fill_null('').str.strip_chars().replace('NA', None))


def typed_column(column) -> list:
    """Values of a column of strings as ints if all of them are integers, as floats if all are numbers, else as strings."""
    blank = column.replace('', None)
    ints = blank.cast(pl.Int64, strict=False)
    if ints.null_count() == blank.null_count():
        return ints.to_list()
    floats = blank.cast(pl.Float64, strict=False)
    if floats.null_count() == blank.null_count():
        return floats.fill_nan(None).to_list()
    return column.to_list()


def _expand_file(file: str, files_list, ignore_errors: bool) -> list:
    """
    Paths that a file stands for: itself, or its entries if it lists files.

    A file lists files if it is a table of one column whose first entry is an existing path (relative to the
    folder of the file) or, when files_list is given, a name from files_list. A table without rows is dropped
    with a warning.
    """
    if file.endswith(_DATA_SUFFIXES) or _is_variant_file(file):
        return [file]
    folder = os.path.dirname(file)
    try:
        df = read_table(file, n_rows=1)
        if df.width == 1:
            df = read_table(file, header=False, n_rows=1)
        if df.height == 0:
            logging.warning(f'File {file} is a table without rows and is skipped.')
            return []
        r = str(df[0, 0])
        if df.width == 1:
            names = {os.path.normpath(str(f)) for f in files_list} if files_list is not None else set()
            is_paths = os.path.isfile(os.path.join(folder, r))
            if is_paths or os.path.normpath(r) in names:
                listed = [str(x) for x in read_table(file, header=False)[:, 0].to_list()]
                if not is_paths:
                    return listed
                res = list()
                for entry in listed:
                    path = os.path.join(folder, entry)
                    if not os.path.isfile(path) and not ignore_errors:
                        logging.error(f'File {path} not found.')
                    else:
                        res.append(path)
                return res
    except Exception:
        pass
    return [file]


def parse_filenames(files: list, files_list=None, ignore_errors=False) -> list:
    """
    Expands files to a sorted list of paths where every file is present once.

    An item is a data file, a folder, a file that lists data files or a mask with the prefix "m:" that is expanded
    by glob. A folder gives its regular files except index files (.tbi, .csi), hidden files, .json files and the
    output files of MixALime projects. Folders are not searched recursively. Files are told apart by their
    normalised absolute paths and the first spelling is kept. A missing file is logged as an error unless
    ignore_errors is set. The names in a file that lists names from files_list are returned as they are.
    """
    if isinstance(files, str) or not isinstance(files, (list, tuple, set)):
        files = [files]
    res = list()
    for file in files:
        file = str(file)
        if file.startswith('m:'):
            res.extend(sorted(glob(file[len('m:'):])))
        elif os.path.isdir(file):
            for f in sorted(os.listdir(file)):
                path = os.path.join(file, f)
                if (os.path.isfile(path) and not f.startswith('.') and not f.endswith(_INDEX_SUFFIXES + ('.json', ))
                        and PROJECT_FILE.fullmatch(f) is None):
                    res.append(path)
        elif os.path.isfile(file):
            res.extend(_expand_file(file, files_list, ignore_errors))
        elif not ignore_errors:
            logging.error(f'File {file} not found.')
    seen = set()
    unique = list()
    for file in res:
        key = os.path.abspath(file)
        if key not in seen:
            seen.add(key)
            unique.append(file)
    return sorted(unique)
