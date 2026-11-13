# -*- coding: utf-8 -*-
from .combine import project_signature
from .utils import get_init_file, load_pickle, closest_bad, openers, scorefiles_qc
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import logging
import tarfile
import re
import os


_PVALUE_EXT = '.pvalue.tsv'
_SEX_CHROMS = {'X': 0, 'Y': 1, 'M': 2, 'MT': 2}


def _is_name(project) -> bool:
    return isinstance(project, (str, os.PathLike))


def _load_init(project) -> dict:
    return load_pickle(get_init_file(os.fspath(project)))


def _load_part(project, kind: str):
    """Reads the results of the stage kind from the files of a project."""
    project = os.fspath(project)
    compression = get_init_file(project).split('.')[-1]
    filename = f'{project}.{kind}.{compression}'
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'No {filename}: the project has no {kind} results.')
    return load_pickle(filename)


def _by_subname(results: dict, subname, source: str):
    """
    The entry of results stored under subname.

    Without a subname, the entry stored without one is taken, or the only
    entry. A KeyError listing the subnames is raised if there is no match.
    """
    if subname in results:
        return results[subname]
    available = ', '.join(sorted(map(repr, results))) or 'none'
    if subname:
        raise KeyError(f'No subname {subname!r} in {source}, the subnames are: {available}.')
    if None in results:
        return results[None]
    if len(results) == 1:
        return next(iter(results.values()))
    raise KeyError(f'No subname is given and {source} has several, the subnames are: {available}.')


def _make_parent(filename):
    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)


def _file_stem(subname) -> str:
    """A subname that is safe as a part of a filename."""
    return re.sub(r'[\\/]', '_', str(subname)).replace('..', '_')


def _table_names(subnames, default: str) -> dict:
    """
    File name of the table of each subname.

    A result without a subname is named default. The other names are the file stems of their subnames, with
    _1, _2, ... added where a name is taken, so that no table replaces another.
    """
    names = dict()
    taken = set()
    for subname in sorted(subnames, key=bool):
        stem = _file_stem(subname) if subname else default
        name, k = stem, 0
        while os.path.normcase(name) in taken:
            k += 1
            name = f'{stem}_{k}'
        taken.add(os.path.normcase(name))
        names[subname] = f'{name}.tsv'
    return names


def _bad_folders(bads) -> dict:
    """Folder name of each BAD: two decimals, with more decimals for the BADs that would share a name."""
    bads = sorted({float(bad) for bad in bads})
    digits = dict.fromkeys(bads, 2)
    for _ in range(16):
        names = {bad: f'BAD{bad:.{digits[bad]}f}' for bad in bads}
        taken = Counter(names.values())
        shared = [bad for bad in bads if taken[names[bad]] > 1]
        if not shared:
            return names
        for bad in shared:
            digits[bad] += 1
    return {bad: f'BAD{bad!r}' for bad in bads}


def _chrom_key(chrom) -> tuple:
    """Sort key of a chromosome name: numbered ones, then X, Y and M, then other names alphabetically."""
    name = str(chrom)
    short = name[3:] if name[:3].lower() == 'chr' else name
    if short.isascii() and short.isdigit():
        return 0, int(short), name
    if short.upper() in _SEX_CHROMS:
        return 1, _SEX_CHROMS[short.upper()], name
    return 2, 0, name


def _sort_genomic(df: pd.DataFrame) -> pd.DataFrame:
    """Rows ordered by chromosome (chr1, chr2, ..., chr10, ..., chrX) and start; rows at one position keep their order."""
    if df.empty:
        return df
    rank = {c: i for i, c in enumerate(sorted(set(df['#chr']), key=_chrom_key))}
    order = np.lexsort((df['start'].to_numpy(), df['#chr'].map(rank).to_numpy()))
    return df.iloc[order].reset_index(drop=True)


def _numeric(values) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)


def _prefer_ref(ref, alt, larger: bool = False) -> tuple:
    """
    Masks of the rows where ref is the preferred allele and where an allele has a value.

    A nan is a missing value. If both values are present, the smaller one is
    preferred (the larger one with larger=True), alt on a tie.
    """
    ref, alt = _numeric(ref), _numeric(alt)
    ref_ok, alt_ok = ~np.isnan(ref), ~np.isnan(alt)
    with np.errstate(invalid='ignore'):
        better = ref > alt if larger else ref < alt
    return ref_ok & (~alt_ok | better), ref_ok | alt_ok


def _take(use_ref, found, ref, alt) -> np.ndarray:
    """Values of ref where use_ref holds and of alt elsewhere, nan where found does not hold."""
    res = np.where(use_ref, _numeric(ref), _numeric(alt))
    res[~found] = np.nan
    return res


def _label(use_ref, found, ref_label: str, alt_label: str) -> np.ndarray:
    return np.where(found, np.where(use_ref, ref_label, alt_label), '')


def _logit2(p):
    return np.log2(p) - np.log1p(-p) / np.log(2)


def _mean_log_ratio(samples: list) -> float:
    """Mean over samples of log2(alt / ref) of their counts."""
    if not samples:
        return float('nan')
    ref = np.array([t[1] for t in samples], dtype=float)
    alt = np.array([t[2] for t in samples], dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.mean(np.log2(alt) - np.log2(ref)))


def _counts_table(counts, names: tuple) -> pd.DataFrame:
    """Table of (ref, alt, n) rows. The column n is integer unless it holds halves."""
    counts = np.asarray(counts).reshape(-1, 3)
    n = counts[:, 2]
    if np.array_equal(n, np.round(n)):
        n = n.astype(np.int64)
    return pd.DataFrame({names[0]: counts[:, 0].astype(np.int64), names[1]: counts[:, 1].astype(np.int64), names[2]: n})


def export_counts(project, out: str, bad: float = None):
    """
    Writes the tables of counts. Without bad, out is a folder with a subfolder
    for each BAD, with bad it is the name of a file.
    """
    counts_d = _load_init(project)['counts'] if _is_name(project) else project['counts']
    if bad is None:
        folders = _bad_folders(counts_d)
        for key, counts in counts_d.items():
            subfolder = os.path.join(out, folders[float(key)])
            os.makedirs(subfolder, exist_ok=True)
            _counts_table(counts, ('ref', 'alt', 'n')).to_csv(os.path.join(subfolder, 'counts.tsv'), sep='\t', index=False)
    else:
        table = _counts_table(counts_d[closest_bad(counts_d, bad)], ('Ref', 'Alt', 'N'))
        _make_parent(out)
        table.to_csv(out, sep='\t', index=False)


def export_scorefiles_qc(project, out: str):
    """Writes the bias and the cover of each scorefile at each BAD and at all BADs together."""
    project = _load_init(project) if _is_name(project) else project
    covers, biases = scorefiles_qc(project)
    scorefiles = project['scorefiles']
    folders = _bad_folders(project['counts'])
    for bad in sorted(project['counts']) + [None]:
        subfolder = out if bad is None else os.path.join(out, folders[float(bad)])
        os.makedirs(subfolder, exist_ok=True)
        rows = [(scorefiles[f], biases[f][bad], covers[f][bad]) for f in covers if covers[f][bad]]
        pd.DataFrame(rows, columns=['name', 'bias', 'cover']).to_csv(os.path.join(subfolder, 'scorefiles_qc.tsv'),
                                                                      sep='\t', index=False)


def _fit_bads(fit: dict) -> set:
    return {bad for allele in ('ref', 'alt') for bad in fit.get(allele, ())}


def _export_params(fit, out: str, allele: str, bad: float):
    df = pd.DataFrame(fit[allele][bad]['params'])
    df = df.rename(columns={'names': 'Name', 'ests': 'Estimate', 'stds': 'Std'})
    df.to_csv(out, sep='\t', index=False)


def _export_stats(fit, out: str, allele: str, bad: float):
    columns = defaultdict(list)
    for i, indices in fit[allele][bad]['stats'].items():
        columns['slice'].append(i)
        for name, value in indices.items():
            columns[name].append(value)
    pd.DataFrame(columns).to_csv(out, sep='\t', index=False)


def _export_fit_tables(project, out: str, bad, allele, export_one, prefix: str):
    if bad is not None and allele is None:
        raise ValueError('Both BAD and allele should be supplied.')
    fit = _load_part(project, 'fit') if _is_name(project) else project
    if bad is not None:
        if not fit.get(allele):
            raise KeyError(f'No allele {allele!r} in the fit.')
        bad = closest_bad(fit[allele], bad)
        _make_parent(out)
        export_one(fit, out, allele, bad)
        return
    os.makedirs(out, exist_ok=True)
    folders = _bad_folders(_fit_bads(fit))
    for allele in ('ref', 'alt'):
        if not fit.get(allele):
            logging.warning(f'No allele {allele} in the fit. Perhaps an experimental model was used?')
            continue
        for bad in fit[allele]:
            subfolder = os.path.join(out, folders[float(bad)])
            os.makedirs(subfolder, exist_ok=True)
            export_one(fit, os.path.join(subfolder, f'{prefix}_{allele}.tsv'), allele, bad)


def export_params(project, out: str, bad: float = None, allele: str = None):
    """Writes the parameter estimates of the fit of each BAD and allele, or of one allele at one BAD to the file out."""
    _export_fit_tables(project, out, bad, allele, _export_params, 'param')


def export_stats(project, out: str, bad: float = None, allele: str = None):
    """Writes the fit indices of each BAD and allele, or of one allele at one BAD to the file out."""
    _export_fit_tables(project, out, bad, allele, _export_stats, 'stat')


def get_name(name: str):
    s = name.split('.')
    if len(s) == 1:
        return name
    for i, c in enumerate(s[::-1]):
        if c not in ('gz', 'vcf', 'bam', 'tsv', 'csv', 'zip', 'lzma', 'bz2', 'bed'):
            break
    if not i:
        return name
    return '.'.join(s[:-i])


def shorten_filenames(filenames: list):
    """Filenames without the leading folders that all of them share. A single filename stays as it is."""
    its = [f.split('/') for f in filenames]
    if len(its) < 2:
        return list(filenames)
    i = 0
    for folders in zip(*[t[:-1] for t in its]):
        if len(set(folders)) > 1:
            break
        i += 1
    return ['/'.join(t[i:]) for t in its]


def _pvalue_paths(scorefiles: list) -> list:
    """
    Path of the table of raw p-values of each scorefile, relative to the output folder.

    The paths are distinct and stay inside the output folder: parts that are
    empty or are '.', '..' or a drive are dropped. A name equal to an earlier
    one keeps its full original name or gets the index of the scorefile.
    """
    paths = list()
    taken = set()
    for index, scorefile in enumerate(scorefiles):
        *folders, base = re.split(r'[\\/]+', scorefile)
        folders = [f for f in folders if f not in ('', '.', '..')]
        if folders and re.fullmatch(r'[A-Za-z]:', folders[0]):
            folders = folders[1:]
        stem = get_name(base)
        k = 0
        candidates = [stem, base, f'{stem}.{index}']
        while True:
            path = os.path.join(*folders, candidates.pop(0) + _PVALUE_EXT)
            if os.path.normcase(path) not in taken:
                break
            if not candidates:
                k += 1
                candidates.append(f'{base}.{index}.{k}')
        taken.add(os.path.normcase(path))
        paths.append(path)
    return paths


def export_pvalues(project, out: str):
    """Writes a table of raw p-values for each scorefile of the project, to the folder out."""
    if _is_name(project):
        init, test = _load_init(project), _load_part(project, 'test')
    else:
        init, test = project
    scorefiles = shorten_filenames(init['scorefiles'])
    columns = ['#chr', 'start', 'end', 'bad', 'id', 'ref', 'alt', 'ref_count', 'alt_count', 'ref_es', 'alt_es', 'ref_pval',
               'alt_pval']
    rows = dict()
    for (chrom, pos, _), its in init['snvs'].items():
        name, ref, alt = its[0]
        for file_id, ref_count, alt_count, bad in its[1:]:
            # A BAD that the fit skipped has no p-values.
            if bad not in test['ref']:
                continue
            counts = (ref_count, alt_count)
            pval_ref, es_ref = test['ref'][bad][counts]
            pval_alt, es_alt = test['alt'][bad][counts]
            rows.setdefault(file_id, list()).append((chrom, pos, pos + 1, bad, name, ref, alt, ref_count, alt_count,
                                                     es_ref, es_alt, pval_ref, pval_alt))
    paths = _pvalue_paths(scorefiles)
    for file_id, table in rows.items():
        df = pd.DataFrame(table, columns=columns)
        use_ref, found = _prefer_ref(df['ref_pval'], df['alt_pval'])
        df['es'] = _take(use_ref, found, df['ref_es'], df['alt_es'])
        df['pval'] = _take(use_ref, found, df['ref_pval'], df['alt_pval'])
        filename = os.path.join(out, paths[file_id])
        _make_parent(filename)
        _sort_genomic(df).to_csv(filename, sep='\t', index=False)


def _combined_table(rows: list, sample_info: bool) -> pd.DataFrame:
    columns = ['#chr', 'start', 'end', 'mean_bad', 'id', 'max_cover', 'ref', 'alt', 'n_reps']
    if sample_info:
        columns += ['bads', 'scorefiles', 'ref_counts', 'alt_counts', 'ref_es', 'alt_es', 'ref_pval', 'alt_pval']
    columns += ['ref_comb_es', 'alt_comb_es', 'ref_comb_pval', 'alt_comb_pval', 'ref_fdr_comb_pval', 'alt_fdr_comb_pval']
    df = pd.DataFrame(rows, columns=columns)
    use_ref, found = _prefer_ref(df['ref_comb_pval'], df['alt_comb_pval'])
    df['pref_allele'] = _label(use_ref, found, 'ref', 'alt')
    df['comb_es'] = _take(use_ref, found, df['ref_comb_es'], df['alt_comb_es'])
    df['comb_pval'] = _take(use_ref, found, df['ref_comb_pval'], df['alt_comb_pval'])
    df['fdr_comb_pval'] = _take(use_ref, found, df['ref_fdr_comb_pval'], df['alt_fdr_comb_pval'])
    return _sort_genomic(df)


def _combined_row(chrom, pos, meta: tuple, samples: list, stats: tuple, sample_info: bool) -> list:
    """
    A row of the table of combined p-values.

    A sample is (scorefile, ref_count, alt_count, bad, (ref_pval, ref_es), (alt_pval, alt_es)).
    """
    (pval_ref, pval_alt), (es_ref, es_alt), (fdr_ref, fdr_alt) = stats
    bads = [s[3] for s in samples]
    name, ref, alt = meta
    row = [chrom, pos, pos + 1, sum(bads) / len(bads), name, max(s[1] + s[2] for s in samples), ref, alt, len(samples)]
    if sample_info:
        row += [','.join(map(str, bads)), ','.join(s[0] for s in samples),
                ','.join(str(s[1]) for s in samples), ','.join(str(s[2]) for s in samples),
                ','.join(str(s[4][1]) for s in samples), ','.join(str(s[5][1]) for s in samples),
                ','.join(str(s[4][0]) for s in samples), ','.join(str(s[5][0]) for s in samples)]
    return row + [es_ref, es_alt, pval_ref, pval_alt, fdr_ref, fdr_alt]


def export_combined_pvalues(project, out: str, sample_info=False, subname=None):
    """Writes the p-values combined across samples with their FDR to the file out. Rows are sorted by position."""
    if _is_name(project):
        init, comb = _load_init(project), _load_part(project, 'comb')
        test = _load_part(project, 'test') if sample_info else None
    else:
        init, test, comb = project
    comb = _by_subname(comb, subname, 'the combined p-values')
    groups = comb['groups']
    if isinstance(groups, (list, tuple)):
        raise ValueError('These p-values are combined across projects, the multiple_pvalues export writes them.')
    scorefiles = shorten_filenames(init['scorefiles'])
    comb_snvs = comb['snvs']
    rows = list()
    for key, its in init['snvs'].items():
        stats = comb_snvs.get(key)
        if stats is None:
            continue
        samples = list()
        for file_id, ref_count, alt_count, bad in its[1:]:
            if groups and file_id not in groups:
                continue
            if sample_info:
                counts = (ref_count, alt_count)
                samples.append((scorefiles[file_id], ref_count, alt_count, bad, test['ref'][bad][counts],
                                test['alt'][bad][counts]))
            else:
                samples.append((scorefiles[file_id], ref_count, alt_count, bad))
        if samples:
            rows.append(_combined_row(key[0], key[1], its[0], samples, stats, sample_info))
    _make_parent(out)
    _combined_table(rows, sample_info).to_csv(out, sep='\t', index=False)


def _check_names(comb_file: str, recorded: list, projects: list):
    """Raises a ValueError unless the base names of projects are the recorded ones, in that order."""
    given = [os.path.basename(os.fspath(project)) for project in projects]
    names = [signature['name'] for signature in recorded]
    if given != names:
        raise ValueError(f'{comb_file} is combined from the projects {", ".join(names)} in this order, '
                         f'but {", ".join(given)} are given.')


def _check_signature(comb_file: str, recorded: dict, signature: dict):
    """Raises a ValueError if the scorefiles or the test results of a project differ from the recorded ones."""
    for key, what in {'scorefiles': 'scorefiles', 'test': 'test results'}.items():
        if signature[key] != recorded[key]:
            raise ValueError(f'The {what} of the project {recorded["name"]} are not those {comb_file} is combined from.'
                             ' Run multiple_combine again.')


def _find_comb_file(output_name: str) -> str:
    folder, name = os.path.split(output_name)
    search_dir = folder if folder else '.'
    for file in sorted(os.listdir(search_dir)):
        if file.startswith(f'{name}.comb.') and file.endswith(tuple(openers)):
            return os.path.join(search_dir, file)
    raise FileNotFoundError(f'Could not find .comb file for {output_name}')


def export_polycombined_pvalues(output_name: str, projects: list, out: str, sample_info=False, subname=None):
    """
    Writes the p-values combined across several projects to the file out.

    The projects are those of the multiple_combine call, in its order. A row
    lists the samples of the SNV in the groups of the projects, or all samples
    if every group is empty, except those without test results. If the result records
    its projects, a ValueError is raised for projects of other names, order, scorefiles
    or test results.
    """
    comb_file = _find_comb_file(output_name)
    comb = _by_subname(load_pickle(comb_file), subname, comb_file)
    groups = comb['groups']
    if not isinstance(groups, (list, tuple)):
        raise ValueError(f'{comb_file} does not hold p-values combined across projects, the pvalues export writes them.')
    if len(groups) != len(projects):
        raise ValueError(f'{comb_file} holds p-values combined across {len(groups)} projects, but {len(projects)} are given.')
    recorded = comb.get('projects')
    if recorded is not None:
        _check_names(comb_file, recorded, projects)
    everything = not any(groups)
    snvs, tests, scorefiles = list(), list(), list()
    for i, project in enumerate(projects):
        init = _load_init(project)
        snvs.append(init['snvs'])
        tag = os.path.basename(os.fspath(project))
        scorefiles.append([f'[{tag}] {f}' for f in shorten_filenames(init['scorefiles'])])
        tests.append(_load_part(project, 'test'))
        if recorded is not None:
            compression = get_init_file(os.fspath(project)).split('.')[-1]
            _check_signature(comb_file, recorded[i], project_signature(os.fspath(project), init['scorefiles'], compression))
    rows = list()
    missing = 0
    for key, stats in comb['snvs'].items():
        meta = None
        samples = list()
        for i, project_snvs in enumerate(snvs):
            its = project_snvs.get(key)
            if its is None:
                continue
            if meta is None:
                meta = its[0]
            for file_id, ref_count, alt_count, bad in its[1:]:
                if not everything and file_id not in groups[i]:
                    continue
                counts = (ref_count, alt_count)
                try:
                    ref_stat, alt_stat = tests[i]['ref'][bad][counts], tests[i]['alt'][bad][counts]
                except KeyError:
                    missing += 1
                    continue
                samples.append((scorefiles[i][file_id], ref_count, alt_count, bad, ref_stat, alt_stat))
        if samples:
            rows.append(_combined_row(key[0], key[1], meta, samples, stats, sample_info))
    if missing:
        logging.warning(f'{missing} samples have no test results in their projects and are left out.')
    _make_parent(out)
    _combined_table(rows, sample_info).to_csv(out, sep='\t', index=False)


def _load_diff(project, kind: str, subname, init, sample_info: bool) -> tuple:
    """The results of a stage under subname and, if sample_info needs them, the init data."""
    if _is_name(project):
        results = _load_part(project, kind)
        if init is None and sample_info:
            init = _load_init(project)
    else:
        first, results = project
        if init is None and sample_info:
            init = first
    if sample_info and not (isinstance(init, dict) and 'scorefiles' in init):
        raise ValueError('Init data should be passed to the export function!')
    return _by_subname(results, subname, f'the {kind} results'), init


def export_anova(project, out: str, subname=None, sample_info=False, init=None):
    """
    Writes the anova results to the file out.

    If the table has the column bad, a row is the SNV at that BAD: mean_bad is
    this BAD and the scorefiles are those of its samples. Otherwise the
    scorefiles are all samples of the SNV and mean_bad is the BAD of the last
    sample of the first group that has the SNV.
    """
    diff, init = _load_diff(project, 'anova', subname, init, sample_info)
    tests, groups = diff['tests'].reset_index(drop=True), diff['snvs']
    has_bad = 'bad' in tests.columns
    n_cols = [c for c in tests.columns if c.startswith('n_')]
    if len(n_cols) < len(groups):
        raise ValueError(f'The table has {len(n_cols)} group sizes for {len(groups)} groups.')
    scorefiles = shorten_filenames(init['scorefiles']) if sample_info else None
    bads = tests['bad'].tolist() if has_bad else None
    rows = list()
    files = list()
    for i, ind in enumerate(tests['ind']):
        by_group = [g.get(ind) or list() for g in groups]
        meta = next((s[0] for s in by_group if s), None)
        if meta is None:
            raise KeyError(f'The SNV {ind} is in none of the groups.')
        samples = [s[1:] for s in by_group]
        if has_bad:
            bad = bads[i]
            samples = [[t for t in s if t[-1] == bad] for s in samples]
        else:
            bad = next((s[-1][-1] for s in samples if s), float('nan'))
        rows.append([ind[0], ind[1], ind[1] + 1, bad, *meta])
        if sample_info:
            names = [','.join(scorefiles[t[0]] for t in s) for s in samples]
            files.append(names + [','.join(n for n in names if n)])
    df = pd.DataFrame(rows, columns=['#chr', 'start', 'end', 'mean_bad', 'id', 'ref', 'alt'])
    if sample_info:
        cols = ['scorefiles_' + c[2:] for c in n_cols[:len(groups)]]
        if 'n_all' in n_cols[len(groups):]:
            cols.append('scorefiles_all')
        df = pd.concat([df, pd.DataFrame([f[:len(cols)] for f in files], columns=cols)], axis=1)
    df = pd.concat([df, tests.drop(columns=['ind', 'bad'] if has_bad else ['ind'])], axis=1)
    use_ref, found = _prefer_ref(df['ref_pval'], df['alt_pval'])
    df['scoring_model'] = _label(use_ref, found, 'ref|alt', 'alt|ref')
    df['pval'] = _take(use_ref, found, df['ref_pval'], df['alt_pval'])
    df['fdr_pval'] = _take(use_ref, found, df['ref_fdr_pval'], df['alt_fdr_pval'])
    _make_parent(out)
    _sort_genomic(df).to_csv(out, sep='\t', index=False)


def export_difftests(project, out: str, sample_info=False, subname=None, init=None):
    """
    Writes the results of the differential test to the file out.

    If the table has the column bad, a row is the SNV at that BAD: mean_bad is
    this BAD, and the counts, the scorefiles and es_count are those of its
    samples. Otherwise they are those of all samples of the SNV.
    """
    diff, init = _load_diff(project, 'difftest', subname, init, sample_info)
    tests = diff['tests'].reset_index(drop=True)
    snvs_a, snvs_b = diff['snvs']
    has_bad = 'bad' in tests.columns
    scorefiles = shorten_filenames(init['scorefiles']) if sample_info else None
    bads = tests['bad'].tolist() if has_bad else None
    rows = list()
    es_count = list()
    for i, ind in enumerate(tests['ind']):
        samples_a, samples_b = snvs_a[ind][1:], snvs_b[ind][1:]
        if has_bad:
            bad = bads[i]
            samples_a = [t for t in samples_a if t[-1] == bad]
            samples_b = [t for t in samples_b if t[-1] == bad]
        else:
            row_bads = [t[-1] for t in samples_a + samples_b]
            bad = sum(row_bads) / len(row_bads)
        es_count.append(_mean_log_ratio(samples_b) - _mean_log_ratio(samples_a))
        row = [ind[0], ind[1], ind[1] + 1, bad, *snvs_a[ind][0]]
        if sample_info:
            row += [','.join(str(t[k]) for t in s) for s in (samples_a, samples_b) for k in (1, 2)]
            row += [','.join(scorefiles[t[0]] for t in s) for s in (samples_a, samples_b)]
        rows.append(row)
    columns = ['#chr', 'start', 'end', 'mean_bad', 'id', 'ref', 'alt']
    if sample_info:
        columns += ['a_ref_counts', 'a_alt_counts', 'b_ref_counts', 'b_alt_counts', 'scorefiles_a', 'scorefiles_b']
    df = pd.DataFrame(rows, columns=columns)
    df = pd.concat([df, tests.drop(columns=['ind', 'bad'] if has_bad else ['ind'])], axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        for allele in ('ref', 'alt'):
            p_control, p_test = _numeric(df[f'{allele}_p_control']), _numeric(df[f'{allele}_p_test'])
            df[f'{allele}_es_p'] = np.log2(p_test) - np.log2(p_control)
            df[f'{allele}_es_logit'] = _logit2(p_test) - _logit2(p_control)
    use_ref, found = _prefer_ref(df['ref_p_control'], df['alt_p_control'], larger=True)
    df['preferred_allele_control'] = _label(use_ref, found, 'ref', 'alt')
    use_ref, found = _prefer_ref(df['ref_p_test'], df['alt_p_test'], larger=True)
    df['preferred_allele_test'] = _label(use_ref, found, 'ref', 'alt')
    use_ref, found = _prefer_ref(df['ref_pval'], df['alt_pval'])
    df['scoring_model'] = _label(use_ref, found, 'ref|alt', 'alt|ref')
    df['es_count'] = es_count
    for col in ('es_p', 'es_logit', 'pval', 'fdr_pval'):
        df[col] = _take(use_ref, found, df[f'ref_{col}'], df[f'alt_{col}'])
    _make_parent(out)
    _sort_genomic(df).to_csv(out, sep='\t', index=False)


def export_all(name: str, out: str, sample_info: bool = None):
    """
    Writes every table the project has results for.

    A stage without a file is skipped. If a stage fails, the others are still
    written and the first error is raised at the end.
    """
    init = _load_init(name)
    compression = get_init_file(name).split('.')[-1]
    errors = list()

    def attempt(label, fun, *args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except Exception as e:
            logging.error(f'Export of {label} failed: {e!r}')
            errors.append(e)

    def has(kind):
        return os.path.isfile(f'{name}.{kind}.{compression}')

    attempt('counts', export_counts, init, out)
    attempt('scorefiles quality', export_scorefiles_qc, init, out)
    if has('fit'):
        fit = attempt('fit', _load_part, name, 'fit')
        if fit is not None:
            attempt('parameters', export_params, fit, out)
            attempt('fit indices', export_stats, fit, out)
    for kind, export, folder, default in (('difftest', export_difftests, 'difftest', 'difftests'),
                                          ('anova', export_anova, 'anova', 'anova')):
        results = attempt(kind, _load_part, name, kind) if has(kind) else None
        for subname, filename in _table_names(results or (), default).items():
            attempt(f'{kind} {subname}', export, (init, results), os.path.join(out, folder, filename),
                    subname=subname, sample_info=sample_info, init=init)
    raw = attempt('test', _load_part, name, 'test') if has('test') else None
    if raw is not None:
        folder = os.path.join(out, 'pvalues')
        attempt('raw p-values', export_pvalues, (init, raw), os.path.join(folder, 'raw'))
        combined = attempt('comb', _load_part, name, 'comb') if has('comb') else None
        for subname, filename in _table_names(combined or (), 'pvals').items():
            attempt(f'combined p-values {subname}', export_combined_pvalues, (init, raw, combined),
                    os.path.join(folder, filename), subname=subname, sample_info=sample_info)
    if errors:
        raise errors[0]


def export_demo(path: str = str()):
    folder = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data')
    filename = os.path.join(folder, 'dnase_k562.tar.gz')
    with tarfile.open(filename, 'r:gz') as f:
        f.extractall(os.path.join(path, 'scorefiles'))
