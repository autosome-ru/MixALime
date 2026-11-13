#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import logging
import pysam
import re
import os
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from rich.progress import track
from .utils import openers, parse_filenames, read_table, save_pickle, typed_column, PROJECT_FILE


# A file is unreadable once this many reads fail in a row, as a damaged file fails at the same place forever.
MAX_FAILURES_IN_A_ROW = 100


class SnpBadError(Exception):
    """An SNV is observed under two different BADs while snp_bad_check is on."""


@dataclass
class BEDRow:
    __slots__ = ['chrom', 'start', 'id', 'ref_count', 'alt_count', 'ref', 'alts', 'qual', 'db', 'bad']
    chrom: str
    start: int
    id: str
    ref_count: int
    alt_count: int
    ref: str
    alts: tuple
    qual: float
    db: bool
    bad: float


def _bed_columns(names) -> dict:
    """
    Index of the column that plays each role in a BED-like table, None for an absent optional role.

    Names are matched case-insensitively by their beginning and a column serves one role only. A ValueError is
    raised if a required role (chr, start, ref_count, alt_count, ref, alt) has no column.
    """
    names = [n.lower() for n in names]
    taken = set()

    def find(test):
        for i, n in enumerate(names):
            if i not in taken and test(n):
                taken.add(i)
                return i

    cols = dict(chr=find(lambda n: n.startswith(('#chr', 'chr'))),
                start=find(lambda n: n.startswith(('start', 'pos'))),
                ref_count=find(lambda n: n.startswith('ref') and 'count' in n),
                alt_count=find(lambda n: n.startswith('alt') and 'count' in n),
                ref=find(lambda n: n.startswith('ref')),
                alt=find(lambda n: n.startswith('alt')),
                qual=find(lambda n: n.startswith('qual')),
                id=find(lambda n: n in ('id', 'name')),
                db=find(lambda n: n == 'db'),
                bad=find(lambda n: n == 'bad'))
    missing = [role for role in ('chr', 'start', 'ref_count', 'alt_count', 'ref', 'alt') if cols[role] is None]
    if missing:
        raise ValueError('No column for: ' + ', '.join(missing) + '.')
    return cols


def _truthy(x) -> bool:
    if isinstance(x, str):
        return x.strip().lower() not in ('', '0', 'false', 'f', 'no', 'n', '.')
    return bool(x)


def _integral(x):
    """x as an int if it is an integral number or a text of one, otherwise None."""
    if isinstance(x, str):
        try:
            x = float(x)
        except ValueError:
            return None
    try:
        i = int(x)
    except (TypeError, ValueError, OverflowError):
        return None
    return i if i == x else None


def _float(x):
    """x as a float if it is a number or a text of one, otherwise None. NaN is no number."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def _position(x):
    """x as an int if it is integral, as a float if it is another number, otherwise None."""
    i = _integral(x)
    return i if i is not None else _float(x)


def _read_bedlike(filename: str) -> list:
    """
    Reads a BED-like table into a list of BEDRow.

    The columns chr, start, ref, alt, ref_count and alt_count are required, qual, id, db and bad are optional.
    Chromosome, id, ref and alt are read as strings whatever they look like, and an id or a BAD of '.' is None. A pos column
    is taken as start, with no shift from 1-based to 0-based. A table without rows raises a ValueError. Entries of
    qual and start are converted one by one, so a stray text spoils only the row that holds it: qual is None for an
    entry that is no number and start is None for one that is no integer.
    """
    df = read_table(filename)
    cols = _bed_columns(df.columns)
    n = df.height
    if n == 0:
        raise ValueError('The table has no rows.')
    text = {cols[role] for role in ('chr', 'id', 'ref', 'alt')}
    used = sorted(i for i in cols.values() if i is not None)
    data = {i: df[:, i].to_list() if i in text else typed_column(df[:, i]) for i in used}

    def column(role, default):
        i = cols[role]
        return data[i] if i is not None else [default] * n

    qual = [_float(q) for q in column('qual', float('inf'))]
    start = [_integral(s) for s in data[cols['start']]]
    db = [_truthy(x) for x in data[cols['db']]] if cols['db'] is not None else [None] * n
    alts = [None if a is None else (a, ) for a in data[cols['alt']]]
    return [BEDRow(*row) for row in zip(data[cols['chr']], start, [None if x == '.' else x for x in column('id', '')], data[cols['ref_count']],
                                        data[cols['alt_count']], data[cols['ref']], alts, qual, db,
                                        [None if x == '.' else x for x in column('bad', None)])]


def _count(x):
    """A non-negative integer read count or None. A text of a number is read as the number."""
    i = _integral(x)
    return i if i is not None and i >= 0 else None


def _is_base(allele) -> bool:
    return isinstance(allele, str) and len(allele) == 1 and allele.isalpha()


class _VcfSource:
    """Records of a VCF/BCF file."""
    def __init__(self, filename):
        self.file = pysam.VariantFile(filename, 'r')
        self.has_db = 'DB' in self.file.header.info

    def __iter__(self):
        return iter(self.file)

    def close(self):
        try:
            self.file.close()
        except OSError:
            pass

    @staticmethod
    def db(row):
        return bool(row.info.get('DB'))

    @staticmethod
    def bad(row):
        return None

    @staticmethod
    def observations(row):
        """
        Triplets (alt base, ref count, alt count), one per sample with a heterozygous ref/alt genotype, and a flag.

        AD is indexed by allele, so a genotype of the reference allele and the alternative allele j gives
        (AD[0], AD[j]) in whatever order and phasing the alleles are written. The flag is set if such a genotype has
        no usable AD. Other genotypes and the alternative alleles that are not single bases are left out.
        """
        alts = row.alts
        obs = list()
        broken = False
        for sample in row.samples.values():
            gt = sample.get('GT')
            if gt is None or len(gt) != 2:
                continue
            a, b = gt
            if a is None or b is None or a == b or (a != 0 and b != 0):
                continue
            j = a + b
            if j > len(alts) or not _is_base(alts[j - 1]):
                continue
            ad = sample.get('AD')
            if ad is None or len(ad) <= j:
                broken = True
                continue
            ref_count, alt_count = _count(ad[0]), _count(ad[j])
            if ref_count is None or alt_count is None:
                broken = True
                continue
            obs.append((alts[j - 1], ref_count, alt_count))
        return obs, broken


class _BedSource:
    """Rows of a BED-like table."""
    def __init__(self, filename):
        self.rows = _read_bedlike(filename)
        self.has_db = self.rows[0].db is not None

    def __iter__(self):
        return iter(self.rows)

    def close(self):
        pass

    @staticmethod
    def db(row):
        return row.db

    @staticmethod
    def bad(row):
        return float(row.bad) if isinstance(row.bad, str) else row.bad

    @staticmethod
    def observations(row):
        ref_count, alt_count = _count(row.ref_count), _count(row.alt_count)
        if ref_count is None or alt_count is None:
            return [], True
        alt = row.alts[0]
        return ([(alt, ref_count, alt_count)] if _is_base(alt) else []), False


def _open_source(filename):
    """VCF/BCF records or, if that fails, rows of a BED-like table; a file that is neither raises a ValueError."""
    try:
        return _VcfSource(filename)
    except Exception as e:
        vcf_error = e
    try:
        return _BedSource(filename)
    except Exception as e:
        raise ValueError(f'{filename} is neither a VCF ({vcf_error}) nor a BED-like table ({e}).') from e


def _mapped_bad(bad_maps, chrom, pos):
    """BAD of the position or None. The lookup adds nothing to bad_maps."""
    positions = bad_maps.get(chrom)
    if positions is None:
        return None
    try:
        return positions[pos]
    except KeyError:
        return None


def _missing(x) -> bool:
    return x is None or x != x or x == 0


def file_to_table(filename: str, counts=None, min_qual=10, min_cnt=5,
                  cnt_max_sum=1000, filter_db=True, filter_rs=True, filter_name=None,
                  filter_chr=None, bad_maps=None, default_bad=1.0,
                  drop_bads=None, snps_set=None, sample_counter=None, snps_pos=None,
                  snp_bad_check=True, filename_id=None):
    """
    Adds the (ref, alt) counts of a VCF/BED-like file to counts.

    A heterozygous genotype of the reference allele and an alternative allele gives the pair (ref, alt) of its
    allele depths, one observation per sample. BED-like rows give their ref_count and alt_count. The results of a
    file are added to counts, snps_set, sample_counter and snps_pos only after the whole file is processed. A
    file that cannot be read raises a ValueError and changes nothing. A record that is malformed is skipped and the
    number of such records is logged once per file.

    Parameters
    ----------
    filename : str
        Path to the VCF/BED-like file (possibly .gz-packed).
    counts : dict, optional
        Mapping BAD->{(ref, alt): number of occurrences}, updated in-place. If None, then it is created.
    min_qual : float, optional
        SNVs whose quality is below this are filtered. A missing quality passes only if min_qual is 0 or less.
        The default is 10.
    min_cnt : int, optional
        Observations whose ref count or alt count is below this are filtered. The default is 5.
    cnt_max_sum : int, optional
        Observations whose ref + alt counts are not below this are filtered. The default is 1000.
    filter_db : bool, optional
        If True, then SNVs that lack DB=True are filtered. A file without DB information (no DB in the VCF header,
        no DB column) is not filtered and a warning is logged. The default is True.
    filter_rs : bool, optional
        If True, then SNV IDs should start with 'rs'. The default is True.
    filter_name : str, optional
        If provided, then SNV IDs should match this regex-pattern from their beginning (re.match). An SNV
        without an ID is filtered. The default is None.
    filter_chr : str, optional
        If provided, then the chromosome name should match this regex-pattern from its beginning (re.match), so
        'chr\\d+' keeps 'chr1_KI270706v1_random' too. A pattern that ends with $ matches the whole name. The default is None.
    bad_maps : dict, optional
        Mapping chr->position->BAD, see read_bad_maps. Chromosome names are strings. The default is None.
    default_bad : float, optional
        The BAD value which is used if no BAD maps are provided or if the particular SNV is not
        found in those BAD maps. If None, then those SNVs are filtered. The default is 1.0.
    drop_bads : list, optional
        List of BADs to be excluded from the further analysis. The default is None.
    snps_set : dict, optional
        A mapping BAD->set used for counting unique SNVs. The default is None.
    sample_counter : dict, optional
        A mapping BAD->int that counts the observations. The default is None.
    snps_pos : dict, optional
        A mapping (chromosome, start, alt)->list of (id, ref, alt) followed by (filename_id, ref count, alt count, BAD)
        for each observation. The default is None.
    snp_bad_check : bool, optional
        Require SNVs to come from the same BAD. A SnpBadError is raised otherwise. The default is True.
    filename_id : int, optional
        Index of the file in the list of scorefiles, required if snps_pos is provided.

    Returns
    -------
    counts : dict
        Mapping BAD->{(ref, alt): number of occurrences}.

    """
    assert bad_maps or default_bad, 'Either bad_maps or default_bad must be provided.'
    drop_bads = {round(float(x), 5) for x in drop_bads} if drop_bads else set()
    if counts is None:
        counts = dict()
    if filter_name is not None:
        filter_name = re.compile(filter_name)
    if filter_chr is not None:
        filter_chr = re.compile(filter_chr)
    f_counts, f_samples, f_pos, f_snps, first_bad = dict(), dict(), dict(), dict(), dict()
    skipped = 0
    save = pysam.set_verbosity(0)
    source = None
    try:
        source = _open_source(filename)
        if filter_db and not source.has_db:
            logging.warning(f'{filename} has no DB information, SNVs are not filtered by DB.')
            filter_db = False
        rows = iter(source)
        failures = 0
        while True:
            try:
                row = next(rows)
            except StopIteration:
                break
            except Exception as e:
                skipped += 1
                failures += 1
                if failures >= MAX_FAILURES_IN_A_ROW:
                    raise ValueError(f'{filename} cannot be read: {e}') from e
                continue
            failures = 0
            try:
                ref, alts, chrom, start, name = row.ref, row.alts, row.chrom, row.start, row.id
                if chrom is None or start is None:
                    raise ValueError('No position.')
                if len(ref) != 1 or not alts:
                    continue
                qual = row.qual
                if qual is None:
                    if min_qual > 0:
                        raise ValueError('No quality.')
                elif qual < min_qual:
                    continue
                if filter_db and not source.db(row):
                    continue
                if filter_chr and not filter_chr.match(chrom):
                    continue
                if filter_rs and (not name or not name.startswith('rs')):
                    continue
                if filter_name and (not name or not filter_name.match(name)):
                    continue
                bad = source.bad(row)
                if _missing(bad):
                    bad = default_bad
                if bad_maps is not None:
                    mapped = _mapped_bad(bad_maps, chrom, start)
                    if mapped is not None:
                        bad = mapped
                    elif not default_bad:
                        continue
                bad = round(float(bad), 5)
                if bad in drop_bads:
                    continue
                obs, broken = source.observations(row)
            except Exception:
                skipped += 1
                continue
            if broken:
                skipped += 1
            for alt, ref_count, alt_count in obs:
                if not (ref_count >= min_cnt and alt_count >= min_cnt and ref_count + alt_count < cnt_max_sum):
                    continue
                key = (chrom, start, alt)
                if snps_pos is not None:
                    fb = first_bad.get(key)
                    if fb is None:
                        previous = snps_pos.get(key)
                        fb = first_bad[key] = previous[1][-1] if previous else bad
                    if snp_bad_check and fb != bad:
                        raise SnpBadError(f'SNV at {chrom}:{start} comes from at least two different BADs: {bad} and {fb}.'
                                          ' If that is expected, consider setting --no-snp-bad-check flag.')
                    lt = f_pos.get(key)
                    if lt is None:
                        lt = f_pos[key] = [(name, ref, alt)]
                    lt.append((filename_id, ref_count, alt_count, bad))
                d = f_counts.setdefault(bad, dict())
                d[(ref_count, alt_count)] = d.get((ref_count, alt_count), 0) + 1
                f_samples[bad] = f_samples.get(bad, 0) + 1
                f_snps.setdefault(bad, set()).add(key)
    finally:
        if source is not None:
            source.close()
        pysam.set_verbosity(save)
    if skipped:
        logging.warning(f'{filename}: {skipped} records were skipped because of missing or malformed fields.')
    for bad, d in f_counts.items():
        total = counts.setdefault(bad, dict())
        for pair, n in d.items():
            total[pair] = total.get(pair, 0) + n
    if sample_counter is not None:
        for bad, n in f_samples.items():
            sample_counter[bad] = sample_counter.get(bad, 0) + n
    if snps_set is not None:
        for bad, keys in f_snps.items():
            snps_set.setdefault(bad, set()).update(keys)
    if snps_pos is not None:
        for key, lt in f_pos.items():
            total = snps_pos.setdefault(key, list())
            if not total:
                total.append(lt[0])
            total.extend(lt[1:])
    return counts


class IntervalMap:
    """
    Maps a position to the value of the last interval that contains it.

    intervals is a sequence of (start, end, value). Where intervals overlap the later one wins. The ends are
    excluded if start_open or end_open is set. A KeyError is raised for a position outside all intervals.
    Construction takes O(n log n) time and a lookup O(log n).
    """
    def __init__(self, intervals, start_open=False, end_open=True):
        # An end is a pair (coordinate, side) and a position x is (x, 1), so every interval is the half-open
        # range [lo, hi) of these pairs whatever its ends are.
        lo_side = 2 if start_open else 0
        hi_side = 0 if end_open else 2
        items = list()
        for i, (start, end, value) in enumerate(intervals):
            lo, hi = (start, lo_side), (end, hi_side)
            if lo < hi:
                items.append((lo, hi, i, value))
        items.sort(key=lambda t: t[0])
        points = sorted({t[0] for t in items} | {t[1] for t in items})
        self._lo, self._hi, self._val = list(), list(), list()
        winners = list()
        heap = list()
        k = 0
        for p, q in zip(points, points[1:]):
            while k < len(items) and items[k][0] <= p:
                lo, hi, i, value = items[k]
                heappush(heap, (-i, hi, value))
                k += 1
            while heap and heap[0][1] <= p:
                heappop(heap)
            if not heap:
                continue
            i = -heap[0][0]
            if winners and winners[-1] == i and self._hi[-1] == p:
                self._hi[-1] = q
            else:
                winners.append(i)
                self._lo.append(p)
                self._hi.append(q)
                self._val.append(heap[0][2])

    def __getitem__(self, pos):
        key = (pos, 1)
        i = bisect_right(self._lo, key) - 1
        if i >= 0 and key < self._hi[i]:
            return self._val[i]
        raise KeyError(pos)

    def __len__(self):
        return len(self._lo)


def read_bad_maps(filename: str, start_open=False, end_open=True) -> dict:
    """
    Reads a BAD map from a table with the columns chr, start, bad and, optionally, end.

    Without an end column a row sets the BAD of the position start. With it, a row sets the BAD of the interval
    from start to end, excluding an end if start_open or end_open says so, [start, end) by default. Where intervals
    overlap, the later row wins. Chromosome names are read as strings. Rows with a missing value or an entry that is
    no number (no integer for a position) are ignored and their number is logged once.
    Returns a dict chr->object that maps a position to its BAD and raises KeyError for an unmapped one.
    """
    df = read_table(filename)
    names = df.columns

    def find(test, role, required=True):
        col = next((c for c in names if test(c.lower())), None)
        if col is None and required:
            raise ValueError(f'The BAD map {filename} has no {role} column.')
        return col

    chr_col = find(lambda n: 'chr' in n, 'chr')
    start_col = find(lambda n: 'start' in n, 'start')
    end_col = find(lambda n: 'end' in n, 'end', required=False)
    bad_col = find(lambda n: n == 'bad', 'bad')
    cols = (chr_col, start_col, end_col, bad_col) if end_col else (chr_col, start_col, bad_col)
    rows = zip(*(df[c].to_list() if c == chr_col else typed_column(df[c]) for c in cols))
    d = dict()
    ignored = 0
    if end_col is None:
        for chrom, pos, bad in rows:
            pos, bad = _integral(pos), _float(bad)
            if chrom is None or pos is None or bad is None:
                ignored += 1
            else:
                d.setdefault(chrom, dict())[pos] = bad
    else:
        intervals = dict()
        for chrom, start, end, bad in rows:
            start, end, bad = _position(start), _position(end), _float(bad)
            if chrom is None or start is None or end is None or bad is None:
                ignored += 1
            else:
                intervals.setdefault(chrom, list()).append((start, end, bad))
        d = {chrom: IntervalMap(lt, start_open, end_open) for chrom, lt in intervals.items()}
    if ignored:
        logging.warning(f'{filename}: {ignored} rows of the BAD map were ignored because of missing or malformed values.')
    return d


def _remove_project_files(name: str, keep: str):
    """Removes the init, fit, test, comb, difftest and anova files of the project name, except the file keep."""
    folder, base = os.path.split(name)
    keep = os.path.abspath(keep)
    for file in os.listdir(folder if folder else None):
        m = PROJECT_FILE.fullmatch(file)
        path = os.path.join(folder, file)
        if m is not None and m['name'] == base and os.path.isfile(path) and os.path.abspath(path) != keep:
            os.remove(path)


def _counts_array(d: dict, symmetrify: bool):
    """
    Array of rows (ref, alt, number of occurrences) sorted by ref and alt from a mapping (ref, alt)->number.

    If symmetrify is set, a pair and its mirror both get half of their summed number, so the total
    stays the same. The array is float then, as a weight can be x.5.
    """
    if not symmetrify:
        return np.array([(ref, alt, d[(ref, alt)]) for ref, alt in sorted(d)], dtype=int)
    lt = list()
    for ref, alt in sorted(set(d) | {(alt, ref) for ref, alt in d}):
        n = d.get((ref, alt), 0) + d.get((alt, ref), 0)
        if n > 0:
            lt.append((ref, alt, n / 2))
    return np.array(lt, dtype=float)


def create_project(name: str, snvs: list, bad_maps=None, drop_bads=None,
                   compression='lzma', min_qual=10, min_cnt=5,
                   max_cover=1000, filter_db=False, symmetrify=False,
                   filter_rs=False, filter_name=None, filter_chr=None,
                   default_bad=1.0, count_snvs=False, snp_bad_check=True,
                   progress_bar=True):
    """
    Initialize MixALime projects initial files.

    The files that cannot be read are skipped with a warning. The files of a previous project of the same name
    (init, fit, test, comb, difftest and anova ones) are removed only after the input is processed. A ValueError
    is raised, with the previous files left alone, if no input file can be read or if no observation passes the
    filters.

    Parameters
    ----------
    name : str
        Project name, possibly with a folder part that must exist.
    snvs : list or str
        A list of, or a single, path to a VCF/BED-like file, to a folder (its regular files except index files and
        hidden files, no subfolders), to a file with a list of those paths or a mask with the prefix "m:".
    bad_maps : dict or str, optional
        BAD maps in the format of chr->position->BAD or a path to a table for read_bad_maps. If None, then all SNVs
        will assume default_bad. The default is None.
    drop_bads : list, optional
        List of BADs to be excluded from the further analysis. The default is None.
    compression : str, optional
        Compression type to use, can be lzma, gzip, bz2 or raw (no compression). The default is 'lzma'.
    min_qual : float, optional
        SNVs whose quality is below this are filtered. A missing quality passes only if min_qual is 0 or less.
        The default is 10.
    min_cnt : int, optional
        SNVs whose ref_count or alt_count is below this are filtered. The default is 5.
    max_cover : int, optional
        SNVs whose ref + alt counts are not below this are filtered. The default is 1000.
    filter_db : bool, optional
        If True, then SNVs that lack DB=True are filtered. A file without DB information is not filtered and
        a warning is logged. The default is False.
    symmetrify : bool, optional
        If True, then counts are symmetrified: a pair (ref, alt) and its mirror (alt, ref) both get half of
        their summed number of occurrences (x.5 is possible). The default is False.
    filter_rs : bool, optional
        If True, then SNV IDs should start with 'rs'. The default is False.
    filter_name : str, optional
        If provided, then SNV IDs should match this regex-pattern from their beginning (re.match). The default is None.
    filter_chr : str, optional
        If provided, then the chromosome name should match this regex-pattern from its beginning (re.match), so
        'chr\\d+' keeps 'chr1_KI270706v1_random' too. A pattern that ends with $ matches the whole name. The default is None.
    default_bad : float, optional
        The default BAD value which is used if no BAD maps are provided or if the particular SNV is not
        found in those BAD maps. If None, then those SNVs are filtered. The default is 1.0.
    snp_bad_check : bool, optional
        Require SNVs to come from the same BAD. The default is True.
    count_snvs : bool, optional
        If True, samples and SNVs are counted and the counts are returned. The default is False.
    progress_bar : bool, optional
        If True, then a progress bar is shown. The default is True.

    Returns
    -------
    res : dict
        Mapping (scorefiles, snvs, counts) -> data. It is stored in the file name.init.compression.
    sample_counter : dict
        Mapping BAD -> sample counts.
    snps_set : dict
        Mapping BAD -> set of unique SNVs. Not None only if count_snvs is True.

    """
    if compression not in openers:
        raise ValueError(f'Unknown compression {compression}, available: ' + ', '.join(openers) + '.')
    folder = os.path.split(name)[0]
    if folder and not os.path.isdir(folder):
        raise FileNotFoundError(f'The folder {folder} of the project {name} does not exist.')
    if isinstance(bad_maps, (str, os.PathLike)) and bad_maps:
        bad_maps = read_bad_maps(str(bad_maps))
    snvs = parse_filenames(snvs)
    data_filename = f'{name}.init.{compression}'
    counts = dict()
    snps_set = defaultdict(set) if count_snvs else None
    sample_counter = defaultdict(int)
    scorefiles = list()
    snps_pos = dict()
    its = track(snvs, description='Processing files...') if progress_bar else snvs
    for filename in its:
        try:
            file_to_table(filename, counts, bad_maps=bad_maps, min_qual=min_qual, min_cnt=min_cnt,
                          cnt_max_sum=max_cover, filter_db=filter_db, drop_bads=drop_bads,
                          filter_rs=filter_rs, filter_name=filter_name, filter_chr=filter_chr,
                          default_bad=default_bad, snps_set=snps_set, sample_counter=sample_counter,
                          snps_pos=snps_pos, filename_id=len(scorefiles), snp_bad_check=snp_bad_check)
        except ValueError as e:
            logging.warning(f'Incorrect file {filename} is skipped: {e}')
            continue
        scorefiles.append(filename)
    if not scorefiles:
        raise ValueError('No input file could be read, the project is not created.')
    if not counts:
        raise ValueError('No observation passed the filters, the project is not created.')
    res = {'counts': {bad: _counts_array(counts[bad], symmetrify) for bad in sorted(counts)},
           'scorefiles': scorefiles,
           'snvs': snps_pos}
    _remove_project_files(name, data_filename)
    save_pickle(res, data_filename)
    return res, sample_counter, snps_set
