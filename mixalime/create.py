#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datatable as dt
import numpy as np
import portion
import logging
import pysam
import dill
import re
from collections import defaultdict
from .utils import openers, parse_filenames
from dataclasses import dataclass
from rich.progress import track
import os

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


def _read_bedlike(filename: str):
    df = dt.fread(filename)
    names = [n.lower() for n in df.names]
    met = set()
    chr_col = next((i for i in range(len(names)) if names[i].startswith(('#chr', 'chr'))))
    met.add(chr_col)
    start_col = next((i for i in range(len(names)) if i not in met and names[i].startswith(('start', 'pos'))))
    met.add(start_col)
    ref_count_col = next((i for i in range(len(names)) if i not in met and (names[i].startswith('ref') and 'count' in names[i])))
    met.add(ref_count_col)
    alt_count_col = next((i for i in range(len(names)) if i not in met and (names[i].startswith('alt') and 'count' in names[i])))
    met.add(alt_count_col)
    ref_col = next((i for i in range(len(names)) if i not in met and names[i].startswith('ref')))
    met.add(ref_col)
    alt_col = next((i for i in range(len(names)) if i not in met and names[i].startswith('alt')))
    met.add(alt_col)
    try:
        qual_col = next((i for i in range(len(names)) if i not in met and names[i].startswith('qual')))
    except StopIteration:
        qual_col = len(names)
        names.append('qual')
        df['qual'] = float('inf')
    met.add(qual_col)
    try:
        id_col = next((i for i in range(len(names)) if i not in met and names[i] in ('id', 'name')))
    except StopIteration:
        id_col = len(names)
        names.append('id')
        df['id'] = str()
    met.add(id_col)
    try:
        db_col = next((i for i in range(len(names)) if i not in met and names[i] == 'db'))
    except StopIteration:
        db_col = len(names)
        names.append('db')
        df['db'] = False
    met.add(db_col)
    try:
        bad_col = next((i for i in range(len(names)) if i not in met and names[i] == 'bad'))
    except StopIteration:
        bad_col = len(names)
        names.append('bad')
        df['db'] = 0
    met.add(db_col)
    inds = (chr_col, start_col, id_col, ref_count_col, alt_count_col,  ref_col, alt_col, qual_col, db_col, bad_col)
    res = list()
    for i in range(df.shape[0]):
        it = [t[0] for t in df[i, inds].to_list()]
        it[-4] = tuple(it[-4])
        it[1] += 1
        res.append(BEDRow(*it))
    return res


def file_to_table(filename: str, counts=None, min_qual=10, min_cnt=5,
                 cnt_max_sum=1000, filter_db=True, filter_rs=True, filter_name=None,
                 filter_chr=None, bad_maps=None, default_bad=1.0,
                 drop_bads=None, snps_set=None, sample_counter=0, snps_pos=None,
                 filename_id=None):
    """
    Transforms a VCF/BED-like file to a pair of mixalime & Pandas friendly meta dictionary and a numeric (ref, alt, num_occurences) aux. numeric table.

    Parameters
    ----------
    filename : str
        Path to the VCF/BED-like file (possibly .gz-packed).
    counts : dict, optional
        Mapping BAD->numeric dataframe of (ref, alt, num_occurences). It will be updated in-place if provided. If None,
        then it is instantiated. The default is None.
    min_qual : float, optional
        SNVs whose quality is below this are filtered. The default is 10.
    min_cnt : int, optional
        SNVs whose ref_count or alt_count is below this are filtered. The default is 5.
    cnt_max_sum : int, optional
        SNVs whose ref + alt counts are above this are filtered. The default is inf.
    filter_db : bool, optional
        If True, then SNVs that lack DB=True are filtered. The default is True.
    filter_rs : bool, optional
        If True, then SNV IDs should start with 'rs'. The default is True.
    filter_name : str, optional
        If provided, then SNV IDs should match this regex-pattern. The default is None.
    filter_chr : str, optional
        If provided, then chr column should math this regex-pattern. Possible applications
        omitting scaffold data, i.e. those chr whose pattern doesnt align with the 'chr\d+'
        pattern.
    bad_maps : IntervalDict or dict, optional
        Mapping chr->pos->BAD. The default is None.
    default_bad : float, optional
        The default BAD value which is used if no BAD maps are provided or if the particular SNV is not
        found in those BAD maps. If None, then those SNVs are filtered. The default is 1.0.
    drop_bads : list, optional
        List of BADs to be excluded from the further analysis. The default is None.
    snps_set : dict, optional
        A mapping BAD->set used for counting unique SNVs. The default is None.
    sample_counter : int, optional
        A mapping BAD->int counter used to count for total number of data samples. The default is None.
    snps_pos : defaultdict, optional
        A mapping BAD->(chromosome, pos)->list of (filename, ref, alt). The default is None.
    filename_id : int, optional
        If snps_pos is provided, then filename_id is also must be provided, which is an index in a sorted list of scorfiles. The default
        is None.

    Returns
    -------
    table : pd.DataFrame
        Pandas DataFrame containing both counts and meta information.
    counts : dict
        Mapping BAD->[Mapping (ref, alt) -> num_occurences].

    """
    assert bad_maps or default_bad, 'Either bad_maps or default_bad must be provided.'
    if not drop_bads:
        drop_bads = list()
    drop_bads = list(map(lambda x: round(x, 5), drop_bads))
    if counts is None:
        counts = defaultdict(lambda: defaultdict(int))
    # table = {'#CHROM': list(), 'START': list(), 'END': list(), 'ID': list(),
    #          'REF': list(), 'ALT': list(), 'REF_COUNTS': list(), 'ALT_COUNTS': list(),
    #          'BAD': list()}
    if filter_name is not None:
        filter_name = re.compile(filter_name)
    if filter_chr is not None:
        filter_chr = re.compile(filter_chr)
    save = pysam.set_verbosity(0)
    try:
        file = pysam.VariantFile(filename, 'r')
    except ValueError:
        try:
            file = _read_bedlike(filename)
        except StopIteration:
            raise ValueError
    for row in file:
        if row.qual < min_qual or len(row.alts) != 1 or len(row.ref) != 1:
            continue
        if filter_db:
            try:
                try:
                    if not row.db:
                        continue
                except AttributeError:
                    if not row.info['DB']:
                        continue
            except KeyError:
                pass
        if len(row.alts[0]) != 1:
            continue
        chrom = row.chrom
        if filter_chr and not filter_chr.match(str(chrom)):
            continue
        name = row.id
        if filter_rs and (not name or not name.startswith('rs')):
            continue
        if filter_name and not filter_name.match(name):
            continue
        start = row.start
        # end = start + 1
        try:
            bad = row.bad if row.bad else default_bad
        except AttributeError:
            bad = default_bad
        if bad_maps is not None:
            try:
                bad = bad_maps[chrom][start]
            except KeyError:
                if not default_bad:
                     continue
        bad = round(bad, 5)
        if bad in drop_bads:
            continue
        try:
            it = row.samples.values()
        except AttributeError:
            it = [{'GT': (0, 1), 'AD': (row.ref_count, row.alt_count)}]
        flag = False
        for sample in it:
            a, b = sample['GT']
            if a == None or (a == b):
                continue
            ac, bc = sample['AD']
            ref = (1 - a) * ac + (1 - b) * bc
            alt = a * ac + b * bc
            if ref >= min_cnt and alt >= min_cnt and (ref + alt < cnt_max_sum):
                counts[bad][(ref, alt)] += 1
                # table['#CHROM'].append(chrom)
                # table['ID'].append(name)
                # table['START'].append(start)
                # table['END'].append(end)
                # table['REF'].append(row.ref)
                # table['ALT'].append(','.join(row.alts))
                # table['REF_COUNTS'].append(ref)
                # table['ALT_COUNTS'].append(alt)
                # table['BAD'].append(bad)
                flag = True
                if sample_counter is not None:
                    sample_counter[bad] += 1
                if snps_pos is not None:
                    lt = snps_pos[(chrom, start)]
                    if not lt:
                        lt.append((bad, name, row.ref, ','.join(row.alts)))
                    else:
                        if lt[0][0] != bad:
                            raise Exception(f'SNV at {chrom}:{start} comes from at least two different BADs: {bad} and {lt[0][0]}.')
                    lt.append((filename_id, ref, alt))
        if flag and snps_set is not None:
            snps_set[bad].add((chrom, start))
    pysam.set_verbosity(save)
    return counts

def read_bad_maps(filename: str, start_open=False, end_open=True, ):
    df = dt.fread(filename)
    chr_col = next((c for c in df.names if 'chr' in c.lower()))
    start_col = next((c for c in df.names if 'start' in c.lower()))
    try:
        end_col = next((c for c in df.names if 'end' in c.lower()))
    except StopIteration:
        end_col = None
    bad_col = next((c for c in df.names if 'bad' == c.lower()))
    names  = (chr_col, start_col, end_col, bad_col) if end_col else (chr_col, start_col, bad_col)
    df = df[:, names].to_tuples()
    if end_col is None:
        d = defaultdict(dict)
        for chr, pos, bad in df:
            d[chr][pos] = bad
    else:
        if start_open == end_open:
            interval = 'open' if end_open else 'closed'
        else:
            interval = 'open' if start_open else 'closed'
            interval = interval + 'open' if end_open else 'closed'
        interval = getattr(portion, interval)
        d = defaultdict(portion.IntervalDict)
        for chr, start_pos, end_pos, bad in df:
            d[chr][interval(start_pos, end_pos)] = bad
    return d
    

def create_project(name: str, snvs: list, bad_maps=None, drop_bads=None,
                   compression='lzma', min_qual=10, min_cnt=5,
                   max_cover=1000, filter_db=False, symmetrify=False,
                   filter_rs=False, filter_name=None, filter_chr=None,
                   default_bad=1.0, count_snvs=False, progress_bar=True):
    """
    Initialize MixALime projects initial files.

    Parameters
    ----------
    name : str
        Project name. 
    snvs : list or str
        Can be either a list of paths to VCF files, a single VCF file or a path to folder where vcf files are stored.
    bad_maps : dict, optional
        BAD maps in the format of chr->interval->bad. If None, then all SNVs will assume default_bad. The default is None.
    compression : str, optional
        Compression type to use, can be lzma, gzip, bz2 or raw (no compression). The default is 'lzma'.
    min_qual : float, optional
        SNVs whose quality is below this are filtered. The default is 10.
    min_cnt : int, optional
        SNVs whose ref_count or alt_count is below this are filtered. The default is 5.
    max_cover : int, optional
        SNVs whose ref + alt counts are above this are filtered. The default is inf.
    filter_db : bool, optional
        If True, then SNVs that lack DB=True are filtered. The default is False.
    symmetrify : bool, optional
        If True, then counts are symmetrified, i.e. (ref, alt) = (alt, ref) for each pair of ref, alt. The default is False.
    filter_rs : bool, optional
        If True, then SNV IDs should start with 'rs'. The default is False.
    filter_name : str, optional
        If provided, then SNV IDs should match this regex-pattern. The default is None.
    filter_chr : str, optional
        If provided, then chr column should math this regex-pattern. Possible applications
        omitting scaffold data, i.e. those chr whose pattern doesnt align with the 'chr\d+'
        pattern.
    bad_maps : IntervalDict or dict, optional
        Mapping chr->pos->BAD. The default is None.
    default_bad : float, optional
        The default BAD value which is used if no BAD maps are provided or if the particular SNV is not
        found in those BAD maps. If None, then those SNVs are filtered. The default is 1.0.
    count_snvs : bool, optional
        If True, samples and SNVs are counted and the counts are returned. The default is False.

    Returns
    -------
    res : dict
        Mapping (scorefiles, snvs, counts) -> data.
    sample_counter : dict
        Mapping BAD -> sample counts.
    snps_set : dict
        Mapping BAD -> set of unique SNVs. Not None only if count_snvs is True.

    """

    if type(bad_maps) is str and bad_maps:
        bad_maps = read_bad_maps(bad_maps)
    open = openers[compression]
    snvs = parse_filenames(snvs)
    data_filename = f'{name}.init.{compression}'
    folder = os.path.split(name)[0]
    for file in os.listdir(folder if folder else None):
        if file.startswith(f'{name}.') and file.endswith(tuple(openers.keys())):
            os.remove(os.path.join(folder, file))
    counts = None
    if count_snvs:
        snps_set = defaultdict(set)
    else:
        snps_set = None
    sample_counter = defaultdict(int)
    res = dict()
    scorefiles = list()
    snps_pos = defaultdict(list)
    file_count = 0
    its = track(snvs, description='Processing files...') if progress_bar else snvs
    for filename in its:
        try:
            counts = file_to_table(filename, counts, bad_maps=bad_maps, min_qual=min_qual, min_cnt=min_cnt,
                                   cnt_max_sum=max_cover, filter_db=filter_db, drop_bads=drop_bads,
                                   filter_rs=filter_rs, filter_name=filter_name, filter_chr=filter_chr,
                                   default_bad=default_bad, snps_set=snps_set, sample_counter=sample_counter,
                                   snps_pos=snps_pos, filename_id=file_count)
        except ValueError:
            logging.warning(f'Incorrect file: {filename}.')
            continue
        scorefiles.append(filename)
        file_count += 1

    for bad, d in counts.items():
        if symmetrify:
            for ind, n in list(d.items()):
                d[ind[::-1]] += n
        lt = list()
        for ref, alt in sorted(d):
            lt.append((ref, alt, d[(ref, alt)]))
        lt = np.array(lt, dtype=int)
        if symmetrify:
            lt[:, 2] //= 2
        counts[bad] = lt
    res['counts'] = counts
    res['scorefiles'] = scorefiles
    res['snvs'] = snps_pos
    with open(data_filename, 'wb') as f:
        dill.dump(res, f)
    return res, sample_counter, snps_set
