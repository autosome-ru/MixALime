#!/usr/bin/env python3

import sys
import logging
from os.path import basename, splitext, join
from argparse import ArgumentParser

import pysam
from scipy.stats import binom_test
import numpy as np

def parse_options(args):

    parser = ArgumentParser(description = "Combined allelic read depths per sample into a larger VCF file")

    parser.add_argument("variant_file", metavar = "variant_file", type = str,
                        help = "Path to VCF-format genotyping file.")

    parser.add_argument("sample_map_file", metavar = "sample_map_file", type = str,
                        help = "Sample to individual mapping file")

    parser.add_argument("outfile_per_sample", metavar = "outfile_per_sample", type = str,
                        help = "Output VCF file")

    parser.add_argument("--chrom", metavar = "chrom", type = str,
                    default=None, help = "Restrict to a specific chromosome")

    return parser.parse_args(args)

class GenotypeError(Exception):
    pass

# Make a new VCF file
def make_vcf_header_from_template(template_header, samples):
    header=pysam.VariantHeader()
    for record in template_header.records:
        header.add_record(record)

    # hack because the clearing the formats doesn't actually delete them "behind the scenes"
    header.formats.clear_header()
    header=header.copy()

    header.formats.add('GT', 1, "String", "Genotype")
    header.formats.add('AD', 2, "Integer", "Allele read depth")
    header.formats.add('RD', 1, "Integer", "Read depth (unfiltered)")
    header.formats.add('ARD', 1, "Integer", "Read depth (passing filters)")
    header.formats.add('FM', 1, "Integer", "Failed  - mapping")
    header.formats.add('FG', 1, "Integer", "Failed - discordant genotypes")
    header.formats.add('FB', 1, "Integer", "Failed - 5' proximity bias")
    header.formats.add('FMR', 1, "Float", "Failed mapping rate")

    for sample in samples:
        header.add_sample(sample)

    return header


def get_variant(tabix, var):
    try:
        for row in tabix.fetch(var.contig, var.start, var.start+1, parser=pysam.asTuple()):
            ref, alt = row[4], row[5]
            if ref==var.ref and alt==var.alts[0]:
                return row
    except Exception as e:
        pass # silences the "cannot create iterator error"
        
    raise StopIteration


def main(argv = sys.argv[1:]):

    args = parse_options(argv)

    samples=[]
    samples_genotype_id={}
    sample_infiles={}

    with open(args.sample_map_file) as f:
        for line in f:
            (sample, genotype_sample_id, het_counts_filepath)=line.strip().split("\t")

            samples.append(sample)
            samples_genotype_id[sample]=genotype_sample_id
            sample_infiles[sample]=pysam.TabixFile(het_counts_filepath)

    infile=pysam.VariantFile(args.variant_file, mode='r', ignore_truncation=True)

    # Output VCF -- allelic tags per sample
    outfile_per_sample=pysam.VariantFile(args.outfile_per_sample, mode='w', 
        header=make_vcf_header_from_template(infile.header, samples))
    outfile_per_sample.header.add_meta('ARD_command', 'recode_vcf.py ' + ' '.join(['%s=%s' %(k, v) for k, v in vars(args).items()]))

    counts=np.zeros(2, dtype=int)

    for var in infile.fetch(contig=args.chrom):
        
        samples_format=[]
        n_total_ref = n_total_alt = 0

        # Loop over individuals
        for sample in samples:
            try:
                genotype_sample_id=samples_genotype_id[sample]
                genotype = var.samples[genotype_sample_id].alleles

                gt = var.samples[genotype_sample_id]['GT']
                
                # haploid genotype call
                if len(genotype)<2:
                    raise StopIteration

                # homozygous
                if genotype[0]==genotype[1]:
                    raise StopIteration 
                
                # if GT tag = (1,0) then swap allele; this happens with "bcftools norm" when splitting multiallelicss
                if gt[0] == 1 and gt[1] == 0:
                    genotype = tuple(reversed(genotype))
                    gt = tuple(reversed(gt))

                # open samaple het counts file
                tabix = sample_infiles[sample]
                row = get_variant(tabix, var)
                
		# Sanity check -- make sure alleles match up in VCF and tabix files
                if row[4]!=genotype[0] or row[5]!=genotype[1]:
                    raise GenotypeError()

                n_ref = int(row[7])
                n_alt = int(row[8])
                n_total = int(row[9])
                n_failed_mapping = int(row[10])
                n_failed_genotyping = int(row[11])
                n_failed_bias = int(row[12])

                n_total_ref += n_ref
                n_total_alt += n_alt

            except ValueError as e:
                logging.critical(e)
                logging.critical(sample)
                sys.exit(1)
            except GenotypeError as e:
                logging.critical(f'Genotype from VCF file does not match in genotype in TABIX file! '
                                 f'{var.contig}:{var.start}:{var.ref}/{var.alt} -- {sample}')
                sys.exit(1)
            except StopIteration as e:
                n_ref=n_alt=n_total=n_failed_mapping=n_failed_genotyping=n_failed_bias=0
            finally:
            
                samples_format.append({
                    'GT': gt, 
                    'AD': [n_ref, n_alt], 
                    'ARD': n_ref + n_alt,
                    'RD': n_total,
                    'FM': n_failed_mapping, 
                    'FMR': 0 if n_total==0 else n_failed_mapping/n_total, 
                    'FG': n_failed_genotyping, 
                    'FB': n_failed_bias,
                    'phased': False})
                
        # End loop individuals
        if n_total_ref + n_total_alt > 0:
            outvar = outfile_per_sample.new_record(contig=var.contig, start=var.start, stop=var.stop, alleles=var.alleles, id=var.id, qual=var.qual, filter=var.filter, info=var.info, samples=samples_format)
            outfile_per_sample.write(outvar)


    for sample, filehandle in sample_infiles.items():
        filehandle.close()

    outfile_per_sample.close()
    infile.close()

    return 0
    
if __name__ == "__main__":
    sys.exit(main())

