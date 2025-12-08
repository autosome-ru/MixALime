#!/usr/bin/env python3
# Jeff Vierstra 2018
# TODO:
# --add filters/etc. as option
import sys
import logging

from argparse import ArgumentParser

import numpy as np
import ssl
import pysam

class snv:
	"""chrom, start, end, id, ref, alt, gt, extra fields
		GT encoded as either 0/1 or with pipe 0|0
	"""
	def __init__(self, line):
		fields = line.strip().split()
		self.contig = fields[0]
		self.start = int(fields[1])
		self.end = int(fields[2])
		self.id = fields[3]
		self.ref = fields[4]
		self.alt = fields[5]
		self.gt = sum(map(int, fields[6].replace('|','/').split('/')))
		#self.line = line

	def __str__(self):

		if self.gt == 0:
			gt_str = '0/0'
		elif self.gt == 1:
			gt_str = '0/1'
		else:
			gt_str = '1/1'

		return f'{self.contig}\t{self.start}\t{self.end}\t{self.id}\t{self.ref}\t{self.alt}\t{gt_str}'

logging.basicConfig(stream = sys.stderr, level = 30)

def parse_options(args):

	parser = ArgumentParser(description = "Count tags by allele")

	parser.add_argument("--chrom", dest = "chrom", type = str,
						default = None, help = "Use a specific contig/chromosome")

	parser.add_argument("var_file", metavar = "var_file", type = str,
						help = "Path to variant file (must have corresponding index)")

	parser.add_argument("original_bam_file", metavar = "original_bam_file", type = str, 
						help = "Path to BAM-format tag sequence file")

	parser.add_argument("remapped_bam_file", metavar = "remapped_bam_file", type = str, 
						help = "Path to BAM-format tag sequence file")

	return parser.parse_args(args)

class GenotypeError(Exception):
	pass

class DiploidError(Exception):
	pass

class ReadBiasError(Exception):
	pass

class ReadAlignmentError(Exception):
	pass

class ReadGenotypeError(Exception):
	pass

def get_5p_offset(pileupread):
	"""
	Returns position of variant relative to 5' of read
	"""
	if pileupread.query_position is None: # pileup overlaps deletion 
		return None
	elif pileupread.alignment.is_reverse:
		return pileupread.alignment.query_length-pileupread.query_position
	else:
		return pileupread.query_position+1

def get_base_quality(pileupread):
	"""
	Returns base call quality at variant position
	"""
	return pileupread.alignment.query_qualities[pileupread.query_position]


def check_bias(pileupread, offset=3, baseq=20):

	if pileupread is None:
		return True

	# if get_5p_offset(pileupread)<=offset:
	#	raise ReadBiasError()

	# if get_base_quality(pileupread)<baseq:
	# 	raise ReadBiasError()

	return True

def get_base_call(pileupread):

	if pileupread is None:
		return None

	if pileupread.query_position is None:
		return None
	else:
		return pileupread.alignment.query_sequence[pileupread.query_position]

def check_alleles(pileupread, ref_allele, nonref_allele):

	if pileupread is None:
		return True

	# if pileupread.alignment.mapping_quality<30:
	# 	raise ReadAlignmentError()
	
	read_allele = get_base_call(pileupread)
	if read_allele != ref_allele and read_allele != nonref_allele:
		return ReadGenotypeError()

	# if read_allele == ref_allele:
	# 	num_permitted_mismatches = 1 
	# elif read_allele == nonref_allele:
	# 	num_permitted_mismatches = 2 
	# else:
	# 	return ReadGenotypeError()

	# mismatches = int(pileupread.alignment.get_tag("XM", with_value_type=False))
	# if mismatches > num_permitted_mismatches:
	# 	raise ReadAlignmentError()

	# if re.search("[^ACGT]", pileupread.alignment.query_sequence):
	# 	raise ReadAlignmentError()
	# 	# raise AlignmentError("Ambiguous base calls within read (not matching {A, C, G, T})")

	# if re.search("[HSPDI]", pileupread.alignment.cigarstring):
	# 	raise ReadAlignmentError()
	# 	# raise AlignmentError("Deletions/indels within read")

	return True

def get_reads(variant, sam_file):

	reads_1 = {}
	reads_2 = {}

	# Go into BAM file and get the reads
	for pileupcolumn  in sam_file.pileup(variant.contig, variant.start, variant.start+1, max_depth=10000000, truncate=True, stepper="nofilter"):

		for pileupread in pileupcolumn.pileups:

			if pileupread.is_del or pileupread.is_refskip:
				continue

			if pileupread.alignment.is_read1:
				reads_1[pileupread.alignment.query_name] = pileupread
			else:
				reads_2[pileupread.alignment.query_name] = pileupread

	# All reads that overlap SNP; unqiue set
	read_pairs = set(reads_1.keys()) | set(reads_2.keys())

	return reads_1, reads_2, read_pairs

def main(argv = sys.argv[1:]):

	args = parse_options(argv)

	vars_file = pysam.TabixFile(args.var_file)
	original_sam_file = pysam.AlignmentFile(args.original_bam_file, "rb" )
	remapped_sam_file = pysam.AlignmentFile(args.remapped_bam_file, "rb" )

	for line in vars_file.fetch(reference=args.chrom):
	        
		#print rec.contig, rec.start, rec.ref, rec.ref

		variant = snv(line)
		ref = variant.ref
		alt = variant.alt
		#print(variant.id)
		# must be heterozygous
		if variant.gt != 1:
			continue

		n_ref = n_alt = n_failed = n_failed_bias = n_failed_mapping = n_failed_genotyping = 0

		try:
			
			_, _, read_pairs = get_reads(variant, original_sam_file)
			n_original_reads = len(read_pairs)

			reads_1, reads_2, read_pairs = get_reads(variant, remapped_sam_file)
			n_remapped_reads = len(read_pairs)

			for read in read_pairs:

				try:

					read1 = reads_1.get(read, None)
					check_alleles(read1, ref, alt) # only returns true if read exists
					check_bias(read1) # only returns true if read exists
					read1_allele = get_base_call(read1) # returns None if read doesn't exist
					
					read2 = reads_2.get(read, None)
					check_alleles(read2, ref, alt) # only returns true if read exists
					check_bias(read2) # only returns true if read exists
					read2_allele = get_base_call(read2) # returns None if read doesn't exist

					read_allele = read1_allele or read2_allele

					# No ba errors
					if read_allele == ref:
						n_ref += 1
					elif read_allele == alt:
						n_alt += 1
					else:
						raise ReadGenotypeError()

				except ReadBiasError as e:
					n_failed_bias += 1
					n_failed += 1
					logging.debug("Failed bias: " + read)
					continue
				except ReadGenotypeError as e:
					n_failed_genotyping += 1
					n_failed += 1
					logging.debug("Failed genotyping: " + read)
					continue

		except KeyboardInterrupt:
			sys.exit(1)
		
		n_failed_mapping = n_original_reads - n_remapped_reads

		print("%s\t%d\t%d\t%d\t%d\t%d\t%d" % (str(variant), n_ref, n_alt, n_original_reads, n_failed_mapping, n_failed_genotyping, n_failed_bias))
				
	original_sam_file.close()
	remapped_sam_file.close()
	vars_file.close()

	return 0
    
if __name__ == "__main__":
    sys.exit(main())

