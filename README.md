# README
## Installation
1) ```git clone https://github.com/wish1/neg_bin_fit```
2) ```cd neg_bin_fit```
3) ```pip3 install .```
4) ```???```
5) ```profit!```

## Usage
 1) Collect coverage statistics ```negbin_fit collect -I <inp_1> <inp_2> ... -O <dir> ```
	 * Files to provide in ```-I``` option should have the same format as described in [<b>Formats</b>](#Formats) section
	 * ```-O``` directory to save BAD-wise statistics into (default: ./)
 2) Fit neg-binomial distribution ```negbin_fit -O <dir>```
    * Directory provided in ```-O``` option should be the same with ```-O``` from <b>step1</b>
    * To visualize result add ```--visualize``` option
    * To use deprecated 'point fit' add ```-p``` option
3) Calculate p-values ```calc_pval -I <inp_1> <inp_2> ... -O <dir> -w <dir>```
    * Provide the same directory to ```-w``` option as in ```-O``` from the <b>step1</b> and <b>step2</b> 
    * To visualize result add ```--visualize``` option
    * ```-O``` directory to save tables with calculated p-value into (names of the files will be the same; extension will be changed to .pvalue_table)
4) Aggregate p-value tables ```calc_pval aggregate -I <inp_1> <inp_2> ... -O <out_file>```
	* ```-I``` option requires obtained on <b>step3</b> tables. Format is described in [<b>Formats</b>](#Formats) section
	* a *FILE* to save aggregated table should be provided with ```-O``` option  

## Formats

1) Input file for <b>step1</b> and <b>step2</b> should have the following columns: 
	- <b>#CHROM</b>, <b>POS</b>, <b>ID</b>: genome position;
	- <b>REF</b>, <b>ALT</b>:  reference and alternative bases;
	- <b>REF_COUNTS</b>, <b>ALT_COUNTS</b>: reference and alternative read counts;
	- <b>BAD</b>: BAD estimates with [BABACHI](https://github.com/autosome-ru/BABACHI).

2) Input file for <b>step 4</b> should have the same format as described above with 4 additional columns:
	- <b>PVAL_REF</b>, <b>PVAL_ALT</b>: Calculated p-values;
	- <b>ES_REF</b>, <b>ES_ALT</b>: Calculated effect sizes.
