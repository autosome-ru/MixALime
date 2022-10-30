# MixALime: Mixture models for Allelic Imbalance Estimation

**Python 3.10+ as of now is not supported due to the fact that datatable package does not support it either. This will change as soon as datatable is released to 1.1.0.**

**MixALime** is a tool for the identification of allele-specific events in high-throughput sequencing data. It works by modelling counts data as a mixture of two Negative Binomial or Beta Negative Binomial distributions (where the latter is more applicable in case of noisy data at a cost of loss of sensitivity).

The package is *almost* easy to use and we advise everyone to just jump straight to installing **MixALime** and invoking the help command in a command line:

```
> pip3 install mixalime
> mixalime --help
```

We believe that the help section of **MixALime** covers its functionality well enough. Furthermore, the package arrives with a small demo dataset included and an easy-to-follow instruction in the abovementioned help section. So do not waste your time looking for how-to-clues or tutorials here, just use `--help`.

*Actually this README.md will probably be more complete and detailed one day than it is now, the README-writing person was just too tired at the moment.*

For the sake of following the social norms that impose a requirement of README files to be useful, in the next section you'll find the excerpt from `--help` command:

# Demo
A typical **MixALime** session consists of sequential runs of `create`, `fit`, `test`, `combine` and, finally, `export all`, `plot` commands. For instance, we provide a demo dataset that consists of a bunch of BED-like files with allele counts at SNVs (just for the record, **MixALime** can work with most vcf and  BED-like file formats):
```
> mixalime export demo
```
A *scorefiles* folder should appear now in a working directory with a plenty of BED-like files.
First, we'd like to parse those files into a **MixALime**-friendly and efficient data structures for further usage, as well as perform some \
basic filtering if necessary:
```
> mixalime create myprojectname scorefiles
```
Then we fit model parameters to the data with Negative Binomial distribution:
```
> mixalime fit myprojectname NB
```
Next we obtain raw p-values:
```
> mixalime test myprojectname
```
Usually we'd want to combine p-values across samples and apply a FDR correction:
```
> mixalime combine myprojectname
```
Finally, we obtain fancy plots fir diagnostic purposes and easy-to-work with tabular data:
```
> mixalime export all myprojectname results_folder
> mixalime plot myprojectname results_folder
```
You'll find everything of interest in *results_folder*.


# Combination of p-values across groups

*Note: a popular synonym for "combination" in this context is _aggregation_.*

One important feature that is not covered by the glorified `--help` in a very obvious fashion is a combination of p-values across separate groups (e.g. one group can be a treatment and the other is a control). The `combine` command with default options combines all the p-values. This can be changed by supplying the `--group` option followed by either a list of filenames that make up that group or a file that contains a list (newline-separated) of those files (the most convenient approach, probably), e.g.:
```
> mixalime combine --subname treatment -g vcfs/file1.vcf.gz -g vfcfs/file2.vfc.gz -g vcfs/file3.vcf.gz myproject
> mixalime combine --subname control -g vcfs/file4.vcf.gz -g vfcfs/file5.vfc.gz -g vcfs/file6.vcf.gz myproject
```
or
```
> mixalime combine --subname treatment -g group_treatment.tsv combine myproject
> mixalime combine --subname control -g group_control.tsv combine myproject
```
The `--subname` option is necessary if you wish to avoid different `combine` invocations overriding each other.
