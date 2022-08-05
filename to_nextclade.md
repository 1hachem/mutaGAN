## nextclade
Nextclade is a tool that identifies differences between your sequences and a reference sequence used by Nextstrain,
uses these differences to assign your sequences to clades, and reports potential sequence quality issues in your data.


## to install nextclade 
-  you can use conda :
```
conda install -c bioconda nextclade
```
- or you can install Nextclade linux-64 CLI from : https://github.com/nextstrain/nextclade/releases/latest/download/nextclade-Linux-x86_64

then you can add the executable to one of the directories included in system `$PATH` 

## to use nextclade with Sars-CoV-2
1. Download SARS-CoV-2 dataset : 
```
nextclade dataset get --name 'sars-cov-2' --output-dir 'path/to/output_dir'
```
2. run next clade on your input **genomic sequences** :
```
nextclade run --input-dataset path/to/nexclade_reference/  -O path/to/outputdir/   path/to/sequences.fasta  
```
I like to add `--verbose` option to get an idea of what is happening

## bonus : how to install data using the ncbi datasets cli

example 
```
datasets download virus protein S 
```

