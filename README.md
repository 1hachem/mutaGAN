# MutaGAN

This is an unofficial implementation of MutaGAN using pytorch.

## abstract
**MutaGAN: A Seq2seq GAN Framework to Predict Mutations of Evolving Protein Populations**

The ability to predict the evolution of a pathogen would significantly improve the ability to control, prevent, and treat disease. Despite significant progress in other problem spaces, deep learning has yet to contribute to the issue of predicting mutations of evolving populations. To address this gap, we developed a novel machine learning framework using generative adversarial networks (GANs) with recurrent neural networks (RNNs) to accurately predict genetic mutations and evolution of future biological populations. Using a generalized time-reversible phylogenetic model of protein evolution with bootstrapped maximum likelihood tree estimation, we trained a sequence-to-sequence generator within an adversarial framework, named MutaGAN, to generate complete protein sequences augmented with possible mutations of future virus populations. Influenza virus sequences were identified as an ideal test case for this deep learning framework because it is a significant human pathogen with new strains emerging annually and global surveillance efforts have generated a large amount of publicly available data from the National Center for Biotechnology Information's (NCBI) Influenza Virus Resource (IVR). MutaGAN generated "child" sequences from a given "parent" protein sequence with a median Levenshtein distance of 2.00 amino acids. Additionally, the generator was able to augment the majority of parent proteins with at least one mutation identified within the global influenza virus population. These results demonstrate the power of the MutaGAN framework to aid in pathogen forecasting with implications for broad utility in evolutionary prediction for any protein population.

paper : <https://arxiv.org/abs/2008.11790>

## general overview
this repo aims to use mutaGAN to generate new variants of SARS-CoV-2 spike protein.
sequences are collected from ncbi.

## step guide 

1. clone this repo

2. install dependencies from `environment.yml`

```
conda env create -f environment.yml
conda activate mutaGAN
```

3. install ncbi spike protein sequences (both proteomic and genomic sequences) using ncbi [datasets](https://www.ncbi.nlm.nih.gov/datasets/docs/) cli 

```
datasets download virus protein S
```
4. split genomic data using `data/split.sh` (since running nextclade on big .fna files is not supported)

5. install [nextclade](https://docs.nextstrain.org/projects/nextclade/en/stable/user/nextclade-cli.html) tool

6. run `data/nextclade_split.sh` 

> edit `configuration/files.json` to point to the right paths

> edit hyperparameters at `configuration/costume_hyper_params.json`  

7. train model 
```
python main.py
```

## todo

- [ ] utils.py/ write_fasta
- [ ] generate.py/ save generated sequences
- [ ] utils.py/ kmering
- [ ] utils.py/ levenshtein distance
- [ ] model.py/ stop generating sequences when `eos` token is reached