# Reproducible deep learning analysis of soybean root aluminum-stress regulatory regions

## Overview

This repository contains computational code and supporting data for CNN–BiLSTM analysis of chromatin accessibility in soybean roots under aluminum (Al) stress.

The workflow includes sequence-based classification of regulatory regions, chromosome-holdout evaluation, model interpretation using DeepLiftShap and saturation mutagenesis, de novo motif discovery, transcription factor motif annotation, and integration of regulatory sequence features with gene expression data.

## Model architecture and configuration

The model is a three-class CNN–BiLSTM classifier:

- **0:** `random_intergenic`
- **1:** `shared_common`
- **2:** `Al_specific`

| Parameter | Value |
|---|---:|
| Input sequence length | 500 bp |
| Random seed | 42 |
| CNN output channels | 256 |
| CNN kernel size | 12 |
| Number of classes | 3 |
| Batch size | 128 |
| Maximum epochs | 100 |
| Minimum epochs | 12 |
| Early-stopping patience | 8 |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Focal-loss gamma | 1.5 |
| Class weights | 1.0, 1.2, 2.4 |

## Repository structure

```text
soybean_al_dl/
├── data/                  Processed peak/window metadata
├── input/                 Input sequences and motif resources
├── models/                Model-related files
├── results/               Data splits and supporting analysis outputs
├── scripts/               Training, prediction, interpretation and motif-analysis scripts
├── README.md
├── requirements.txt
└── SOFTWARE_VERSIONS.tsv
```

## Input data and data splitting

The processed dataset contains **62,301 sequence windows** representing **10,745 distinct peaks**.

The original Seed 42 split assignments and the peaks with available sequence windows are summarized below:

| Split | Assigned peaks | Peaks with valid windows |
|---|---:|---:|
| Training | 7,432 | 6,522 |
| Validation | 929 | 810 |
| Testing | 929 | 818 |
| **Total** | **9,290** | **8,150** |

A total of 1,140 assigned Al-specific peaks did not have corresponding windows in the processed window table (910 training, 119 validation, and 111 testing). These peaks were excluded when split assignments were matched to available sequence windows.

Class balancing was performed at the **window level for training only**. The validation and test sets were not balanced by this function.

## Model inference and performance evaluation

Model predictions were evaluated across 62,301 windows. Chromosome-holdout evaluation used 20 folds with distinct test chromosomes; training, validation and test assignments were checked for chromosome- and peak-level overlap.

Mean peak-level performance across 20 folds:

| Metric | Mean | SD |
|---|---:|---:|
| Macro AUROC | 0.839861 | 0.022240 |
| Macro AUPR | 0.727011 | 0.028811 |
| Al-specific AUROC | 0.709265 | 0.036950 |
| Al-specific AUPR | 0.549594 | 0.048228 |

Peak-level class scores were obtained using max pooling and should not be interpreted as a jointly normalized probability distribution.

## Model interpretation

The analysis includes DeepLiftShap attribution, saturation mutagenesis, attribution-derived seqlet extraction, and downstream motif identification.

For the GmALS3-associated window `Al_peak_004707__w0003`, multiclass saturation mutagenesis evaluated **1,500 alternate-base substitutions** across 500 positions.

## Motif discovery

Positive and background seqlet extraction produced **2,748 positive seqlets** and **2,748 background seqlets**, each 31 bp long.

Original motif discovery used MEME Suite 5.5.7; the released workflow was checked using MEME Suite 5.5.9.

STREME parameters:

```bash
streme \
  --p positive_seqlets.fa \
  --n background_seqlets.fa \
  --dna \
  --minw 6 \
  --maxw 20 \
  --nmotifs 10 \
  --oc streme_output
```

All ten motif consensuses and E-values agreed between the original and comparison analyses. **STREME-1** met an E-value threshold of 0.05; the other reported motifs are exploratory candidates.

TOMTOM was run against **JASPAR 2026 CORE Plants non-redundant** motifs using Pearson similarity and a minimum motif overlap of 5 bp. The analysis reported 180 motif matches.

## FIMO motif scanning

FIMO was run with a p-value threshold of `1e-4`.

| Dataset | Motif hits |
|---|---:|
| Al-specific windows | 6,965 |
| Figure 5C Al-specific windows | 6,965 |
| Figure 5C shared/common windows | 6,413 |

## Motif–gene analysis

Figure 5D integrates motif hits with peak-associated gene annotations, RNA-seq results and attribution scores. The matrix includes **2,200 motif–gene records** and **52 selected visualization records**.

GmALS3 (`Glyma.10G047100`) is represented by STREME-1 and STREME-2 entries in the selected matrix. Motif matches and computational gene associations do not by themselves establish direct binding or causal regulation.

## Software requirements

Core Python dependencies are listed in [`requirements.txt`](requirements.txt); installed software versions are recorded in [`SOFTWARE_VERSIONS.tsv`](SOFTWARE_VERSIONS.tsv).

MEME Suite is required separately for STREME, TOMTOM and FIMO.

## Citation

Publication details will be added when available.
