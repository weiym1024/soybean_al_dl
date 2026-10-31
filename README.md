# Reproducible deep learning analysis of soybean root aluminum-stress regulatory regions

## Overview

This repository contains the code, processed sequence windows, model checkpoint,
data splits, motif-analysis inputs, and reference outputs supporting the
CNN–BiLSTM analysis of chromatin accessibility in soybean roots under
aluminum (Al) stress.

The repository is a curated computational release, rather than a copy of the
complete original research project.

The available checkpoint supports independent inference and downstream
interpretation. Full model retraining was not repeated during release
validation.

## Model architecture and configuration

The final model is a three-class CNN–BiLSTM classifier.

Classes:

- 0: random_intergenic
- 1: shared_common
- 2: Al_specific

Main configuration:

| Parameter | Value |
|---|---|
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

The original training configuration is provided in `run_config.json`.

## Repository structure

```text
data/
  al_shared_bg_peak_windows.tsv

input/
  sequences/
    al_shared_bg_peak_windows.fa
  motif/
    JASPAR2026_CORE_plants_non-redundant_pfms_meme.txt

scripts/
  Training, prediction, attribution, mutagenesis,
  motif interpretation, and figure-data preparation scripts.

results/
  al_shared_bg_multiclass_jaspar/
    seed_42/
      Original train/validation/test peak assignments.

  al_shared_bg_multiclass_jaspar_bestfix_k12_g15_w124_balanced/
    seed_42/
      Final model checkpoint, prediction outputs,
      effective data splits, and interpretation results.
Input data and data splitting
The processed dataset contains 62,301 sequence windows representing
10,745 distinct peaks.

The original Seed 42 split assignments contain:
Split	Assigned peaks	Peaks with valid windows
Training	7,432	6,522
Validation	929	810
Testing	929	818
Total	9,290	8,150

A total of 1,140 assigned Al-specific peaks did not have corresponding
windows in the released input table (910 training, 119 validation,
and 111 testing).

These peaks were excluded when the training script matched split
assignments to the available sequence windows.

The training script additionally performs class balancing at the
window level. Validation and test sets are not balanced by this function.

The original assignments and effective peak subsets are both preserved
for traceability.
Model inference and performance validation
The final Seed 42 checkpoint was independently loaded using the
released sequence inputs.

Predictions were checked across all 62,301 windows.

The validation confirmed identical predicted classes and agreement
of class scores within numerical tolerance.

Chromosome-holdout evaluation used 20 folds with distinct test chromosomes.
Training, validation, and test assignments were checked for chromosome
and peak-level overlap.

The previously saved prediction outputs were independently used to
recalculate classification performance.

Mean peak-level performance across 20 folds:
Metric	Mean	SD
Macro AUROC	0.839861	0.022240
Macro AUPR	0.727011	0.028811
Al-specific AUROC	0.709265	0.036950
Al-specific AUPR	0.549594	0.048228

Peak-level class scores were obtained using max pooling and should
not be interpreted as a jointly normalized probability distribution.

The full training of the 20 fold-specific models was not rerun.
Model interpretation
The released analysis includes:
- DeepLiftShap attribution;
- GmALS3-associated window interpretation;
- single-nucleotide saturation mutagenesis;
- attribution-derived positive seqlet extraction;
- de novo motif discovery;
- motif comparison against JASPAR 2026;
- FIMO motif scanning;
- motif–gene matrix construction.
For the GmALS3-associated window Al_peak_004707__w0003,
the multiclass saturation mutagenesis analysis evaluated
1,500 alternate-base substitutions across 500 positions.

The saturation mutagenesis results were independently reproduced.

Single-window DeepLiftShap attribution showed high positional
agreement with the previously saved results, but not exact
identity to the historic attribution arrays.

The genome-wide attribution files are supplied as saved
intermediate results and were not all independently recomputed.
Motif discovery
Positive and background seqlet extraction produced:
- 2,748 positive seqlets;
- 2,748 background seqlets;
- seqlet length: 31 bp.
The seqlet FASTA and tabular outputs were reproduced byte-for-byte.

The original motif discovery used MEME Suite 5.5.7.
Independent reproduction used MEME Suite 5.5.9.

STREME parameters:
streme \
  --p positive_seqlets.fa \
  --n background_seqlets.fa \
  --dna \
  --minw 6 \
  --maxw 20 \
  --nmotifs 10 \
  --oc streme_output


All ten motif consensuses and reported E-values agreed
between the original and independently reproduced analyses.

Only STREME-1 met the conventional E-value < 0.05 criterion.
The remaining motifs should be interpreted as exploratory candidates.

TOMTOM was run using:
- JASPAR 2026 CORE Plants non-redundant motifs;
- Pearson motif similarity;
- minimum overlap of 5 bp.
All 180 reported TOMTOM matches were reproduced, including
matching motif identifiers, scores, and statistical values.
FIMO validation
FIMO motif scanning was independently reproduced with
a p-value threshold of 1e-4.
Dataset	Matching hits
Al-specific windows	6,965
Figure 5C Al-specific windows	6,965
Figure 5C shared/common windows	6,413

All hit identifiers, positions, matched sequences, scores,
p-values, and q-values agreed with the original results.
Motif–gene analysis
Figure 5D integrates motif hits with peak-associated gene
annotations, RNA-seq results, and model attribution scores.

Independent regeneration reproduced:
- 2,200 complete motif–gene records;
- 52 selected visualization records;
- all 20 fields within numerical tolerance.
GmALS3 (Glyma.10G047100) was associated with STREME-1 and
STREME-2 records in the selected matrix.

These computational associations do not, by themselves,
demonstrate direct transcription-factor binding or causal regulation.
Software requirements
See:
- requirements.txt for core Python package versions;
- SOFTWARE_VERSIONS.tsv for the software versions observed
  in the validation environment.
MEME Suite is required separately for STREME, TOMTOM, and FIMO.

The precise GPU/CUDA environment may affect the reproducibility
of model training, even with the same random seed.
Data integrity
The repository includes a SHA256 file manifest.

Run:
python scripts/verify_release_sha256.py


to check the integrity of the released files.
Reproducibility scope
Independently validated:
1. Seed 42 model inference.
2. Chromosome-holdout split integrity and performance recalculation.
3. GmALS3 saturation mutagenesis.
4. Positive/background seqlet extraction.
5. STREME motif discovery summary.
6. TOMTOM motif matches.
7. Three FIMO datasets.
8. Figure 5D motif–gene tables.
Not independently rerun:
1. Full Seed 42 model training.
2. Full 20-fold model training.
3. Genome-wide DeepLiftShap attribution for all windows.
The trained model checkpoint and saved intermediate results
are provided to support verification and downstream reuse.
Citation and data availability
Publication details, permanent repository URL, and archival DOI
will be added after the public release is finalized.
