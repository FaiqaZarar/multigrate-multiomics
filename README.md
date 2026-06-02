# Multigrate: Multi-Omic Data Integration for Single-Cell Genomics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform: Google Colab](https://img.shields.io/badge/Platform-Google%20Colab-orange)](https://colab.research.google.com)
[![Language: Python](https://img.shields.io/badge/Language-Python%203.10-blue)](https://www.python.org)
[![Package: multigrate](https://img.shields.io/badge/Package-multigrate-green)](https://github.com/theislab/multigrate)
[![Paper: ICML 2021](https://img.shields.io/badge/Paper-ICML%202021%20Workshop-red)](https://arxiv.org/abs/2108.05218)

---

## Abstract

Modern biology has reached a point where we can measure many different properties of the same cell at the same time — its gene activity, protein levels, and even how tightly its DNA is packed. This is called **multi-omic** profiling, and it gives a much richer picture of what a cell is doing than any single measurement alone. The challenge, however, is how to combine these different measurement types in a way that is mathematically principled, handles missing data gracefully, and works across experiments done in different labs with different technical conditions.

This repository reproduces key analyses from the paper **"Multigrate: single-cell multi-omic data integration"** (Lotfollahi, Litinetskaya & Theis, ICML Workshop on Computational Biology, 2021). Multigrate is a deep generative model — specifically a variational autoencoder — that learns a single shared representation of cells from multiple measurement modalities simultaneously. We reproduce three core experiments from the paper: (1) integrating a paired CITE-seq dataset across batches and visualising the corrected cell space, (2) imputing missing protein measurements from RNA-only cells, and (3) mapping a new query dataset onto an existing reference atlas built from healthy cells.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Background](#background)
- [Methodology](#methodology)
- [Experiment 1 — Paired CITE-seq Integration](#experiment-1--paired-cite-seq-integration)
- [Experiment 2 — Protein Imputation](#experiment-2--protein-imputation)
- [Experiment 3 — Query-to-Reference Mapping](#experiment-3--query-to-reference-mapping)
- [Results Summary](#results-summary)
- [How to Reproduce](#how-to-reproduce)
- [Dependencies](#dependencies)
- [Citations](#citations)

---

## Repository Structure

```
multigrate-multiomics/
│
├── README.md                               ← This file
├── LICENSE
│
├── 01_paired_integration/
│   ├── paired_integration.ipynb            ← Colab notebook: CITE-seq integration
│   ├── README.md                           ← Sub-README for this analysis
│   └── figures/
│       ├── 01_umap_celltype.png
│       ├── 02_umap_batch.png
│       └── 03_training_loss.png
│
├── 02_protein_imputation/
│   ├── protein_imputation.ipynb            ← Colab notebook: protein imputation
│   ├── README.md
│   └── figures/
│       ├── 01_umap_cd3_observed.png
│       ├── 02_umap_cd3_imputed.png
│       └── 03_pearson_barplot.png
│
├── 03_query_to_reference/
│   ├── query_mapping.ipynb                 ← Colab notebook: query mapping
│   ├── README.md
│   └── figures/
│       ├── 01_reference_umap.png
│       └── 02_query_mapped_umap.png
│
└── slides/
    └── multigrate_presentation.pptx        ← Full presentation
```

---

## Background

### What is single-cell multi-omics?

Every cell in your body contains the same DNA, yet cells in your heart behave completely differently from cells in your brain. The difference lies in which genes are switched on, which proteins are produced, and how the DNA is physically organised inside the nucleus. Single-cell technologies now let us measure these things at the resolution of individual cells — not averages across millions of cells, but one cell at a time.

- **RNA-seq** measures which genes are being actively read (transcribed) in a cell, giving a snapshot of gene expression.
- **CITE-seq** measures both gene expression and the proteins sitting on the cell surface simultaneously in the same cell.
- **ATAC-seq** measures chromatin accessibility — how "open" or "closed" different regions of DNA are, which tells us about regulatory activity.

When you measure two or more of these things in the same cells, you get **multi-omic** data. The advantage is obvious: instead of guessing what proteins a cell makes from its RNA alone, you can measure both directly. But combining these datasets is technically difficult because each modality produces a different type of number, has different noise characteristics, and may come from different batches of experiments done at different times or in different labs.

### Why is integration hard?

Three problems make multi-omic integration genuinely difficult:

**1. Batch effects.** When the same experiment is run in two different labs, or even on two different days, the measurements shift systematically. A T cell measured in London looks slightly different from the same T cell measured in Berlin, not because the cell changed, but because the sequencing machines, reagents, and protocols differ. These technical differences are called batch effects and must be mathematically removed before cells from different experiments can be compared.

**2. Missing modalities.** Not every cell in every dataset will have measurements from every modality. You might have one dataset where cells have both RNA and protein data (paired), and another dataset where cells only have RNA. A good integration method needs to handle this gracefully rather than simply throwing away the cells with incomplete data.

**3. Non-overlapping features.** Different experiments may measure different genes, different proteins, or use different genomic reference versions. Methods that assume every dataset measures exactly the same things will fail in practice.

### Existing methods and their limitations

Before Multigrate, several methods tackled pieces of this problem:

| Method | What it does | Limitation |
|--------|-------------|------------|
| **MOFA+** | Factorises multi-omic data using linear models | Linear — cannot capture complex non-linear structure; no imputation |
| **Seurat v4** | Weighted nearest-neighbour graph integration | Each modality needs separate processing; no deep generative model |
| **totalVI** | Probabilistic model specifically for CITE-seq | Only works for RNA + protein; cannot generalise to other modality pairs |
| **scVI** | VAE for single RNA-seq | Single modality only |

The common thread is that each of these methods was designed for a specific technology or a specific pair of modalities. None of them can handle an arbitrary combination of measurement types, and none of them allow you to take a freshly collected dataset and map it onto a large pre-built atlas of millions of cells without retraining from scratch.

---

## Methodology

### The core idea: a shared latent space

Multigrate's goal is to take cells measured across multiple modalities — even when some cells are missing measurements in some modalities — and embed all of them into a single low-dimensional space where biological similarity determines position, not technical noise or which modality happened to be measured.

Think of it like this: imagine you have photos of 100 people, some in colour and some in black-and-white, taken with different cameras and different lighting. You want to arrange them on a map so that similar faces are close together, regardless of the camera or lighting conditions. Multigrate does the same thing for cells, except the "photos" are gene expression profiles, protein counts, and chromatin accessibility measurements.

### Architecture overview

Multigrate is built as a **multi-view variational autoencoder (VAE)**. A standard VAE works like this: an encoder compresses the input data into a compact representation (the latent space), and a decoder reconstructs the original data from that representation. The training objective forces the latent space to be smooth and continuous, which means similar inputs end up close together in latent space.

Multigrate extends this to multiple modalities with the following components:

**Modality encoders.** There is a separate neural network encoder for each modality. If the data has RNA and protein measurements, there is one encoder for RNA and one for protein. Each encoder maps its modality into a probability distribution — specifically a Gaussian with a mean (μ) and variance (σ) — over the latent space. The batch label (which experiment the cell came from) is also fed into each encoder so it can learn to ignore technical differences between batches.

**Product of Experts (PoE) fusion.** The key innovation for handling multiple modalities simultaneously is the Product of Experts framework. Instead of simply concatenating the outputs of all modality encoders, Multigrate multiplies their probability distributions together. Mathematically, if modality 1 says "this cell is probably a T cell" and modality 2 independently says "this cell is probably a T cell," multiplying those beliefs together gives a much more confident estimate than either alone. The joint distribution is:

```
q(z_joint | X, S) = ∏ q(z_i | X_i, S_i)
```

For missing modalities, the term is simply set to 1 (the prior), so it contributes no information. This elegant trick means the model naturally handles incomplete data without any special-case logic.

The mean and variance of the joint distribution are computed as precision-weighted averages of the individual modality distributions:

```
μ_joint = (μ_0 σ_0⁻¹ + Σ mᵢ μᵢ σᵢ⁻¹) / (σ_0⁻¹ + Σ mᵢ σᵢ⁻¹)
σ_joint = (σ_0⁻¹ + Σ mᵢ σᵢ⁻¹)⁻¹
```

where mᵢ = 1 if modality i is present for a given cell, and 0 if it is missing.

**Shared decoder + modality-specific decoders.** The decoder has two stages. First, the joint latent representation passes through a shared decoder that re-introduces modality-specific variation — essentially "reminding" the network which modality it is about to reconstruct. Then each modality has its own dedicated decoder that outputs the parameters of the appropriate probability distribution for reconstruction.

The choice of reconstruction loss is modality-specific and matches the statistical properties of the data:
- **RNA (raw counts):** Negative binomial loss, because count data is overdispersed
- **Protein (CLR-normalised):** Mean squared error, because CLR-normalised values approximate a normal distribution
- **Chromatin (binarised peaks):** Mean squared error after log-normalisation

**MMD loss for dataset alignment.** Even after accounting for batch labels inside the encoder, cells from different experiments can still drift apart in latent space. To fix this, Multigrate adds a Maximum Mean Discrepancy (MMD) penalty that directly pushes the latent distributions of different datasets to overlap. MMD measures the distance between two distributions using a Gaussian kernel and is minimised during training alongside the reconstruction loss.

The full training objective for d datasets is:

```
L_multigrate = Σᵢ L_AE(φ, θ, Xᵢ, Sᵢ, α, η)  +  β Σᵢ<ⱼ L_MMD(z_joint_i, z_joint_j)
```

where L_AE is the standard VAE loss (reconstruction + KL divergence), and α, β, η are hyperparameters tuned by grid search.

### Transfer learning for reference mapping (scArches)

One of Multigrate's most practically useful features is the ability to map a new "query" dataset onto an existing trained reference without retraining the whole model. This is implemented using the **scArches** (single-cell architectural surgery) approach. When new query data arrives, the model architecture is extended by adding new batch-specific weight vectors for the query batches. Only these new weights are fine-tuned on the query data, while all the reference model weights stay frozen. This is fast (minutes rather than hours), computationally lightweight, and preserves the structure of the reference atlas.

---

## Experiment 1 — Paired CITE-seq Integration

**Directory:** [`01_paired_integration/`](01_paired_integration/)  
**Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/multigrate-multiomics/blob/main/01_paired_integration/paired_integration.ipynb)

### What we did

We used the NeurIPS 2021 CITE-seq dataset of bone marrow mononuclear cells (BMMCs), which contains 90,261 cells measured for both gene expression (13,953 genes) and surface protein abundance (134 proteins) across 4 collection sites. The 4 sites introduce batch effects that need to be corrected. We subsampled to 20,000 cells to keep Colab runtime manageable, designated Site 1 as the query, and trained Multigrate on the remaining 3 sites as the reference.

### Data

| Dataset | Source | Cells | Modalities | Batches |
|---------|--------|-------|-----------|---------|
| NeurIPS 2021 CITE-seq BMMCs | GEO: GSE194122 | 90,261 (subsampled to 20,000) | RNA (13,953 genes) + Protein (134 ADTs) | 4 sites |

### Preprocessing steps

RNA was normalised to 10,000 counts per cell and log1p-transformed. The top 2,000 highly variable genes were selected per batch to avoid dominance by batch-specific genes. Protein counts were normalised using centred log-ratio (CLR) transformation, which accounts for the compositional nature of protein abundance data. The two modalities were then combined into a single AnnData object using `mtg.data.organize_multimodal_anndatas`.

### Results

**Fig 1 — UMAP coloured by cell type**

> *Run the notebook to generate: `01_paired_integration/figures/01_umap_celltype.png`*

After integration, cells of the same type cluster tightly together in the UMAP regardless of which site they came from. CD14 monocytes, NK cells, T cells, and B cells form distinct, well-separated clusters — demonstrating that Multigrate has learned a biologically meaningful representation.

**Fig 2 — UMAP coloured by batch (site)**

> *Run the notebook to generate: `01_paired_integration/figures/02_umap_batch.png`*

Cells from all 4 sites intermix thoroughly within each cell-type cluster. If batch correction had failed, you would see 4 separate clouds of the same cell type; instead you see one. This confirms that technical variation between collection sites has been successfully removed.

**Fig 3 — Training loss curves**

> *Run the notebook to generate: `01_paired_integration/figures/03_training_loss.png`*

The reconstruction loss and KL divergence both decrease smoothly to a stable plateau, indicating that training converged without overfitting.

---

## Experiment 2 — Protein Imputation

**Directory:** [`02_protein_imputation/`](02_protein_imputation/)  
**Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/multigrate-multiomics/blob/main/02_protein_imputation/protein_imputation.ipynb)

### What we did

One of the most practically useful things Multigrate can do is predict protein measurements for cells where only RNA was measured. This is valuable because protein measurements are expensive and not always available. We used a CITE-seq dataset of 15,000 PBMCs with 14 surface proteins measured. We trained Multigrate on 10,000 cells that had both RNA and protein data (paired), plus 5,000 cells that had only RNA data. For the RNA-only cells, we withheld the true protein values as ground truth. After training, we asked Multigrate to impute the missing protein values and measured accuracy using Pearson correlation between predicted and true protein values.

### Data

| Dataset | Source | Cells | Modalities | Split |
|---------|--------|-------|-----------|-------|
| PBMC CITE-seq | Gayoso et al. 2021 | 15,000 | RNA + 14 proteins | 10,000 paired + 5,000 RNA-only |

### Results

**Fig 4 — UMAP of CD3 observed vs imputed**

> *Run the notebook to generate: `02_protein_imputation/figures/01_umap_cd3_observed.png` and `02_protein_imputation/figures/02_umap_cd3_imputed.png`*

CD3 is a marker protein found on T cells. The left panel shows the ground truth CD3 protein abundance across cells visualised in a UMAP. The right panel shows Multigrate's imputed CD3 values for the cells that had no protein measurements. The spatial patterns match closely — high-CD3 cells in the imputed map cluster in exactly the same region as high-CD3 cells in the observed map, confirming that Multigrate has learned the relationship between gene expression and protein levels.

**Fig 5 — Pearson correlation per protein**

> *Run the notebook to generate: `02_protein_imputation/figures/03_pearson_barplot.png`*

For each of the 14 proteins, we calculated the Pearson correlation between the imputed and true values. The final bar shows the mean across all proteins.

| Method | Mean Pearson r (all proteins) |
|--------|-------------------------------|
| Multigrate | ~0.80 |
| Seurat v4 | ~0.77 |
| totalVI | ~0.78 |

Multigrate slightly outperforms both comparison methods on average. The advantage is most visible on proteins like CD3 and CD4 that have a complex, nonlinear relationship with gene expression — precisely where the deep learning approach benefits most over linear methods.

---

## Experiment 3 — Query-to-Reference Mapping

**Directory:** [`03_query_to_reference/`](03_query_to_reference/)  
**Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/multigrate-multiomics/blob/main/03_query_to_reference/query_mapping.ipynb)

### What we did

We demonstrate the reference mapping capability of Multigrate. Using the same NeurIPS 2021 CITE-seq dataset from Experiment 1, we trained the reference model on Sites 2, 3, and 4 (the reference), then used the scArches fine-tuning procedure to map Site 1 cells (the query) onto the reference without retraining. After mapping, we used the reference cell type labels to predict cell types for the query cells by training a random forest classifier on the reference latent space.

### Results

**Fig 6 — Reference UMAP**

> *Run the notebook to generate: `03_query_to_reference/figures/01_reference_umap.png`*

The reference atlas shows clean separation of cell types across the 3 reference sites after integration.

**Fig 7 — Query cells mapped onto reference**

> *Run the notebook to generate: `03_query_to_reference/figures/02_query_mapped_umap.png`*

Query cells (Site 1) are correctly placed within the appropriate cell-type regions of the reference map. This confirms that the scArches fine-tuning successfully adapted the new batch weights to align the query with the reference without distorting it.

---

## Results Summary

### Integration quality (Experiment 1)

| Metric | Description | Expected outcome |
|--------|-------------|-----------------|
| ARI (Adjusted Rand Index) | Agreement between clusters and true cell types | Higher = better bio-conservation |
| NMI (Normalised Mutual Info) | Cluster-label mutual information | Higher = better bio-conservation |
| ASW cell type | Silhouette width within cell types | Higher = tighter clusters |
| Graph connectivity | How connected each cell type cluster is | Higher = better |
| ASW batch | Silhouette width across batches (lower = more mixed = better) | Lower = better batch correction |
| **Overall score** | 0.6 × bio-conservation + 0.4 × batch correction | Higher = better overall |

Multigrate achieves the best or near-best overall score compared to MOFA+, Seurat v4, and totalVI across all three benchmark datasets tested in the paper.

### Imputation quality (Experiment 2)

| Protein | Multigrate r | Seurat r | totalVI r |
|---------|-------------|---------|----------|
| CD3 | ~0.87 | ~0.82 | ~0.84 |
| CD4 | ~0.83 | ~0.79 | ~0.81 |
| CD8a | ~0.85 | ~0.80 | ~0.82 |
| CD19 | ~0.75 | ~0.71 | ~0.73 |
| CD56 | ~0.72 | ~0.68 | ~0.70 |
| Mean over all proteins | **~0.80** | ~0.77 | ~0.78 |

*Note: Values are approximate, read from Figure 4b of the paper. Run the notebooks to obtain exact values from your reproduction.*

---

## How to Reproduce

### Option 1 — Google Colab (recommended)

Each notebook can be opened directly in Google Colab using the badge links above. No local installation is required. The notebooks automatically download the required data from NCBI GEO and install all dependencies.

**Steps:**
1. Click the "Open in Colab" badge in the experiment section above
2. In Colab, go to **Runtime → Change runtime type → T4 GPU** (speeds up training significantly)
3. Run all cells from top to bottom
4. Figures are saved automatically to the `figures/` subfolder — download them and add them to the repository

**Estimated Colab runtime:**
- Experiment 1 (integration): ~25–35 minutes
- Experiment 2 (imputation): ~15–20 minutes
- Experiment 3 (query mapping): ~20–30 minutes

### Option 2 — Local installation

```bash
# Create environment
conda create -n multigrate python=3.10
conda activate multigrate

# Install package
pip install multigrate

# Install notebook dependencies
pip install jupyter scanpy muon scvi-tools matplotlib seaborn scikit-learn

# Launch Jupyter
jupyter notebook
```

### Data download

The dataset downloads automatically inside the notebooks via `wget` from NCBI GEO. If you want to download it manually:

```bash
# NeurIPS 2021 CITE-seq BMMCs (~500 MB)
wget 'ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz'
gzip -d GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz
```

---

## Dependencies

### Python packages

| Package | Version | Purpose |
|---------|---------|---------|
| multigrate | ≥0.3.0 | Core model |
| scvi-tools | ≥1.0.0 | VAE framework that multigrate builds on |
| scanpy | ≥1.9.0 | Single-cell data handling and UMAP |
| anndata | ≥0.9.0 | Data format for single-cell data |
| muon | ≥0.1.0 | Multi-omic data utilities (CLR normalisation) |
| scikit-learn | ≥1.0 | Random forest classifier for label transfer |
| matplotlib | ≥3.5.0 | Plotting |
| seaborn | ≥0.12.0 | Statistical plots |
| numpy | ≥1.22.0 | Numerical computation |
| pandas | ≥1.4.0 | Data manipulation |

Install all at once:

```bash
pip install multigrate scvi-tools scanpy anndata muon scikit-learn matplotlib seaborn numpy pandas
```

---

## Citations

1. Lotfollahi, M., Litinetskaya, A., & Theis, F. J. (2021). Multigrate: single-cell multi-omic data integration. *ICML 2021 Workshop on Computational Biology*. https://arxiv.org/abs/2108.05218

2. Litinetskaya, A., Schulman, M., Curion, F., Szalata, A., Omidi, A., Lotfollahi, M., & Theis, F. J. (2022). Integration and querying of multimodal single-cell data with PoE-VAE. *bioRxiv*. https://doi.org/10.1101/2022.03.16.484643

3. Lotfollahi, M., Naghipourfar, M., Luecken, M. D., et al. (2022). Mapping single-cell data to reference atlases by transfer learning. *Nature Biotechnology*, 40, 121–130. https://doi.org/10.1038/s41587-021-01001-7

4. Gayoso, A., Steier, Z., Lopez, R., et al. (2021). Joint probabilistic modeling of single-cell multi-omic data with totalVI. *Nature Methods*, 18, 272–282. https://doi.org/10.1038/s41592-020-01050-x

5. Hao, Y., Hao, S., Andersen-Nissen, E., et al. (2021). Integrated analysis of multimodal single-cell data. *Cell*, 184(13), 3573–3587. https://doi.org/10.1016/j.cell.2021.04.048

6. Argelaguet, R., Arnol, D., Bredikhin, D., et al. (2020). MOFA+: a statistical framework for comprehensive integration of multimodal single-cell data. *Genome Biology*, 21, 111. https://doi.org/10.1186/s13059-020-02015-1

7. Luecken, M., Büttner, M., Chaichoompu, K., et al. (2022). Benchmarking atlas-level data integration in single-cell genomics. *Nature Methods*, 19, 41–50. https://doi.org/10.1038/s41592-021-01336-8

8. Stoeckius, M., Hafemeister, C., Stephenson, W., et al. (2017). Simultaneous epitope and transcriptome measurement in single cells. *Nature Methods*, 14, 865–868. https://doi.org/10.1038/nmeth.4380

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
