# Experiment 3 — Query-to-Reference Mapping

## What this experiment does

One of Multigrate's most powerful practical features is the ability to **map a new dataset onto an existing trained reference atlas** without retraining the entire model from scratch. This is implemented using the **scArches** (single-cell architectural surgery) approach.

The real-world use case is this: a large healthy reference atlas has been built from millions of cells across multiple studies. A new patient dataset arrives — perhaps from a disease cohort — and the researcher wants to know: where do these new cells sit in the reference? What cell types are they? How do they compare to healthy cells?

With Multigrate + scArches, this is possible in minutes rather than hours, because only a tiny number of new batch-specific weights need to be trained.

This experiment corresponds to **Figure 3** in the paper, where the authors map 50,000 COVID-19 cells onto a healthy blood cell reference atlas of 160,000 cells.

---

## How scArches works

```
Step 1: Train reference model on healthy cells (Sites 2, 3, 4)
        → Learns a stable joint latent space

Step 2: Load query data (Site 1 — new experiment)
        → Add new batch weight vectors for query batches only
        → Freeze all reference model weights

Step 3: Fine-tune only the new batch weights
        → Takes 2-5 minutes (vs 20-30 min for full retraining)
        → Reference atlas structure is preserved

Step 4: Extract query latent embeddings
        → Query cells are now in the same space as reference
        → Cell type labels can be transferred by KNN or classifier
```

The key insight is that **only the batch correction weights** need to change for a new dataset — the encoder and decoder weights that capture biological variation are already well-trained and do not need updating.

---

## Why this experiment is not fully reproduced here

The full experiment from the paper requires:

| Requirement | Paper | Colab free tier |
|-------------|-------|-----------------|
| Reference cells | ~160,000 | Max ~30,000 |
| RAM needed | ~64 GB | ~12 GB |
| Modalities | RNA + protein + ATAC | — |
| Training time | Several hours | — |

The memory requirements exceed what is available on a free Colab instance. The reference atlas in the paper was built on a high-performance computing cluster (Helmholtz Center Munich).

A scaled-down version (using 20,000 cells, 3 sites as reference, 1 site as query) was attempted but ran into compatibility issues between the current version of multigrate's `load_query_data` function and the version of scvi-tools installed in Colab's Python 3.12 environment.

---

## Methodology (documented for understanding)

Even though the code does not run end-to-end in Colab, the methodology is fully documented here for understanding.

### Reference building

```python
import multigrate as mtg

# Train reference model on sites 2, 3, 4
mtg.model.MultiVAE.setup_anndata(
    adata_reference,
    categorical_covariate_keys=['Site'],
    rna_indices_end=4000,
)

vae = mtg.model.MultiVAE(
    adata_reference,
    losses=['nb', 'mse'],
    loss_coefs={'kl': 1e-1, 'integ': 0},
)

vae.train(max_epochs=200, lr=1e-3, batch_size=256)
```

### Query mapping (scArches)

```python
# Load query data into pre-trained reference model
# Only new batch weights are added; all reference weights stay frozen
new_vae = mtg.model.MultiVAE.load_query_data(
    adata_query,
    reference_model=vae,
)

# Fine-tune only the new batch weights (2-5 minutes)
new_vae.train(weight_decay=0, max_epochs=100)

# Extract query latent embeddings
new_vae.get_model_output(adata_query)
```

### Cell type transfer

```python
from sklearn.ensemble import RandomForestClassifier

# Train classifier on reference latent space
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(
    adata_reference.obsm['X_multigrate'],
    adata_reference.obs['cell_type']
)

# Predict cell types for query
query_pred = clf.predict(adata_query.obsm['X_multigrate'])
accuracy = (query_pred == adata_query.obs['cell_type']).mean()
print(f"Cell type prediction accuracy: {accuracy:.2%}")
# Paper reports ~79% accuracy
```

---

## Paper results

From Figure 3 of the paper:

- Query COVID-19 cells correctly mapped into their corresponding cell type clusters in the healthy reference
- Random forest classifier achieved **79% cell type prediction accuracy** across all cell types
- Cell types with < 100 cells in the reference (ASDC, Treg) were harder to predict — expected behaviour
- The UMAP shows clear separation between healthy (reference) and diseased (query) cells within each cell type cluster, enabling disease state comparison

---

## Files

```
03_query_to_reference/
├── README.md    ← This file (methodology documentation)
└── figures/     ← Empty — figures require HPC to reproduce
```

---

## References

- Lotfollahi et al. (2022). Mapping single-cell data to reference atlases by transfer learning. *Nature Biotechnology*, 40, 121–130. https://doi.org/10.1038/s41587-021-01001-7
- Official scArches documentation: https://docs.scarches.org
- multigrate_reproducibility q2r notebooks: https://github.com/theislab/multigrate_reproducibility/tree/main/q2r
