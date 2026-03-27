# Sparse Autoencoders — Monosemantic Feature Discovery

**University of Hertfordshire | Machine Learning and Neural Networks | 2025**

> Learn interpretable dictionaries from neural networks. From polysemantic neurons to monosemantic features.

---

## Overview

This tutorial covers Sparse Autoencoders (SAEs): why neural networks encode multiple concepts per neuron (superposition), how SAEs recover monosemantic features via overcomplete sparse dictionaries, and why SAEs are now the primary tool for LLM mechanistic interpretability.

Key topics: the superposition hypothesis, L1 vs L2 sparsity (subgradient argument), TopK variant, feature recovery metrics, and real results from Bricken et al. (2023) applying SAEs to Claude.

---

## Repository Contents

| File | Description |
|------|-------------|
| `sparse_autoencoder_tutorial.docx` | Full tutorial document |
| `sparse_autoencoder_tutorial.ipynb` | Jupyter notebook with full PyTorch implementation |
| `README.md` | This file |
| `LICENSE` | MIT licence |

---

## How to Run

```bash
pip install torch matplotlib numpy scipy
jupyter notebook sparse_autoencoder_tutorial.ipynb
```

No dataset download required — synthetic sparse-dictionary data is generated in the notebook.

---

## Figures

| Figure | Content |
|--------|---------|
| Figure 1 | SAE architecture diagram + activation sparsity distribution |
| Figure 2 | L1 vs L2 penalty shapes + sparsity-reconstruction trade-off |
| Figure 3 | SAE training curves + neuron activation frequency analysis |
| Figure 4 | Polysemantic neurons vs monosemantic SAE features (heatmaps) |
| Figure 5 | Reconstruction quality (PCA) + feature ablation study |
| Figure 6 | SAE features on LLM tokens + autoencoder comparison table |

---

## References

1. Olshausen & Field (1997) 'Sparse coding with an overcomplete basis set'. https://doi.org/10.1016/S0042-6989(97)00169-7
2. Bricken et al. (2023) 'Towards Monosemanticity'. https://transformer-circuits.pub/2023/monosemantic-features
3. Cunningham et al. (2023) 'Sparse Autoencoders Find Highly Interpretable Features in Language Models'. https://arxiv.org/abs/2309.08600
4. Elhage et al. (2022) 'Toy Models of Superposition'. https://transformer-circuits.pub/2022/toy_model
5. Makhzani & Frey (2013) 'k-Sparse Autoencoders'. https://arxiv.org/abs/1312.5663
6. Lee et al. (2006) 'Efficient sparse coding algorithms'. https://arxiv.org/abs/cs/0608094

---

## Licence

MIT — free to use, adapt, and share with attribution.
