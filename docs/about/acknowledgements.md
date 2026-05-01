# Acknowledgements

pykurucz stands on the shoulders of extraordinary work by generations of atomic physicists, stellar spectroscopists, and software engineers. This page records the intellectual debts and community contributions that make the project possible.

## Authors

pykurucz is developed by:

- **Elliot M. Kim** — Cornell University
- **[Yuan-Sen Ting](https://ysting.space/)** — The Ohio State University

The `kurucz-a1` neural-network atmosphere emulator that powers the warm-start
stage was developed in a separate project led by Jiadong Li, Mingjie Jian,
Yuan-Sen Ting, and Gregory M. Green
([Li et al. 2025, arXiv:2507.06357](https://arxiv.org/abs/2507.06357)).

## Robert L. Kurucz (1944–2025)

This project is dedicated to the memory of **Robert L. Kurucz**, whose nearly six decades of work at the Center for Astrophysics | Harvard & Smithsonian laid the foundations for modern stellar spectroscopy. Bob's ATLAS and SYNTHE codes, his atomic and molecular line lists, and his freely shared data have been used by thousands of astronomers worldwide, accumulating tens of thousands of citations.

Bob received the AAS Van Biesbroeck Prize in 1992 for "long-term extraordinary or unselfish service to astronomy." He spent seven days a week in his office for decades, driven by a simple goal he never stopped pursuing: *"I wanted to determine stellar effective temperatures, gravities, and abundances to study solar and stellar evolution. I still want to."*

This Python reimplementation of his codes is our tribute to his extraordinary legacy of open science.

## Scientific Software Communities

pykurucz would not be possible without the modern scientific Python ecosystem:

- **NumPy & SciPy** — The bedrock of numerical computing in Python. The array-oriented programming model makes it feasible to express complex physics kernels at near-Fortran speed.
- **Numba** — JIT compilation that closes the performance gap between Python and Fortran for hot loops. Without Numba, a faithful Python reimplementation at competitive speed would be impractical.
- **PyTorch** — The differentiable computing framework behind the `kurucz-a1` emulator. Its automatic differentiation and GPU support open the door to future physics-informed training and gradient-based optimization.
- **Matplotlib** — Visualization tools that help us debug spectra, compare with Fortran references, and produce publication-quality figures.

## Data Providers

The line lists and molecular catalogs distributed with pykurucz were originally compiled by:

- **R. L. Kurucz** — GFALL atomic line list, predicted-line binaries, ASCII molecular catalogs, and bound-free tables
- **David W. Schwenke** — TiO line list (Schwenke 1998)
- **Robert J. Partridge & David W. Schwenke** — H₂O line list (Partridge & Schwenke 1997)

We are grateful that these authors made their data freely available to the community.

## Institutional Support

- **The Ohio State University**

## Individual Contributors

We thank the early users and beta testers who provided feedback, reported bugs, and suggested improvements during the development of pykurucz. Your contributions have made the code more robust and more usable.

## How to Contribute

pykurucz is an open-source project, and we welcome contributions:

- **Bug reports** — Open an issue on GitHub with a minimal reproducing example.
- **Documentation** — Improvements to this docs site, tutorials, or examples are always welcome.
- **Physics extensions** — NLTE support, 3D atmosphere ingestion, and additional molecular line lists are active areas of interest.

See the repository's `README.md` and issue tracker for contribution guidelines.

## Next Steps

- Read the [License](license.md) for terms of use.
- See [Citation](citation.md) for how to acknowledge pykurucz in your publications.
