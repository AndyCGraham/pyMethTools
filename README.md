# pyMethTools

A package of tools utilising the beta binomial distribution, designed for analysis of targeted DNA methylation sequencing data (e.g. RRBS, Twist Biosciences) in Python. Although, this package can be applied to any methylation data which provides the number of total DNA molecules assayed at a specific position (coverage), and the number of times that position was found to be methylated.

Functions include:
- Finding regions of cpgs with similar methylation profiles, by testing if contigous genomic sites are likely to come from one (codistributed) or multiple beta binomial distributions. These regions may represent functional domains.
- Simulating new samples, using beta binomial models fit to existing samples. A user specified number of differentially methylated sites or regions can be introduced at desired effect sizes, to facilitate tool benchmarking.
- Differential methylation analysis of individual sites or codistributed regions, while accounting for any covariates, using beta binomial regression.
- Identification of differentially methylated regions.

Please see [vignette](https://github.com/AndyCGraham/pyMethTools/blob/main/vignettes/vignette.ipynb) for examples of how to perform these tasks.

### Installation
```bash
pip install git+https://github.com/AndyCGraham/pyMethTools/
```

In progress:
- Celltype deconvolution, using the average beta values of genomic regions hypomethylated in specific celltypes.
