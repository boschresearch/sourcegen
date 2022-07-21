# Overcoming Shortcut Learning in a Target Domain by Generalizing Basic Visual Factors from a Source Domain

Shortcut learning occurs when a deep neural network overly relies on spurious correlations in the training dataset in order to solve downstream tasks. Prior works have shown how this impairs the compositional generalization capability of deep learning models. To address this problem, we propose a novel approach to mitigate shortcut learning in uncontrolled target domains. Our approach extends the training set with an additional dataset (the source domain), which is specifically designed to facilitate learning independent representations of basic visual factors. We benchmark our idea on generated target domains where we explicitly control shortcut opportunities as well as real-world target domains. Furthermore, we analyze the effect of different specifications of the source domain and the network architecture on compositional generalization. Our main finding is that leveraging data from a source domain is an effective way to mitigate shortcut learning. By promoting independence across different factors of variation in the learned representations, networks can learn to consider only predictive factors and ignore potential shortcut factors during inference.

For more information about this work, please read our [ECCV 2022 paper](https://arxiv.org/abs/2207.10002):

> Saranrittichai, P., Mummadi, C., Blaiotta, C., Munoz, M., & Fischer, V. (2022). Overcoming Shortcut Learning in a Target Domain by Generalizing Basic Visual Factors from a Source Domain. In Proceedings of the European Conference on Computer Vision (ECCV).

## Table of Contents
- [Installation](#installation)
- [Run Studies](#run-studies)
- [Questions and Reference](#questions-and-reference)

## Installation

1. First we recommend to setup a python environment using the provided `environment.yml` and install the package:

```
conda env create -f environment.yml
source activate sourcegen
pip install -e .
```

2. Navigate to `data/diagvibsix` and follow the instruction on `diagvib_setup_instruction.txt` to prepare data for the DiagViB-6 framework. In this work, we customize DiagViB-6 for our use cases. Official DiagViB-6 release can be found [here](https://github.com/boschresearch/diagvib-6).

## Run Studies

We provide python scripts to run studies on the color animal dataset with FactorSRC variations. For fully-correlated setup, the study can be performed by running the script below:
```
python -m sourcegen.studies.run_study_fully_correlated
```

Similarly, for semi-correlated setup, the study can be performed by running the script below:

```
python -m sourcegen.studies.run_study_semi_correlated
```


## Questions and Reference
Please contact [Piyapat Saranrittichai](mailto:piyapat.saranrittichai@de.bosch.com?subject=[GitHub]%20SourceGen)
or [Volker Fischer](mailto:volker.fischer@de.bosch.com?subject=[GitHub]%20SourceGen) with
any questions about our work and reference it, if it benefits your research:
```
@InProceedings{Saranrittichai_2022_ECCV,
author = {Saranrittichai, Piyapat and Mummadi, Chaithanya Kumar and Blaiotta, Claudia and Munoz, Mauricio and Fischer, Volker},
title = {
Overcoming Shortcut Learning in a Target Domain by Generalizing Basic Visual Factors from a Source Domain},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
month = {October},
year = {2022}
