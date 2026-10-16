# Project provenance

**Author:** Siyun Chen · UC Davis · EEC 174AY · January 2026 coursework; October 2026 public package.

## Original work

I completed the individual supervised-learning lab using the course's ventilator dataset and notebook scaffold. My implementation covered eight additional waveform features, model comparisons, repeated Random Forest runs, the supplied feature-rule baseline, and stratified cross-validation.

The course scaffold and dataset were supplied by the teaching team. The original README credits **Kartik Patwari, Jeff Lai, Yui Ishihara, and Chen-Nee Chuah**. The rule definitions and task design are credited to the course.

Source record:

- Repository: `EEC174ABY/lab-2-supervised-ml-ChenSiyun1234` (private GitHub Classroom repository).
- File: `Lab2.ipynb`.
- Commit: `19198b215b024105d0c0024723d9a024bcdf506b` · January 29, 2026.
- Notebook Git blob: `c21bd85b5cb4257b629284aec24f44f722e2d7f8`.
- Source and retained outputs checked against GitHub on October 16, 2026.

## Public package

The code in `src/`, tests, documentation, synthetic generator, and plots are a fresh standalone implementation. The package adds explicit input contracts, keyed label joins, patient-separated splitting, training-only grouped cross-validation, and reproducible run records.

Only aggregate historical metrics are carried over from the saved notebook. No patient recordings, identifiers, original assignment prompts, instructor notebook, or third-party illustrations are redistributed. Public waveform images are generated from the included mathematical demo.

The MIT license covers the public code and original documentation here. It grants no rights to the private course dataset or teaching materials. Using other data requires the appropriate permission and a compatible adapter.

## Project scope

This project studies medical-signal classification and evaluation. The original medical models are Random Forest, Logistic Regression, KNN, and feature rules. My ResNet18 transfer-learning experiments used a separate image-classification dataset. The model names and metrics in this repository refer to the ventilator project.
