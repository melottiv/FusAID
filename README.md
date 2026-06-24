# FusAID

**FusAID: Integrating Sequence and 3D Structural Information for Gene Fusion Oncogenicity Prediction**

FusAID is a machine learning framework for predicting the oncogenic potential of gene fusion events by integrating protein sequence and predicted 3D structural information.

---

## Overview

Gene fusions play a key role in cancer development, but their functional impact is often difficult to assess computationally. FusAID addresses this problem by combining:

- Sequence-based representations of fusion proteins
- Structure-based descriptors derived from predicted protein conformations
- A multimodal predictive framework for oncogenicity classification

The model outputs a probability score indicating whether a fusion event is likely oncogenic or non-oncogenic.

---

## Model Architecture

FusAID integrates two complementary branches:

- **Sequence encoder**: transformer-based protein sequence embedding
- **Structure encoder**: geometric and physicochemical descriptors derived from predicted 3D structures

The final prediction is obtained via weighted soft-voting strategy

---

## Repository Structure
