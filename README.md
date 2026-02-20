# Feature Engineering for Fraud Detection in Financial Transactions

[![Research](https://img.shields.io/badge/Research-Honours-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Feature%20Engineering-green.svg)]()
[![Domain](https://img.shields.io/badge/Domain-Financial%20Fraud-red.svg)]()

> **Honours Research Project** — University of Limpopo, 2025

**Title:** Feature Engineering for Fraud Detection in Financial Transactions  
**Author:** Musa Brown Mhlaba (Student ID: 202219155)  
**Supervisor:** Mr. T. Mulaudzi  
**Co-Supervisor:** Mr. M.S. Mokoele  
**Faculty:** Faculty of Science and Agriculture  
**School:** School of Mathematics and Computer Science  

---

## Abstract

This research analyses how the choice of feature engineering techniques influences the performance of machine learning models for financial fraud detection. It compares two feature engineering techniques—**Principal Component Analysis (PCA)** and **Mutual Information (MI)**—across two machine learning models: **Random Forest** (supervised) and **Isolation Forest** (unsupervised).

The findings reveal that the choice of feature engineering technique significantly influences model performance. The **Mutual Information method consistently yielded superior results** across both models, demonstrating that preserving the most relevant original features is more effective than creating new synthetic components. The **Random Forest classifier, when combined with Mutual Information feature selection, proved to be the most effective strategy** for building a practical fraud detection system, achieving high F1-scores, balanced precision and recall, and strong overall accuracy.

**Keywords:** Principal Component Analysis, Mutual Information, Isolation Forest, Random Forest, Feature Engineering, Financial Fraud Detection

---

## Table of Contents

- [Research Overview](#research-overview)
- [Problem Statement](#problem-statement)
- [Research Questions](#research-questions)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusions](#conclusions)
- [Recommendations](#recommendations)
- [References](#references)

---

## Research Overview

### Background

The proliferation of digital transactions has been accompanied by a significant rise in financial fraud, making it a critical area of concern. Machine learning and anomaly detection techniques are increasingly being deployed to automate and enhance the detection of fraudulent activities. However, many studies primarily focus on improving model architectures while overlooking the impact of feature selection techniques.

### Aim

To evaluate and compare the impact of **Principal Component Analysis (PCA)** and **Mutual Information (MI)** feature selection techniques on the performance of anomaly detection models for financial fraud detection.

### Objectives

1. Implement and evaluate PCA and Mutual Information as feature selection techniques on a financial transaction dataset
2. Compare the performance of Isolation Forest (unsupervised) and Random Forest (supervised) models before and after applying these feature engineering techniques
3. Determine which feature selection method yields optimal performance for fraud detection in terms of accuracy, precision, recall, and computational efficiency

---

## Problem Statement

Financial fraud detection remains a critical challenge in the digital economy, where fraudulent transactions cause significant financial losses to businesses and consumers. The increasing complexity of fraud schemes makes it difficult for traditional rule-based systems to detect fraudulent activities effectively.

Despite progress in fraud detection research, many studies primarily focus on improving model architectures while overlooking the impact of feature selection techniques. Feature selection can enhance model performance by reducing dimensionality, removing noise, and improving model interpretability.

**Principal Component Analysis (PCA)** and **Mutual Information (MI)** are two commonly used feature selection techniques in machine learning, yet no direct comparison exists between these methods specifically for anomaly detection models in financial fraud detection.

---

## Research Questions

1. **How do PCA and Mutual Information compare** in terms of their impact on the performance of Random Forest for fraud detection?

2. **How do the performances of Random Forest and Isolation Forest compare** when using no feature selection, PCA, and Mutual Information?

3. **How does the choice of feature selection technique affect the interpretability** of the machine learning models?

---

## Methodology

### Dataset

**Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Characteristics:**
- **Transactions:** 284,807 (European cardholders, September 2013)
- **Fraud Cases:** 492 (0.172%)
- **Features:** 30 (V1-V28: PCA-transformed, Time, Amount)
- **Class Imbalance:** Severe (1:578 ratio)

### Feature Engineering Techniques

#### 1. Principal Component Analysis (PCA)
- **Type:** Unsupervised dimensionality reduction
- **Approach:** Applied second-level PCA on already PCA-transformed features
- **Variance Retention:** 95% of total variance
- **Output:** New synthetic principal components

#### 2. Mutual Information (MI)
- **Type:** Supervised feature selection
- **Approach:** Selected top 15 features with highest MI scores
- **Output:** Original features ranked by relevance to target

### Models

#### Random Forest (Supervised)
- Ensemble of decision trees
- Uses bagging and random subspace method
- Built-in feature importance ranking

#### Isolation Forest (Unsupervised)
- Anomaly detection via isolation
- Random partitioning to isolate anomalies
- Anomaly score based on path length

### Experimental Scenarios

| Scenario | Feature Engineering | Model |
|----------|---------------------|-------|
| 1 | PCA | Random Forest |
| 2 | PCA | Isolation Forest |
| 3 | Mutual Information | Random Forest |
| 4 | Mutual Information | Isolation Forest |

### Evaluation Metrics

- **Accuracy:** Overall correctness
- **Precision:** Accuracy of positive predictions
- **Recall:** Coverage of actual positives
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under ROC curve

---

## Results

### Key Findings Summary

| Model | Feature Engineering | Key Performance |
|-------|---------------------|-----------------|
| **Random Forest** | **Mutual Information** | **Best Overall** |
| Random Forest | PCA | Good precision |
| Isolation Forest | Mutual Information | Good ROC-AUC |
| Isolation Forest | PCA | Lower performance |

### Detailed Results

#### Accuracy Comparison
- Random Forest models (both PCA and MI) achieved very high accuracy due to class imbalance
- Isolation Forest showed lower accuracy but this is expected for anomaly detection

#### F1-Score Comparison ⭐
- **Random Forest with Mutual Information achieved the highest F1-score**
- This metric is critical for imbalanced datasets as it penalizes models favoring the majority class

#### ROC-AUC Comparison
- Isolation Forest with Mutual Information showed strong ROC-AUC performance
- Indicates good ability to distinguish between classes

#### Precision
- **Random Forest with PCA demonstrated the highest precision**
- When this model predicted fraud, it was most likely correct

#### Recall
- **Random Forest with Mutual Information yielded the top recall score**
- Successfully identified the highest proportion of actual fraud cases

### Comparative Analysis

#### PCA vs Mutual Information

| Criterion | PCA | Mutual Information |
|-----------|-----|-------------------|
| **Performance** | Good | **Superior** |
| Interpretability | Low (synthetic features) | **High** (original features) |
| Relationship Type | Linear only | Linear and non-linear |
| Redundancy Handling | Implicit (orthogonal) | None inherent |

**Conclusion:** Mutual Information was the superior feature engineering technique for both models.

#### Random Forest vs Isolation Forest

| Aspect | Random Forest | Isolation Forest |
|--------|---------------|------------------|
| **Precision/Recall** | **Higher** | Lower |
| ROC-AUC | Good | **Higher** |
| Training | Supervised | Unsupervised |
| Use Case | **Production fraud detection** | Novelty detection |

**Conclusion:** Random Forest is recommended for practical fraud detection systems.

---

## Conclusions

### Research Summary

This study successfully:
1. ✅ Implemented and evaluated PCA and Mutual Information feature selection techniques
2. ✅ Compared Random Forest and Isolation Forest performance under different feature engineering scenarios
3. ✅ Identified the optimal combination: **Random Forest + Mutual Information**

### Key Contributions

1. **Methodological:** Developed a replicable framework for evaluating feature engineering methods within a dual-paradigm approach (supervised and unsupervised)

2. **Practical Guidelines:** Offered data-driven recommendations for selecting appropriate feature engineering techniques based on performance, computational efficiency, and interpretability

3. **Performance Analysis:** Provided comprehensive evaluations across multiple metrics to optimize the balance between fraud detection sensitivity and false positive minimization

4. **Interpretability Assessment:** Empirically assessed how the choice between PCA and MI influences model interpretability—a critical consideration in regulated financial environments

---

## Recommendations

### For Financial Institutions

1. **Mandate Mutual Information (MI)** as the primary feature selection method for financial fraud data pipelines
   - Empirically proven to deliver superior predictive performance (F1-score)
   - Higher interpretability for regulatory compliance

2. **Deploy Random Forest for Real-Time Detection**
   - Implement specifically trained on MI-selected features
   - Ensures the highest balance of precision and recall

3. **Utilize Isolation Forest for Complementary Scoring**
   - Leverage strong ROC-AUC performance as a secondary layer
   - Score transactions for novelty or extreme anomalies

### For Future Research

1. **Incorporation of Temporal Features:** Explore time-based features (transaction velocity, time since last transaction) using RNNs or LSTM models

2. **Addressing Concept Drift:** Investigate adaptive and online learning techniques for evolving fraud patterns

3. **Deep Learning Integration:** Test feature selection impact on neural network architectures

4. **Real-time Pipeline:** Develop production-ready streaming detection systems

---

## Repository Structure

```
feature-engineering-fraud-detection-research/
├── README.md                    # This research documentation
├── docs/
│   ├── research_paper.pdf       # Full research paper (MHLABA MB 202219155)
│   └── presentation.pdf         # Defense presentation
├── results/
│   ├── summary.md               # Extended results
│   └── figures/                 # Research figures
└── data/
    └── README.md                # Dataset information
```

### Implementation Code

The experimental code for this research is available at:
**[github.com/MusaBrown/Anomaly-Detection](https://github.com/MusaBrown/Anomaly-Detection)**

---

## References

Selected key references:

1. Hilal, W., Gadsden, S. A., & Yawney, J. (2022). Financial Fraud: A Review of Anomaly Detection Techniques and Recent Advances. *Expert Systems with Applications*, 193, 116429.

2. Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. *Philosophical Transactions of the Royal Society A*, 374(2065), 20150202.

3. Tourassi, G. D., Frederick, E. D., Markey, M. K., & Floyd, C. E. (2001). Application of the mutual information criterion for feature selection in computer-aided diagnosis. *Medical Physics*, 28(12), 2394–2402.

4. Thangavel, K., & Pethalakshmi, A. (2009). Dimensionality reduction based on rough set theory: A review. *Applied Soft Computing*, 9(1), 1–12.

5. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *IEEE International Conference on Data Mining (ICDM)*.

*See full reference list in the research paper (52 references total).*

---

## Author

**Musa Brown Mhlaba**  
Student ID: 202219155  
Honours in Computer Science  
University of Limpopo, 2025

 
🐙 [GitHub: @MusaBrown](https://github.com/MusaBrown)

---

## Acknowledgments

- **Supervisor:** Mr. T. Mulaudzi
- **Co-Supervisor:** Mr. M.S. Mokoele
- **Institution:** University of Limpopo, School of Mathematics and Computer Science
- **Dataset:** Machine Learning Group at ULB (Université Libre de Bruxelles) via Kaggle

---

*This research contributes to the field of financial fraud detection by providing empirical evidence for feature selection techniques in machine learning-based anomaly detection systems.*
