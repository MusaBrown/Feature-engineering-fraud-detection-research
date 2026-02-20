# Extended Results and Analysis

## Detailed Performance Metrics

### Experiment 1: Random Forest with PCA
- **Accuracy:** High (benefited from class imbalance)
- **Precision:** Highest among all configurations
- **Recall:** Moderate
- **F1-Score:** Good
- **ROC-AUC:** Good

**Analysis:** PCA reduced dimensionality but created synthetic features that lacked direct business interpretability. The model performed well but the feature space became abstract.

### Experiment 2: Isolation Forest with PCA
- **Accuracy:** Lower than Random Forest
- **Precision:** Lower
- **Recall:** Lower
- **F1-Score:** Lower
- **ROC-AUC:** Good

**Analysis:** Isolation Forest's unsupervised approach combined with PCA's synthetic features resulted in lower precision and recall for fraud detection.

### Experiment 3: Random Forest with Mutual Information ⭐
- **Accuracy:** High
- **Precision:** High
- **Recall:** **Highest**
- **F1-Score:** **Highest**
- **ROC-AUC:** High

**Analysis:** The winning combination. MI selected the most relevant original features, preserving interpretability while maximizing predictive performance. The Random Forest could leverage these meaningful features effectively.

### Experiment 4: Isolation Forest with Mutual Information
- **Accuracy:** Moderate
- **Precision:** Moderate
- **Recall:** Moderate
- **F1-Score:** Moderate
- **ROC-AUC:** **Highest**

**Analysis:** Strong ROC-AUC performance indicates good class separation capability. However, lower precision and recall compared to Random Forest make it less suitable for primary fraud detection.

## Statistical Significance

The results demonstrate that:
1. **Feature selection technique matters:** MI consistently outperformed PCA
2. **Model choice matters:** Random Forest outperformed Isolation Forest for this task
3. **Interaction effect:** The combination of RF + MI yielded synergistic improvements

## Feature Selection Impact

### PCA Characteristics in This Study
- Reduced 28 V-features to principal components explaining 95% variance
- Created orthogonal, uncorrelated features
- Lost direct feature interpretability
- Assumed linear relationships

### Mutual Information Characteristics in This Study
- Selected top 15 features from original 28 V-features
- Preserved original feature meanings
- Captured non-linear dependencies
- Required discretization for continuous variables

## Computational Efficiency

| Configuration | Training Time | Inference Time | Memory Usage |
|---------------|---------------|----------------|--------------|
| RF + PCA | Moderate | Fast | Low |
| RF + MI | **Fast** | **Fast** | **Low** |
| IF + PCA | Moderate | Moderate | Low |
| IF + MI | Fast | Fast | Low |

**Winner:** Random Forest with Mutual Information offered the best balance of performance and efficiency.

## Business Impact Analysis

### Cost of False Negatives (Missed Fraud)
- Direct financial losses
- Customer trust erosion
- Regulatory penalties

### Cost of False Positives (False Alarms)
- Customer inconvenience
- Investigation costs
- Operational overhead

### Recommendation
Random Forest + MI provides the optimal balance, minimizing both types of errors while maintaining interpretability for regulatory compliance.

## Limitations

1. **Dataset:** Single dataset (Kaggle Credit Card Fraud)
2. **Temporal:** No time-based features included
3. **Static:** No concept drift adaptation
4. **Scope:** Limited to two feature selection methods

## Future Improvements

1. **Multi-dataset validation** across different financial institutions
2. **Temporal feature engineering** for sequential patterns
3. **Online learning** for concept drift adaptation
4. **Deep learning comparison** with neural networks
5. **Ensemble methods** combining multiple feature selection approaches
