# Consumer-Centric ITS: Sensor-Driven Explainable Federated Learning for Transparent Sleep Behavior Prediction in Autonomous Driving

## Overview

This research presents a consumer-centric Explainable Federated Learning (XFL) framework for transparent sleep-behaviour prediction in smart and autonomous driving environments within Intelligent Transportation Systems (ITS). It combines decentralized model training with Explainable Artificial Intelligence (XAI) to address the privacy, interpretability, and scalability concerns associated with centralized driver-monitoring systems.

The study uses the publicly available **Sleep Health and Lifestyle Dataset** to classify three sleep-disorder categories: **None**, **Sleep Apnea**, and **Insomnia**. Its proposed architecture treats sensor-enabled vehicles as local learning clients and describes Federated Averaging (FedAvg) for global model aggregation, with SHAP and LIME for interpretation. In the reported testing results, **Logistic Regression** achieved the best overall performance: **92.04% accuracy**, **90.65% precision**, **90.74% recall**, **90.69% F1-score**, **95.45% specificity**, and a **7.96% reported miss rate**. These are reported experimental results, not a demonstration of performance in a deployed vehicle.

## Research Objectives

- Develop a transparent framework for classifying sleep-disorder patterns relevant to driver-state monitoring.
- Support privacy-aware collaboration among distributed, sensor-enabled vehicle clients without centralizing raw client data in the proposed federated architecture.
- Compare conventional, ensemble, boosting, and neural-network classifiers on behavioural, lifestyle, and physiological attributes.
- Describe local model training and FedAvg-based global aggregation for collaborative prediction.
- Interpret global and individual model predictions using SHAP and LIME.
- Examine the potential of interpretable sleep-behaviour predictions to support driver warnings and consumer-oriented road safety.

## Dataset

The study cites the **Sleep Health and Lifestyle Dataset**, available publicly on Kaggle:

**Dataset URL:** https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

The data contain demographic, lifestyle, sleep, and physiological measurements. The paper analyzes these recorded attributes as inputs to a proposed sensor-driven vehicle-monitoring framework; it does not establish that the original dataset was collected directly from vehicle-mounted sensors during real-world driving.

### Prediction target

**Sleep Disorder** is a three-class classification target:

- **None:** No reported sleep disorder.
- **Sleep Apnea:** Sleep-apnea category.
- **Insomnia:** Insomnia category.

### Dataset fields discussed in the paper

| Field | Description |
|---|---|
| Person ID | Individual record identifier; not a physiological predictor |
| Gender | Recorded gender category |
| Age | Age of participant |
| Occupation | Participant occupation |
| Sleep Duration | Reported sleep duration |
| Quality of Sleep | Sleep-quality measurement |
| Physical Activity Level | Recorded activity level |
| Stress Level | Reported stress measurement |
| BMI Category | Body-mass-index category |
| Blood Pressure | Blood-pressure reading, analyzed as high and low pressure components in the paper |
| Heart Rate | Recorded heart rate |
| Daily Steps | Daily step count |
| Sleep Disorder | Target category: None, Sleep Apnea, or Insomnia |

The manuscript discusses these attributes and derived blood-pressure components in its exploratory visualizations. The precise encoded feature matrix and complete preprocessing configuration are not provided as executable files in the supplied materials.

## Behavioural, Lifestyle, and Physiological Features

The framework considers multiple kinds of sleep-relevant attributes:

- **Demographic and contextual:** Age, gender, and occupation.
- **Sleep-related:** Sleep duration and sleep quality.
- **Lifestyle:** Physical activity level and daily steps.
- **Physiological:** Heart rate and blood-pressure measurements.
- **Health-related:** BMI category and stress level.

### Exploratory data analysis

The paper reports distribution plots, box plots, categorical breakdowns, correlation analysis, pair plots, and scatter plots. Its selected correlation findings include:

| Feature pair | Reported Pearson correlation |
|---|---:|
| Sleep Duration / Quality of Sleep | 0.88 |
| High Pressure / Low Pressure | 0.97 |
| Stress Level / Sleep Duration | −0.81 |
| Stress Level / Quality of Sleep | −0.90 |

These correlations describe associations within the analyzed dataset and should not be interpreted as causal relationships.

## Methodology

The proposed XFL workflow comprises the following stages:

1. **Data acquisition:** Obtain sleep, lifestyle, demographic, and physiological attributes from the cited dataset; the architecture also conceptualizes data collection through in-vehicle and wearable sensors.
2. **Preprocessing:** Perform data cleaning, feature engineering, label standardization, exploratory analysis, correlation/multicollinearity assessment, and feature scaling as described in the paper.
3. **Client-level partitioning:** Represent distributed vehicles as separate learning clients in the federated architecture.
4. **Training/testing split:** Use a reported **70% training / 30% testing** split at each local client.
5. **Local model training:** Train and compare multiple classification algorithms. The methodology describes GridSearchCV-based hyperparameter optimization with cross-validation, but the exact fold count and search spaces are not specified in the supplied paper.
6. **Local performance assessment:** Evaluate classifier predictions using confusion matrices and standard classification metrics.
7. **Global aggregation:** Describe weighted **FedAvg** aggregation of participating clients' model updates to obtain a global model.
8. **Global evaluation:** Assess the aggregated model on the reported testing setup.
9. **Explainability:** Apply LIME to individual predictions and SHAP to feature contributions.
10. **Proposed feedback and warnings:** The architecture describes performance-driven retraining and warning generation when abnormal sleep behaviour is detected; field deployment of these components is not demonstrated in the supplied manuscript.

### Federated learning architecture

In the proposed architecture, vehicles retain private raw observations for local training while contributing model updates to a coordinating server. FedAvg is described as a dataset-size-weighted aggregation process:

```text
Local vehicle clients
    -> local preprocessing and training
    -> model parameter updates
    -> central FedAvg aggregation
    -> shared global model
    -> SHAP / LIME interpretation
    -> prediction and proposed warning workflow
```

**Implementation note:** The manuscript describes the federated architecture and aggregation equation but does not document all details necessary to independently verify the number of participating clients, client partition strategy, federated rounds, communication costs, or privacy-attack resistance. The claim that raw data remain local is a property of the proposed workflow, not proof of formal differential privacy or cryptographically secure aggregation.

## Machine-Learning Models

The study compares a broad set of local classification methods, including:

- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM (LGBM)
- CatBoost
- HistGradientBoosting
- Extra Trees
- Bagging
- Gaussian Naive Bayes
- Quadratic Discriminant Analysis (QDA)
- Gaussian Process Classifier
- Multilayer Perceptron (MLP)

**Logistic Regression** is identified as the strongest model in the reported local testing comparison and is also reported as the global model in Table 8.

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Proportion of correctly classified samples |
| Precision | Reliability of positive-class predictions under the paper's metric averaging approach |
| Recall | Proportion of relevant instances successfully detected |
| F1-score | Harmonic mean of precision and recall |
| Specificity | Ability to identify negative instances under the reported multiclass evaluation |
| Miss rate | Reported error indicator, numerically equal to `1 − accuracy` in the presented results |
| Confusion matrix | Per-class correct predictions and misclassifications |

The paper uses a three-class classification setting. Its precise averaging conventions for all multiclass metrics are not fully detailed in the supplied text; reported metric values are reproduced without assuming a different convention.

## Results Reported in the Paper

### Best-performing local and global model

| Testing metric | Logistic Regression result |
|---|---:|
| Accuracy | **0.9204 (92.04%)** |
| Precision | **0.9065 (90.65%)** |
| Recall | **0.9074 (90.74%)** |
| Specificity | **0.9545 (95.45%)** |
| F1-score | **0.9069 (90.69%)** |
| Reported miss rate | **0.0796 (7.96%)** |

Table 7 identifies Logistic Regression as the strongest local-testing classifier. Table 8 reports the same metric values for the global Logistic Regression model after federated aggregation. The supplied report does not establish an independently measured accuracy gain from federation over the best local model.

### Comparison with selected local classifiers

| Model/group | Reported testing accuracy | Reported miss rate |
|---|---:|---:|
| Logistic Regression | 0.9204 | 0.0796 |
| SVC, MLP, and QDA | 0.9115 | 0.0885 |
| Decision Tree and Gaussian Process Classifier | approximately 0.9027 | 0.0973 |
| Random Forest, Gradient Boosting, XGBoost, LightGBM, Bagging, and CatBoost | approximately 0.8938–0.8940 | 0.1062 |
| Gaussian Naive Bayes | 0.8761 | 0.1239 |
| Extra Trees | 0.8584 | 0.1416 |

These figures summarize values stated in the manuscript's discussion of Table 7; they are not newly computed benchmark results.

### Training and testing observations

The manuscript reports training accuracy of **0.9349** for several ensemble and boosting models, but the highest training score does not translate into the highest testing score. Logistic Regression performs better on the reported testing split. The confusion-matrix analyses also highlight occasional misclassifications among **None**, **Insomnia**, and **Sleep Apnea**, which are important when interpreting per-class performance in a safety-oriented setting.

## Explainable AI: SHAP and LIME

### LIME: Sleep Apnea prediction

The manuscript's Figure 14 reports one individual explanation with predicted probabilities:

| Predicted class | Probability |
|---|---:|
| Sleep Apnea | **0.91** |
| Insomnia | 0.08 |
| None | 0.01 |

The discussion highlights stress level, gender, sleep duration, quality of sleep, daily steps, blood pressure, and age among the contributing features. Its reported local explanation values include **+1.52** for stress level, **+1.00** for gender, **−1.47** for sleep duration, **−1.15** for sleep quality, and **−1.17** for daily steps. These are values associated with the presented local explanation and must not be interpreted as universal causal effects or as changes in predicted probability measured in percentage points.

### LIME: Insomnia prediction

Figure 15 presents another local explanation:

| Predicted class | Probability |
|---|---:|
| Insomnia | **0.85** |
| Sleep Apnea | 0.08 |
| None | 0.07 |

The paper discusses sleep quality, physical activity, daily steps, sleep duration, stress level, BMI, heart rate, occupation, and blood pressure as features influencing this specific prediction. These contributions are specific to the explained observation.

### SHAP: Global interpretation

Figure 16 emphasizes the following influential attributes in the global model's insomnia-related interpretation:

- Physical Activity Level
- Quality of Sleep
- Sleep Duration
- Daily Steps
- BMI Category
- Stress Level
- Heart Rate
- Blood Pressure

Age, gender, and occupation are described as having comparatively smaller influences. SHAP describes how features contribute to model outputs for the analyzed data; it does not establish physiological causation.

## Software and Computational Environment

The methodological description explicitly refers to **GridSearchCV**, cross-validation, **FedAvg**, **SHAP**, and **LIME**. Python's scikit-learn ecosystem is consistent with the named classifiers and GridSearchCV procedure, but the supplied manuscript does **not** provide an independently verified complete software environment, dependency list, package versions, hardware specification, or executable pipeline.

Accordingly, this README does not prescribe unverified installation commands, fixed package versions, or a fictional repository layout.

## Reproducing the Study

The paper documents the broad analytical workflow. A fuller reproduction would require:

1. Downloading the [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset).
2. Confirming the exact dataset revision, record count, treatment of identifiers and missing values, class labels, and blood-pressure parsing method.
3. Recreating preprocessing, categorical encoding, scaling, exploratory analysis, and the stated 70:30 split without data leakage.
4. Obtaining the original training random seed, hyperparameter grids, and cross-validation fold configuration.
5. Recreating the reported local classifier comparisons and confusion matrices.
6. Specifying the number of federated clients, client data distributions, aggregation rounds, client-model representation, and FedAvg implementation.
7. Re-evaluating the resulting global model using the paper's metric definitions and original test data.
8. Reproducing SHAP and LIME explanations for the same model states, samples, class encodings, and preprocessing transforms.

**Reproducibility status:** The supplied article does not contain sufficient executable source material or detailed configuration to claim exact numerical reproduction from this README alone.

### Repository-link clarification

The manuscript's data-availability statement lists a GitHub URL containing **`Diabetic-Retinopathy-ResNet-Models`**, which does not match the title or subject of this sleep-behaviour study. That URL should be checked and replaced with the correct study repository, if one exists, before it is presented as the implementation for this research.

## Limitations and Future Work

The paper identifies or motivates several practical challenges, including transparency, sensor heterogeneity, privacy, scalability, computational overhead, and real-time responsiveness. The evidence supplied in the manuscript also places limits on interpretation:

- The cited Sleep Health and Lifestyle Dataset is not itself demonstrated to be a real-world, vehicle-acquired stream of driver monitoring observations.
- The stated 70:30 evaluation does not by itself establish robustness across new drivers, geographical regions, vehicle platforms, or road conditions.
- Complete federated simulation settings and communication/latency evaluations are not sufficiently documented for independent replication.
- A federated architecture that avoids routine sharing of raw data does not automatically guarantee resistance to model-update inference or other privacy attacks.
- The paper's reported global Logistic Regression metrics match the local best-model metrics; additional ablation and multi-client validation are needed to quantify the federated contribution.
- XAI explanations reflect model associations within the evaluated observations, not clinically validated causal explanations of sleep disorders.
- A functioning on-vehicle monitoring or warning system, including safety testing under actual driving conditions, is not established by the reported offline experiments.

Future work could include independent driver-collected sensor datasets, non-IID client evaluations, communication and inference-latency measurements, formal privacy protection, uncertainty calibration, external validation, and end-to-end safety evaluation in realistic driving scenarios.

## Data Availability

- **Dataset:** [Sleep Health and Lifestyle Dataset on Kaggle](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset).
- **Source code:** The manuscript's listed GitHub URL appears unrelated to this particular study; the correct sleep-behaviour implementation repository should be verified before publication.

## Reference

**Research article:** *Consumer-Centric ITS: Sensor-Driven Explainable Federated Learning for Transparent Sleep Behavior Prediction in Autonomous Driving.*

This README summarizes the research methodology and results presented in the supplied manuscript. It does not constitute an independently audited implementation or a claim of real-world clinical or vehicle-system validation.
