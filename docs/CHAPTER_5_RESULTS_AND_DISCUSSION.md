# Chapter 5: Results and Discussion

## 5.1 Introduction

This chapter presents the implementation and evaluation results of the plant-monitoring application and its machine-learning (ML) pipeline. The system combines sensor-data validation, temporal feature engineering, stress-level classification, anomaly detection, model explanations, recommendations, a backend API, and a web dashboard.

The results must be interpreted in light of data provenance: the supplied dataset is treated by the project as **synthetic**, and its stress labels are simulation labels rather than biological ground truth. Accordingly, the reported model scores measure agreement with those labels on the specified data split; they do not establish accuracy on real Tulsi plants or validate plant-health diagnoses.

## 5.2 Dataset and evaluation setup

The supplied CSV contains **5,376 readings** from **16 plants**, with **336 readings per plant**, collected at a two-hour cadence over four weekly sessions. It includes soil moisture, temperature, humidity, light, stress-level labels, and stress-type labels. The project’s dataset inspection reports no exact duplicate rows or duplicate plant–timestamp pairs. Sensor values include missing entries, while the validation run reported no range violations. The dataset also contains 65 positive values in its synthetic-anomaly helper label.

The classifier’s primary evaluation uses a chronological hold-out: the first three sessions (4,032 rows) are used for training and the fourth session (1,344 rows) is used for testing. Features are designed to use only current and past information. Plant identifiers, timestamps, experimental labels, and the synthetic-anomaly helper label are excluded from classifier inputs. The reported data and model provenance are synthetic; no real-plant validation set was available.

## 5.3 Data validation and implementation results

The build report records successful execution of the data-validation, training, evaluation, and inference scripts. It also reports successful API checks for health, metadata, prediction, and invalid-input handling. The supplied data had 1.3–1.6% missing values per sensor, which the application preserves as missing rather than silently imputing in stored sensor records. This is important when interpreting model outputs: predictions may be accompanied by warnings when readings or the preceding history have missing values.

The application pipeline processes the bundled 5,376 readings and stores a prediction and anomaly result for each successfully processed reading. It also stores explanations for each plant’s latest reading and creates recommendations from model output and sensor evidence. According to the root project README, a first full run takes approximately **75 seconds on a laptop CPU**; a subsequent idempotent run with no missing predictions takes approximately **5 seconds**. These are reported development-machine observations, not a benchmark of Jetson Orin Nano hardware.

## 5.4 Stress-classification results

The shipped classifier is an XGBoost multiclass model using 53 sensor and engineered features. It applies balanced class weights and smooths class probabilities over a plant’s latest 12 readings. On the chronological test session, the build report records the following results:

| Measure | Shipped model result |
|---|---:|
| Accuracy | 0.803 |
| Macro-F1 | 0.743 |
| Weighted-F1 | 0.792 |
| Balanced accuracy | 0.748 |
| Test samples | 1,344 |

Per-class F1 scores were **0.822 for healthy**, **0.495 for low**, **0.762 for moderate**, and **0.895 for high**. The low-stress class is the weakest: the report notes that 93 of 189 low-stress test examples were predicted as healthy. Therefore, the overall accuracy alone does not describe performance uniformly across stress levels. The per-class scores indicate stronger agreement for the high and healthy classes than for low stress.

The same model without temporal smoothing reached accuracy 0.765, macro-F1 0.723, and balanced accuracy 0.732 on the reported hold-out. This comparison suggests that smoothing improved the reported aggregate metrics for this split. It does not show that smoothing will provide the same benefit on other plants, environments, or real-world data.

The shipped model also outperformed the reported baselines on this test: the majority-class baseline had accuracy 0.141 and macro-F1 0.062, while logistic regression using the same 53-feature set had accuracy 0.459 and macro-F1 0.464. This indicates that the evaluated XGBoost configuration captured patterns in the supplied dataset more effectively than those baselines. Because the labels are synthetic, this is evidence about the dataset and evaluation protocol—not proof of biological generalization.

### 5.4.1 Cross-validation and generalization considerations

Additional reported evaluations examined leave-one-session-out and unseen-plant splits. For the shipped configuration, the build report gives macro-F1 of **0.678 ± 0.155** across leave-one-session-out folds and **0.627 ± 0.069** across unseen-plant folds. The variability, particularly across sessions, shows that performance depends on which session or plants are held out. The report also identifies a regression in the fold testing on the experiment’s first session and notes that trajectories from plants seen during training contribute to the easier in-time evaluation.

These results make the chronological hold-out the clearest primary result for the chosen train/test arrangement, while the cross-validation results are useful robustness checks. The folds are limited in number and the project does not report confidence intervals; the mean and standard deviation should therefore be treated as descriptive rather than as a precise estimate of future performance.

### 5.4.2 Feature and model changes

The reported ablation study found that adding deviations from concurrent healthy-control plants produced the largest improvement among the tested changes. On the chronological hold-out, macro-F1 increased from 0.582 for the earlier 62-feature model to 0.696 after adding cross-plant deviations, then increased further with longer context, feature pruning, class weighting, and smoothing in the tested configurations. The final shipped configuration includes missing-reference augmentation; its ablation macro-F1 was 0.743 for the hold-out, 0.678 ± 0.155 for leave-one-session-out, and 0.627 ± 0.069 for unseen plants.

The report’s interpretation is that concurrent control readings help account for shared room conditions. This is specific to the experiment’s design: it assumes control plants are available and measured concurrently. Missing-reference augmentation was included to make inference more robust when those readings are absent, although the reported results still show that model behavior can change when reference context is unavailable.

## 5.5 Anomaly-detection results

The anomaly detector is an Isolation Forest with configured contamination of 0.02. On the test session, compared with the dataset’s 14 positive synthetic-helper labels, the build report records:

| Measure | Result |
|---|---:|
| Precision | 0.286 |
| Recall | 0.429 |
| F1 | 0.343 |
| PR-AUC | 0.334 |
| ROC-AUC | 0.951 |

The high ROC-AUC alongside the lower precision, recall, and F1 illustrates why a single metric can be misleading, especially with few positive examples. The positive helper labels are synthetic and sparse, so these numbers should be read as a comparison against those labels, not as validated detection of real sensor faults or plant stress. In the application, an anomaly is intended as a prompt to inspect a sensor or reading; it is not interpreted as a plant-health class.

## 5.6 Explainability and recommendations

The pipeline provides local SHAP feature contributions when the optional explanation dependency and generated report are available. Local explanations report a limited set of top feature contributions and do not provide a model base value in the service contract. These contributions describe how features influence a model output; they do not establish causal biological effects. Global SHAP reports are generated as an explicit pipeline step and are not automatically regenerated by the backend.

Recommendations combine model output, sensor values, and heuristic rules. They are advisory outputs and should be reviewed against direct observations and appropriate horticultural guidance. In particular, neither a high predicted stress probability nor an anomaly flag should be presented as a confirmed plant-health diagnosis.

## 5.7 Application and service results

The implemented system separates the React/Vite frontend, FastAPI application backend, and ML inference service. The backend seeds the bundled dataset, provides data and monitoring APIs, invokes the ML service in batches, stores prediction results, and supports recommendations. The frontend presents plant readings, predictions, anomalies, explanations, and operational information. The ML service is stateless and receives a reading together with relevant plant history (and, where available, reference-plant readings).

The build report records **160 passing ML-pipeline tests** and **319 passing backend tests with 2 skipped** at the reported build stage. The skipped backend tests require a live ML-service integration. These tests support the correctness of implemented software behavior under the test conditions; they are not evidence of field performance or hardware deployment.

The project documentation explicitly states that Jetson Orin Nano hardware performance, JetPack installation, aarch64 dependencies, and on-device resource use were not verified in the reported build. Thus, the application and pipeline were tested on a development environment, but edge-device latency, memory use, power consumption, and sustained reliability remain to be measured on the target hardware.

## 5.8 Discussion

The results show that the implemented pipeline can validate the supplied data, train and serve models, and integrate their outputs into a monitoring application. On the chronological synthetic-data hold-out, the shipped stress classifier achieved macro-F1 0.743 and accuracy 0.803, exceeding the reported majority-class and logistic-regression baselines. Temporal smoothing and cross-plant contextual features were beneficial in the reported experiments. At the same time, low-stress classification remained substantially weaker than the other classes, and results varied across session and unseen-plant evaluations.

The anomaly results require similarly careful interpretation. The ROC-AUC was high, but precision, recall, and F1 were much lower, and the comparison used a small number of synthetic helper labels. SHAP contributions and recommendations add transparency and operational context, but do not convert simulated labels or heuristic outputs into biological evidence.

Overall, the software implementation demonstrates an end-to-end monitoring workflow and repeatable model-evaluation process. The evaluation does **not** demonstrate that the system can reliably diagnose stress in real Tulsi plants. Such a claim would require independently collected and quality-controlled plant observations, labels established through an appropriate protocol, evaluation on plants and conditions not used in development, and validation of the complete system on its target edge hardware.

## 5.9 Limitations

The principal limitations identified from the repository’s data and build documentation are:

1. **Synthetic provenance:** the dataset and stress labels are synthetic; no real-plant validation was available.
2. **Limited scale:** the evaluation covers 16 plants and four weekly sessions, with one session used as the primary chronological test set.
3. **Class imbalance and weak low-stress performance:** the low class has the lowest reported F1 and is frequently confused with healthy.
4. **Context dependence:** cross-plant features rely on concurrent healthy-control readings and the experiment’s group design.
5. **Sparse anomaly reference labels:** only 14 positive helper labels occur in the reported test session, and they are not verified real faults.
6. **No target-device benchmark:** Jetson hardware performance and resource consumption were not measured in the reported build.
7. **No field evaluation:** recommendation usefulness, alert burden, and biological relevance have not been established in a real growing environment.

## 5.10 Chapter summary

The application branch contains a complete software workflow for dataset validation, model training and inference, anomaly detection, explainability, recommendations, backend persistence, and dashboard visualization. The latest model results reported in the build documentation show improvement over the stated baselines on a synthetic chronological hold-out, while also exposing weaker low-stress classification and limitations in cross-session and unseen-plant generalization. These findings support the system’s feasibility as an experimental monitoring prototype. Real-plant data collection and target-device testing are required before drawing conclusions about practical agricultural performance.

## Sources in this repository

- [`money_plant_ml_pipeline/BUILD_REPORT.md`](../money_plant_ml_pipeline/BUILD_REPORT.md) — reported model metrics, ablations, test results, and evaluation limitations.
- [`money_plant_ml_pipeline/README.md`](../money_plant_ml_pipeline/README.md) — pipeline design, data provenance, feature engineering, and model interpretation.
- [`docs/PHASE0_FINDINGS.md`](PHASE0_FINDINGS.md) — dataset inspection and project-specific data findings.
- [`README.md`](../README.md) — application architecture and end-to-end pipeline behavior.
