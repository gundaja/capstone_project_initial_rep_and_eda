# Diabetes Patient Readmission Prediction

**Author:** Jayson Gundayao  
**Contact:** jayson.gundayao@gmail.com  
**Institution:** UC Berkeley (Professional Certificate in Machine Learning and Artificial Intelligence)  
**Dataset:** [UCI Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

---

## Executive Summary

This capstone project leverages machine learning to predict 30-day all-cause readmission risk for hospitalized diabetes patients. Utilizing a highly complex, retrospective Electronic Health Record (EHR) dataset containing over 101,000 encounters across 130 U.S. hospitals, this end-to-end data science pipeline cleans noisy clinical data, engineered high-dimensional features, resolves severe class imbalance (~11% minority target), and evaluates robust ensemble classifiers. 

The final tuned pipeline serves as an explainable risk-stratification asset, bridging the gap between passive data summary and actionable clinical decision support.

---

## Rationale & Business Case
Hospital readmissions are highly disruptive for patients and represent an immense financial burden on healthcare systems. Under the Hospital Readmission Reduction Program (HRRP), institutions face severe penalties for excessive 30-day return rates. For chronic conditions like diabetes, proactive discharge management can drastically mitigate these risks. 

By identifying vulnerable patients *before* they leave the ward, clinical care teams can optimize transitional care, schedule early follow-ups, and allocate intensive outpatient resources where they will have the highest preventive impact.

---

## Research Question

Can patient demographics, encounter history, diagnoses, medications, and hospital utilization variables predict whether a diabetes patient will be re-admitted — particularly within 30 days of discharge?

---

## Data Source

This project uses the [Diabetes Hospital re-admission Dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008). The dataset is derived from the UCI Diabetes 130-US Hospitals dataset and covers ten years (1999–2008) of clinical care across 130 U.S. hospitals and integrated delivery networks.

The dataset contains over 100,000 de-identified inpatient encounter records for diabetic patients aged 18 and older, with more than 50 features representing patient demographics and hospital outcomes. Encounters were included only if they met the following criteria:

- The encounter was an inpatient hospital admission
- Diabetes was recorded as a diagnosis
- The length of stay was between 1 and 14 days
- Laboratory tests were performed during the encounter
- Medications were administered during the encounter

Key features include patient age, race, gender, admission type, discharge disposition, time in hospital, number of lab procedures, number of medications, HbA1c test result, primary and secondary diagnosis codes, and prior inpatient, outpatient, and emergency visit counts.

---

## Methodology

The project follows a supervised machine learning workflow:

1. **Data Cleaning & Preprocessing** — Handle missing values, recode categorical variables (including consolidating diagnosis codes into broader categories per published methodology), remove deceased patient encounters, and assess class balance.

2. **Exploratory Data Analysis** — Examine feature distributions, identify correlations, and surface candidate predictors of re-admission.

3. **Feature Engineering** — Derive features from ICD diagnosis codes, utilization counts, and medication-related variables.

4. **Model Development** — Build and compare classification models including logistic regression, decision trees, random forests, and gradient boosting (e.g., XGBoost).

5. **Evaluation** — Assess model performance using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices, with particular attention to recall for the minority (re-admitted) class.

6. **Interpretation** — Identify the strongest predictors of re-admission risk and assess practical implications for healthcare operations.

---

## Key Performance Findings & Results

The primary optimization metric selected was **ROC-AUC**, ensuring a threshold-agnostic evaluation capable of ranking high-risk instances correctly despite heavy class imbalance.

### Baseline vs. Tuned Performance

| Model Pipeline | Handling Imbalance | Validation ROC-AUC | Test ROC-AUC | Key Clinical Trade-off |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline Models** | Unweighted | ~0.64 – 0.65 | — | Extremely poor positive class recall (~0%). |
| **Random Forest** | Class Weighted | 0.654 | 0.651 | Highly stable, but limited macro ranking power. |
| **XGBoost (Tuned)** | Balanced Weighting | **0.672** | **0.671** | **Best-in-class ranking. Captures ~45-50% of true readmissions at an acceptable precision trade-off.** |
| **XGBoost + SMOTE** | Synthetic Sampling | 0.661 | 0.658 | Induced slight overfitting; lower generalization score on held-out test data. |

> **Contextualizing the Results:** The final Test ROC-AUC of **0.671** aligns perfectly with established academic benchmarks for this noisy, retrospective dataset (Strack et al., 2014, which marks top-tier performance between 0.64–0.68). 

---

## Explainable AI (XAI) & Clinical Implementation

### SHAP Interpretability Takeaways
Rather than operating as a black-box model, the final XGBoost architecture was unpacked using SHAP (SHapley Additive exPlanations) values to identify the primary clinical drivers of readmission risk:
* **Prior Utilization:** The single strongest predictor of a future return is historical behavior—specifically, the total number of inpatient admissions and emergency room visits in the preceding year.
* **Discharge Destination:** Discharges to high-acuity environments (Skilled Nursing Facilities or Home Health agencies) push individual risk scores up substantially compared to standard routine home discharges.
* **Severity Matrix:** Longer length of stay combined with a high volume of lab procedures strongly correlated with positive risk predictions.

---

## Project Limitations & Next Steps

* **EHR Data Limitations:** The data represents a retrospective historical slice (1999–2008) and lacks modern Social Determinants of Health (SDOH) variables like economic status, housing stability, or transportation access, which heavily influence actual readmission success.
* **Calibration Enhancements:** Future developments will explore Post-Hoc Probability Calibration (such as Platt Scaling or Isotonic Regression) to turn the model's ordinal ranking values into reliable, absolute risk percentages.
* **Ensemble Expansion:** Investigate stacked meta-learners (e.g., blending the diverse decision borders of Random Forest and XGBoost via a Logistic Regression meta-classifier) for incremental gains in ROC-AUC.

---

## Project Notebooks

- **Notebook 1:** [Data cleaning and exploratory analysis](https://github.com/gundaja/capstone_project_initial_rep_and_eda/blob/main/notebook1_eda.ipynb)
- **Notebook 2:** [Feature engineering and baseline models](https://github.com/gundaja/capstone_project_initial_rep_and_eda/blob/main/notebook2_features_models.ipynb)
- **Notebook 3:** [Model evaluation and conclusions](https://github.com/gundaja/capstone_project_initial_rep_and_eda/blob/main/notebook3_evaluation.ipynb)

---

## References

Strack, B., DeShazo, J., Gennings, C., Olmo, J., Ventura, S., Cios, K., & Clore, J. (2014). Impact of HbA1c measurement on hospital re-admission rates: Analysis of 70,000 clinical database patient records. *BioMed Research International*, 2014. https://doi.org/10.24432/C5230J
