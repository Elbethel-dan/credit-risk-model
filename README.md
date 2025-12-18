# 📘 Credit Scoring Business Understanding

Credit risk refers to the likelihood that a borrower will fail to meet their financial obligations, resulting in a loss to the lender. It sits at the core of lending and financial stability, influencing decisions such as loan approval, interest rates, and required capital reserves. Key topics within credit risk include probability of default (PD), which measures how likely a borrower is to default; loss given default (LGD) and exposure at default (EAD), which quantify the severity of losses when defaults occur; and credit scoring models, which use borrower data to estimate risk. Modern credit risk management also emphasizes model interpretability, regulatory compliance (such as Basel II/III), data quality, and ongoing model validation and monitoring to ensure fair, reliable, and transparent lending decisions.

### 1) Why Basel II’s emphasis on risk measurement means we need an interpretable, well-documented model

Basel II (Pillars 1–3) requires banks to measure, validate and disclose their credit risk parameters (e.g., PD, LGD, EAD) and subjects internal models to supervisory review — which raises the bar on model governance, validation and transparency. Regulators expect banks to demonstrate how models map applicant features to risk and to show that models are robust, stable and properly validated.

Practical implication: a model that is easy to explain (feature definitions, transformations, why a variable matters) makes regulatory approval, internal validation, audit trails and ongoing monitoring much easier and less risky than an opaque black box. Strong documentation reduces model risk (governance, mis-use, calibration errors) and speeds supervisory review.

### 2) Why we must create a proxy “default” label, and the business risks that follow

When an explicit default flag (ground-truth label) is not available, a proxy outcome (for example: 90+ days past due, charge-off, or other behavioral signals) is necessary to produce supervised PD estimates. Credit scoring literature and operational guidance explicitly recommend constructing careful, business-aligned proxy definitions when true default data are missing or sparse.

Key business risks when using a proxy label:

- Label misspecification / bias: the proxy may systematically under- or over-represent true defaults (e.g., early collection practices, product changes), producing biased PDs and poor capital/pricing decisions.

- Temporal drift & policy dependency: proxies tied to short-term operational rules (e.g., cobranded collection policy) can change, invalidating the model unless re-labeled and re-validated.

- Compliance & fairness exposure: if the proxy correlates with protected attributes or certain data-sources (alternative data), the proxy can create disparate impacts or regulatory concerns. Regulators expect the rationale for proxy construction and fairness checks.

- Business consequence: decisions (accept/decline, pricing, capital buffers) based on a weak proxy can lead to unexpected losses, mispriced portfolios, reputational damage and supervisory pushback.

Mitigations (brief): pick proxies that are economically meaningful (e.g., 90+ DPD or charge-off), document assumptions, backtest with whatever ground truth exists, run sensitivity analyses, and treat proxy creation as a formal model component subject to validation and governance.

### 3) Trade-offs: simple interpretable model (Logistic + WoE) vs complex high-performance model (Gradient Boosting) in a regulated context

Interpretability & governance

- Logistic Regression + WoE: Highly interpretable (coefficients, monotonic WoE bins), easy to document, explain to regulators and embed into scorecards and decision rules; simpler to validate and monitor. Favored where auditability and regulatory clarity are primary constraints.

- Gradient Boosting (GBM/XGBoost/etc.): Usually higher predictive power on complex features, but inherently less transparent. Explainability tools (SHAP, LIME, surrogate models) help but do not fully replace the straightforwardness of a logistic scorecard when supervisors demand clear causal or economic interpretation.

Performance vs. operational risk

- GBM advantage: can capture nonlinearities and interactions automatically, improving discrimination (AUC) and possibly economic value (better pricing/approval).

- GBM downside: greater model risk, harder to validate, more sensitive to data shifts, requires stronger monitoring, and can be harder to implement into legacy decision pipelines (scorecard points, thresholds). This increases governance burden and potentially supervisory scrutiny.

Regulatory & business considerations

- Regulators (and model risk policies) prioritize demonstrable controls: data lineage, feature definition, stability, backtesting and explainability. A high-performing model that cannot be explained or validated to a regulator is a liability.

- In many real programs the pragmatic approach is: start with a well-documented logistic/WoE baseline (scorecard) to satisfy business/regulatory requirements, then explore more complex models as a complementary channel (e.g., second-look automated decisions, or a GBM whose outputs are mapped to a calibrated scorecard), but only after implementing stronger validation, governance and ongoing monitoring for the complex model.
  
---
# 🔍 Task 2: Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is conducted to understand the structure, quality, and statistical properties of the dataset before modeling.

✔ Data Overview

Inspection of data shape, column names, and data types

Identification of numerical vs categorical variables

✔ Missing Value Analysis

Detection of missing or null values

Assessment of their potential impact on modeling

✔ Univariate Analysis

Distribution analysis of numerical features (e.g., Amount, Value)

Frequency counts for categorical variables

✔ Bivariate Analysis

Relationships between features and the fraud/risk-related indicators

Correlation analysis among numerical variables

✔ Key Insights

Identification of skewed distributions and outliers

Motivation for feature scaling and aggregation

Justification for downstream feature engineering steps
---
⚖️ Task 3: Feature Engineering Pipeline

A robust data processing pipeline is built using sklearn.pipeline.Pipeline.

✔ Aggregate Features

Computed per customer:

Total Transaction Amount

Average Transaction Amount

Transaction Count

Standard Deviation of Transaction Amounts
✔ Date-Time Feature Extraction

From TransactionStartTime:

Transaction Hour

Transaction Day

Transaction Month

Transaction Year

✔ Categorical Encoding

One-Hot Encoding for nominal variables

Handled using ColumnTransformer

✔ Feature Scaling

Standardization (StandardScaler) applied to numerical features

WoE transformation implemented using the xverse library

Used to improve interpretability and monotonic relationship with risk

IV used to assess predictive strength of features

---
# 🎯 Task 4: Credit Risk Target Engineering (RFM-Based)

Step 1: RFM Metrics

For each CustomerId:

Recency: Days since last transaction

Frequency: Number of transactions

Monetary: Total transaction value

A fixed snapshot date ensures consistency.

Step 2: Customer Segmentation

K-Means clustering (k=3) applied on scaled RFM features

random_state fixed for reproducibility

Step 3: High-Risk Label Assignment

Cluster with lowest frequency and monetary value identified as high risk

Binary target created:

is_high_risk = 1 → High-risk customers

is_high_risk = 0 → Others

Step 4: Integration

Target variable merged back into the processed dataset

---

# 🤖 Task 5: Model Training & Experiment Tracking
Models Trained

Logistic Regression

Random Forest

Data Preparation

Train-test split with fixed random_state

Pipeline-compatible feature matrix

Hyperparameter Tuning

Grid Search / Random Search

Experiment Tracking (MLflow)

Logged for each run:

Model parameters

Evaluation metrics

Model artifacts

Best-performing model is registered in the MLflow Model Registry.

---

# 📊 Model Evaluation Metrics

Models are evaluated using:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

 🧪 Unit Testing

Implemented using pytest

Tests validate:

Feature engineering outputs

Expected columns and transformations

Tests are located in:
bash
```
  tests/test_data_processing.py
```
🚀 Task 6: Model Deployment (FastAPI)
API Features

Built with FastAPI

Loads best model from MLflow Registry

Endpoint:

POST /predict

Returns probability of high credit risk

Data Validation

Request & response schemas defined using Pydantic

---
# 🐳 Containerization


Dockerized FastAPI application

docker-compose.yml simplifies build & run

Run Locally
bash
```
  docker-compose up --build
```

API available at:
bash
```
  http://localhost:8000/docs
```
