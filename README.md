# Hospital Readmission Prediction: Production ML System

> **Enterprise-grade MLOps platform for healthcare predictive analytics using Databricks, AWS SageMaker, and Kubernetes**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Terraform](https://img.shields.io/badge/Terraform-1.0+-purple.svg)](https://www.terraform.io/)
[![AWS](https://img.shields.io/badge/AWS-Cloud-orange.svg)](https://aws.amazon.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-EKS-326CE5.svg)](https://kubernetes.io/)

---

## Table of Contents
- [Business Context](#business-context)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
- [Architecture & Design Decisions](#architecture--design-decisions)
- [Technology Stack](#technology-stack)
- [Project Journey](#project-journey)
- [Results & Impact](#results--impact)
- [Contact](#contact)

---

## Business Context

Hospital readmissions within 30 days represent a critical challenge for healthcare systems globally:

- **UK**: NHS spends **£2.5 billion/year** on preventable readmissions (15-20% rate)
- **US**: Medicare spends **$26 billion/year** on unplanned readmissions (~20% of discharged patients)
- **Regulatory Impact**: Both CMS and NHS actively penalize hospitals with high readmission rates

This makes readmission prediction a **top priority** for healthcare organizations worldwide.

### The Challenge

**Predict whether a diabetic patient will be readmitted to the hospital within 30 days** based on:
- Demographics (age, gender, race)
- Medical history (diagnoses, medications, procedures)
- Hospital utilization patterns
- Lab results and medication changes

### The Dataset

**101,766 patient records** from 130 US hospitals (1999-2008)
- Source: [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- Features: 73 original attributes
- Target: Binary classification (readmitted ≤30 days vs. not readmitted)

---

## Solution Overview

This project demonstrates a **production-grade machine learning system** that predicts readmission risk for diabetic patients, enabling proactive clinical interventions.

**What makes this production-grade?**
- ✅ **Automated MLOps**: Self-healing pipelines with weekly retraining and drift-triggered retraining
- ✅ **GenAI Integration**: AI-powered explanations and similar patient search using AWS Bedrock
- ✅ **Observability**: Comprehensive logging, metrics (Prometheus/Grafana), and alerting
- ✅ **Enterprise Security**: HTTPS/TLS, WAF protection, IAM role-based access, secrets management
- ✅ **High Availability**: Multi-AZ deployment (us-east-1a, us-east-1b) with 3-replica services
- ✅ **Scalable Infrastructure**: Kubernetes on EKS with auto-scaling and zero-downtime updates

---

## Architecture

![Architecture Overview](docs/diagrams/architecture-overview.png)

### System Components

The system is organized into four key workflows:

#### 1. **User Flow** (Prediction Service)
```
User → Route53 → WAF → ALB → EKS → FastAPI → Model Prediction
```
- **Route53**: DNS routing (`api.machinelearning.hamidhirsi.com`)
- **WAF**: DDoS protection, rate limiting (2000 req/5min), OWASP rules
- **ALB**: HTTPS termination with ACM certificates
- **EKS**: Multi-AZ Kubernetes cluster with 3-replica FastAPI pods
- **Model Serving**: Zero-downtime model reload (5-min polling from S3)

#### 2. **ML Lifecycle** (Training & Deployment)
```
EventBridge (Schedule + Drift Alarm) → Step Functions → SageMaker → Lambda Evaluation → MLflow → S3 → FastAPI
```
- **Triggers**: Dual-mode triggering
  - Time-based: Weekly schedule (Sundays 2 AM UTC)
  - Event-based: CloudWatch alarm when drift detected (high-risk predictions > 20%)
- **Orchestration**: AWS Step Functions with error handling
- **Training**: SageMaker with XGBoost on ml.m5.2xlarge instances
- **Validation**: Lambda evaluates ROC-AUC (must be > 0.75)
- **Registry**: MLflow on Databricks for model versioning
- **Promotion**: Automatic staging → production transition on validation pass

#### 3. **RAG Pipeline** (AI Explanations)
```
S3 Upload → S3 Event → SQS → Lambda → Bedrock Embeddings → Qdrant Vector DB
```
- **Ingestion**: Event-driven embedding generation for 81k+ patients
- **Embeddings**: AWS Bedrock Titan Embeddings (1,536 dimensions)
- **Storage**: Qdrant vector database on Kubernetes
- **Search**: Semantic similarity search for case-based reasoning
- **Explanations**: Claude 3.5 Sonnet generates clinical insights

#### 4. **Observability**
```
All Services → CloudWatch Logs + Prometheus → Grafana Dashboards → SNS Alerts
```
- **Logging**: Structured JSON logs in CloudWatch
- **Metrics**: Prometheus scrapes FastAPI `/metrics`, Kubernetes node metrics
- **Visualization**: Grafana dashboards for API latency, throughput, model performance
- **Alerting**: SNS email notifications for training failures, CloudWatch alarms for drift detection

## Architecture & Design Decisions

Building a production ML system requires making deliberate architectural choices. Here's why I chose specific technologies and patterns for this project.

### **Why Databricks + SageMaker (Not Just SageMaker)?**

**The Question**: Why use both Databricks and SageMaker when SageMaker has notebooks and feature store?

**The Answer**: Separation of concerns and team scalability.

- **Databricks**: Data engineering team's territory. PySpark pipelines for distributed ETL, Delta Lake for versioned features, collaborative notebooks for exploratory analysis.
- **SageMaker**: ML engineering team's territory. Managed training jobs, hyperparameter tuning, model artifacts in S3.
- **MLflow on Databricks**: Single source of truth for model registry accessible to both teams.

This pattern mirrors real enterprise setups where data engineers and ML engineers work in different tools but share a common registry. SageMaker Feature Store would have simplified this, but the Databricks + Delta Lake approach provides better feature lineage and time-travel capabilities.

### **Why Event-Driven RAG Pipeline (Not Real-Time Sync)?**

**The Question**: Why use S3 → SQS → Lambda instead of directly calling Bedrock from FastAPI?

**The Answer**: Decoupling and cost control.

- **S3 Event Notifications**: Trigger embedding generation automatically when new data arrives
- **SQS Queue**: Buffer for Lambda functions, handles traffic spikes gracefully
- **Lambda**: Auto-scales to 1000 concurrent executions, processes embeddings asynchronously
- **Dead Letter Queue**: Captures failures for retry without losing data

Real-time embedding generation would increase API latency by 500-1000ms per request. Pre-computing embeddings offline keeps prediction latency low and Bedrock costs predictable.

### **Why Drift Detection via CloudWatch Alarms (Not Statistical Tests)?**

**The Question**: Why use simple metric thresholds instead of KS tests or PSI for drift detection?

**The Answer**: Pragmatism over perfection.

CloudWatch Metric Filters extract `HighRiskPredictions / TotalPredictions` from existing FastAPI logs. When this ratio exceeds 20% (baseline is 11%), it signals potential data drift. This approach:
- **No new code**: Uses existing prediction logs
- **No new infrastructure**: CloudWatch alarms already in place
- **Actionable**: Triggers retraining immediately via EventBridge

Statistical tests (KS, PSI) would be more rigorous but require storing historical distributions, running scheduled jobs, and managing state. For a v1 drift detection system, simple heuristics provide 80% of the value with 20% of the complexity.

### **Why Step Functions (Not Airflow or Prefect)?**

**The Question**: Why AWS Step Functions instead of a dedicated orchestrator like Airflow?

**The Answer**: Serverless simplicity and AWS-native integration.

Step Functions orchestrates the ML pipeline: SageMaker training → Lambda evaluation → Lambda promotion. Benefits:
- **No infrastructure**: No Airflow servers to manage, patch, or scale
- **Native AWS**: Direct integration with SageMaker, Lambda, SNS without custom operators
- **Visual editor**: Stakeholders can see the pipeline flow in AWS Console
- **Pay-per-use**: Only pay for state transitions, not idle server time

For complex DAGs with dozens of tasks and custom sensors, Airflow wins. For a linear ML pipeline with 3-4 steps, Step Functions is the right tool.

### **Why MLflow on Databricks (Not SageMaker Model Registry)?**

**The Question**: Why host MLflow on Databricks instead of using SageMaker's built-in model registry?

**The Answer**: Consistency with data engineering workflow.

Since feature engineering happens in Databricks notebooks, keeping the model registry there creates a unified workflow:
1. Data engineers create features in Databricks → Delta Lake
2. ML engineers log experiments in Databricks → MLflow
3. SageMaker training jobs register models → MLflow (via API)
4. FastAPI pulls models from S3 (MLflow-backed storage)

SageMaker Model Registry would work, but it creates a split: features in Databricks, models in SageMaker. MLflow acts as the integration layer between both platforms.

### **Why External Secrets Operator (Not Hardcoded Secrets)?**

**The Question**: Why use External Secrets Operator instead of Kubernetes secrets directly?

**The Answer**: Security and centralization.

External Secrets Operator syncs AWS Secrets Manager → Kubernetes secrets automatically:
- **Single source of truth**: Secrets managed in AWS Secrets Manager (audited, versioned, rotated)
- **No secret sprawl**: No hardcoded secrets in Helm charts or ConfigMaps
- **Automatic rotation**: When secrets rotate in AWS, pods get updated automatically

This pattern is essential for enterprise compliance (SOC 2, HIPAA). Hardcoded secrets would fail security audits immediately.

### **Why Multi-AZ Deployment (Not Single AZ)?**

**The Question**: Why deploy across us-east-1a and us-east-1b instead of single AZ?

**The Answer**: High availability for production SLAs.

Multi-AZ protects against:
- **AZ failures**: AWS occasionally has zone-level outages (us-east-1a went down in 2021)
- **Planned maintenance**: Can drain one AZ for upgrades without downtime
- **Load distribution**: ALB distributes traffic across zones, reducing blast radius

For a portfolio project, single AZ would suffice. But this demonstrates understanding of production reliability requirements.

### **Why Terraform (Not AWS Console or CDK)?**

**The Question**: Why Terraform instead of ClickOps or AWS CDK?

**The Answer**: Infrastructure reproducibility and multi-cloud portability.

Terraform provides:
- **Version control**: Infrastructure changes tracked in Git
- **Plan before apply**: See changes before execution, prevent accidents
- **Modular design**: Reusable VPC, EKS, Lambda modules across projects
- **Declarative syntax**: State file ensures convergence to desired config

AWS CDK is excellent for TypeScript/Python developers, but Terraform's HCL is more readable for infrastructure teams and easier to onboard new contributors.

---
## Technology Stack

### **Infrastructure & DevOps**
| Technology | Purpose |
|------------|---------|
| **AWS EKS** | Managed Kubernetes for container orchestration |
| **Terraform** | Infrastructure as Code (100% of AWS resources) |
| **Helm** | Kubernetes package management |
| **GitHub Actions** | CI/CD automation |
| **Docker** | Container images for FastAPI, Streamlit, Qdrant |

### **Machine Learning**
| Technology | Purpose |
|------------|---------|
| **Databricks** | Data engineering with PySpark, Delta Lake feature store |
| **AWS SageMaker** | Distributed model training (XGBoost, PyTorch) |
| **MLflow** | Model versioning, experiment tracking, registry |
| **XGBoost** | Primary classification model (production) |
| **PyTorch** | Neural network experimentation |
| **Scikit-Learn** | Baseline models, preprocessing |

### **GenAI & RAG**
| Technology | Purpose |
|------------|---------|
| **AWS Bedrock** | Claude 3.5 Sonnet (explanations), Titan Embeddings |
| **Qdrant** | Vector database for patient similarity search |

### **Data Engineering**
| Technology | Purpose |
|------------|---------|
| **PySpark** | Distributed data processing on Databricks |
| **Delta Lake** | Versioned feature store with ACID transactions |
| **S3** | Data lake for raw data, features, models |
| **Pandas** | Local data manipulation |

### **Application Layer**
| Technology | Purpose |
|------------|---------|
| **FastAPI** | High-performance ML serving API |
| **Streamlit** | Interactive web UI for predictions |
| **Uvicorn** | ASGI server for FastAPI |

### **Observability**
| Technology | Purpose |
|------------|---------|
| **Prometheus** | Metrics collection and storage |
| **Grafana** | Metrics visualization dashboards |
| **CloudWatch** | AWS-native logging and alarms |
| **SNS** | Email alerting for critical events |

### **Security**
| Technology | Purpose |
|------------|---------|
| **AWS WAF** | Web application firewall, rate limiting |
| **ACM** | SSL/TLS certificates for HTTPS |
| **IAM IRSA** | Kubernetes pod-level AWS permissions |
| **External Secrets Operator** | Sync AWS Secrets Manager → Kubernetes secrets |
| **AWS Secrets Manager** | Centralized credential storage |

### **Serverless & Event-Driven**
| Technology | Purpose |
|------------|---------|
| **AWS Lambda** | Model evaluation, promotion, RAG ingestion |
| **Step Functions** | ML pipeline orchestration |
| **EventBridge** | Scheduled and event-driven triggers |
| **SQS** | Decoupled RAG ingestion queue |

---

## Project Journey

This project evolved through **6 phases**, transitioning from local prototyping to enterprise-grade production infrastructure.

### Phase 1: Data Engineering & Feature Development ✅

- Configured Databricks workspace with AWS S3 integration
- Built PySpark pipelines for data ingestion, cleaning, and transformation
- Engineered 44 features from 73 raw attributes (demographics, hospital utilization, medication complexity, comorbidities, interaction features)
- Implemented Delta Lake feature store for versioned features
- Integrated MLflow experiment tracking
- **Result**: Processed 101,766 patient records → 81,412 training / 20,354 test samples

![Feature Engineering Notebook](docs/screenshots/feature-engineering-notebook.png)
*Databricks Jupyter notebook showing PySpark feature engineering pipeline*

![Model Training Notebook](docs/screenshots/model-training-notebook.png)
*Jupyter notebook demonstrating model training with Scikit-Learn and PyTorch*

---

### Phase 2: ML Training & Model Registry ✅

- Trained XGBoost and PyTorch models on SageMaker
- Tuned hyperparameters (class weights, learning rates, depths, architectures)
- Configured MLflow model registry on Databricks with stage management (None → Staging → Production)
- **Best Model**: XGBoost (Tuned) with scale_pos_weight=8.0, max_depth=3, learning_rate=0.1, n_estimators=200
- **Performance**: ROC-AUC 0.66, Precision 0.17, Recall 0.57, F1-Score 0.26

![MLflow Model Registry](docs/screenshots/mlflow-model-registry.png)
*Databricks MLflow Model Registry showing hospital-readmission-model with 2 versions*

![MLflow Model Version](docs/screenshots/mlflow-model-version.png)
*MLflow Model Registry Version 1 details with transition options*

![MLflow Experiment Run](docs/screenshots/mlflow-experiment-run.png)
*MLflow experiment run showing training metrics*

![SageMaker Training Job](docs/screenshots/sagemaker-training-job.png)
*AWS SageMaker training job status history*

---

### Phase 3: Kubernetes Deployment & Infrastructure ✅

- Deployed multi-AZ EKS cluster with FastAPI backend (3 replicas) and Streamlit frontend
- Configured ALB with HTTPS (ACM certificates), Route53 DNS, AWS WAF (rate limiting 2000 req/5min)
- Implemented Prometheus + Grafana monitoring stack
- Deployed External Secrets Operator for AWS Secrets Manager integration
- **Infrastructure**: 100% Terraform-managed, multi-AZ high availability, auto-scaling

---

### Phase 4: GenAI/RAG Integration ✅

- Integrated AWS Bedrock (Claude 3.5 Sonnet for explanations, Titan Embeddings for patient similarity)
- Deployed Qdrant vector database on Kubernetes with 81,412 patient embeddings
- Built event-driven RAG pipeline (S3 → SQS → Lambda → Bedrock → Qdrant)
- Created `/chat` endpoint for conversational AI and `/similar-patients` for semantic search

![AWS Bedrock CloudWatch Metrics](docs/screenshots/bedrock-cloudwatch-metrics.png)
*AWS Bedrock CloudWatch dashboard showing invocation metrics*

---

### Phase 5: MLOps Automation ✅

- Implemented automated retraining pipeline with dual triggers:
  - Time-based: EventBridge schedule (Sundays 2 AM UTC)
  - Event-based: CloudWatch alarm on drift detection (high-risk predictions > 20%)
- Built Step Functions orchestration (SageMaker training → Lambda evaluation → MLflow promotion)
- Configured zero-downtime model reload (FastAPI polls S3 every 5 minutes, hot-swaps models)
- Created GitHub Actions CI/CD workflows for infrastructure and application deployment

---

### Phase 6: Production Hardening ✅

- Implemented security hardening (private subnets, VPC endpoints, TLS 1.2+, WAF rules, IAM IRSA)
- Configured health checks (liveness/readiness probes), graceful degradation, error handling
- Optimized costs (VPC endpoints, S3 lifecycle policies, Lambda concurrency limits)
- Ensured high availability (Pod Disruption Budgets, multi-AZ, resource requests/limits)

---
## Results & Impact

### **Model Performance**
| Metric | Value | Clinical Interpretation |
|--------|-------|------------------------|
| **ROC-AUC** | 0.66 | Better than random (0.5), aligned with published research |
| **Precision** | 0.17 | Model prioritizes recall over precision due to class imbalance |
| **Recall** | 0.57 | Identifies 57% of actual readmissions |
| **F1-Score** | 0.26 | Reflects precision-recall tradeoff for minority class |

**Clinical Use Case**:
- Model outputs **risk scores (0-1)**, not binary predictions
- Hospitals can set custom thresholds based on resource capacity
- Example: Top 20% highest-risk patients → proactive follow-up calls
- Even modest AUC provides significant value for resource allocation

---

### **System Architecture Capabilities**
| Component | Configuration | Expected Performance |
|-----------|--------------|---------------------|
| **API Deployment** | Multi-AZ, 3 replicas, HPA | Designed for high availability |
| **WAF Rate Limit** | 2,000 requests/5min | Protects against DDoS |
| **Model Reload** | 5-min S3 polling | Zero-downtime updates |
| **Vector Search** | Qdrant HNSW index | Sub-second semantic search |

---

### **Cost Optimization Strategies**
| Strategy | Expected Impact | Implementation |
|----------|----------------|----------------|
| **VPC Endpoints** | Reduces NAT costs | S3/ECR/CloudWatch traffic stays in AWS network |
| **Spot Instances** | Reduces EC2 costs | EKS worker nodes (non-critical workloads) |
| **S3 Lifecycle Policies** | Reduces storage costs | Archive old models to Glacier after 90 days |
| **Lambda Reserved Concurrency** | Prevents runaway costs | Caps Bedrock invocations |

---

### **Top Predictive Features**
Feature importance analysis from XGBoost model:

1. **`number_inpatient`** (0.18) - Prior inpatient visits in past year
2. **`time_in_hospital`** (0.14) - Length of current hospital stay
3. **`num_medications`** (0.11) - Number of medications prescribed
4. **`num_procedures`** (0.09) - Number of procedures during stay
5. **`discharge_disposition_id`** (0.07) - Discharge destination (home, SNF, etc.)

**Clinical Insight**: Patients with high hospital utilization (frequent admissions, long stays, complex medication regimens) are strongest readmission predictors.

---

---

## About This Project

This project was built as a **portfolio demonstration** of production-grade MLOps engineering skills. It showcases:
- End-to-end ML system design (data → training → deployment → monitoring)
- Enterprise infrastructure (Kubernetes, Terraform, CI/CD)
- Modern MLOps tools (SageMaker, MLflow, Bedrock)
- Software engineering best practices (IaC, testing, observability)

**Not included in this public repository**:
- Source code (private repository to prevent plagiarism)
- Infrastructure configurations (contains sensitive AWS account details)
- Credentials and API keys

**What you're seeing**: Case study documentation with architecture diagrams and screenshots.

---

## Contact

**Hamid Hirsi**
Machine Learning Engineer | MLOps Specialist

- **GitHub**: [@hamidhirsi](https://github.com/hamidhirsi)
- **LinkedIn**: [linkedin.com/in/hamidhirsi](https://linkedin.com/in/hamidhirsi)
- **Email**: hamidhirsi7@gmail.com

**Interested in discussing this project?** I'm happy to walk through the technical architecture, design decisions, and lessons learned in an interview setting.

---

## Acknowledgments

- **Dataset**: [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- **Inspiration**: Real-world MLOps best practices from AWS, Databricks, and Netflix engineering blogs
- **Community**: AWS, Kubernetes, and MLOps communities for documentation and support

---

*Last Updated: November 2025*
