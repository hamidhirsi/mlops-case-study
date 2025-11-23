# Production MLOps System for Medicine/Healthcare Industry

> **Enterprise-grade MLOps platform for healthcare using Pytorch, Databricks, AWS SageMaker, and Kubernetes**

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

## Business Problem/Context

Hospital readmissions within 30 days represent a critical challenge for healthcare systems globally:

- **UK**: NHS spends **£2.5 billion/year** on preventable readmissions (15-20% rate)
- **US**: Medicare spends **$26 billion/year** on unplanned readmissions (~20% of discharged patients)
- **Regulatory Impact**: Both CMS and NHS actively penalise hospitals with high readmission rates

This makes readmission prediction a **top priority** for healthcare organisations worldwide.

### Business Goal

**Predict whether a diabetic patient will be readmitted to the hospital within 30 days** based on:
- Demographics (age, gender, race)
- Medical history (diagnoses, medications, procedures)
- Hospital utilisation patterns
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
- ✅ **Automated MLOps**: MLOps pipelines with weekly scheduled retraining and drift-detection triggered retraining
- ✅ **GenAI Integration**: AI Chatbot that can explain the Models predictions and semantic search using AWS Bedrock, RAG, and Vector Database
- ✅ **Observability**: Comprehensive logging, metrics (Prometheus/Grafana), and alerting
- ✅ **Enterprise Security**: HTTPS/TLS, WAF protection, IAM role-based access, secrets management
- ✅ **High Availability**: Multi-AZ deployment (us-east-1a, us-east-1b) with 3-replica services
- ✅ **Scalable Infrastructure**: Terraform IaC + Helm Charts on Kubernetes, with auto-scaling and zero-downtime updates

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

Production ML systems require deliberate architectural choices that balance scalability, maintainability, and operational complexity. This section outlines the key design decisions and their rationale.

### **Hybrid Cloud ML Platform: Databricks + SageMaker**

The architecture leverages both Databricks and SageMaker to separate data engineering from ML engineering concerns, mirroring enterprise team structures.

**Databricks** handles data-intensive workloads: PySpark pipelines process 100k+ patient records, Delta Lake provides versioned feature storage with time-travel capabilities, and collaborative notebooks enable exploratory data analysis. This platform serves as the data engineering layer where features are engineered and validated.

**SageMaker** manages model training infrastructure: distributed training jobs run XGBoost and PyTorch models on managed compute (ml.m5.2xlarge instances), hyperparameter tuning experiments execute in parallel, and model artifacts are stored in S3 with versioning. This separation allows ML engineers to focus on model development without managing Spark clusters.

**MLflow on Databricks** acts as the integration point, providing a unified model registry accessible to both platforms. SageMaker training jobs register models via MLflow APIs, while FastAPI pulls production models from MLflow-backed S3 storage. This pattern maintains feature lineage (Databricks → Delta Lake → MLflow) while enabling flexible training infrastructure (SageMaker, local, Databricks clusters).

### **Event-Driven RAG Architecture**

The GenAI pipeline uses an asynchronous, event-driven architecture to decouple embedding generation from prediction serving, optimizing for both latency and cost.

Patient data uploaded to S3 triggers Event Notifications that publish to an SQS queue. Lambda functions consume messages, generate embeddings via Bedrock Titan (1,536 dimensions), and write to Qdrant vector database. This architecture ensures prediction API latency remains under 200ms—calling Bedrock synchronously would add 500-1000ms per request.

SQS provides buffering and backpressure management, preventing Bedrock throttling during traffic spikes. Lambda's auto-scaling (up to 1000 concurrent executions) handles batch processing efficiently. Dead Letter Queues capture failures for manual retry, ensuring no data loss.

Cost predictability is another key benefit: pre-computed embeddings eliminate per-request Bedrock charges, moving costs from variable (per-prediction) to fixed (per-data-upload). For 81,412 patients, embeddings are generated once rather than on-demand for each similarity search.

### **Pragmatic Drift Detection**

Drift detection uses CloudWatch Metric Filters to extract prediction patterns from existing FastAPI logs, avoiding the complexity of statistical distribution comparisons.

The system tracks two metrics: total predictions and high-risk predictions (probability > 0.6). When the ratio exceeds 20% (baseline is 11% from class distribution), a CloudWatch alarm triggers EventBridge, which invokes the same Step Functions retraining workflow used for weekly schedules. This heuristic detects distribution shifts without storing historical feature distributions or running Kolmogorov-Smirnov tests.

The implementation requires zero additional code in FastAPI—logs are already written to CloudWatch for debugging. CloudWatch alarms provide built-in alerting infrastructure, eliminating the need for custom monitoring services. While statistical tests (PSI, KS) would detect subtle drift earlier, this approach provides actionable signals with minimal operational overhead.

### **Serverless MLOps with Step Functions**

Step Functions orchestrates the ML pipeline (training → evaluation → promotion) using AWS-native integrations, eliminating the need for dedicated orchestration infrastructure.

The state machine coordinates SageMaker training jobs, Lambda evaluation functions (ROC-AUC validation), and Lambda promotion logic (MLflow stage transitions). Retry logic and error handling are declarative, reducing custom code. SNS notifications alert on failures, while CloudWatch Logs capture execution traces.

This serverless approach avoids running Airflow or Prefect servers, which require compute resources, patching, and monitoring. For linear ML pipelines with 3-4 tasks, Step Functions' visual editor and pay-per-execution model outweigh Airflow's advantages (complex DAGs, custom sensors, community operators). The pipeline executes weekly and on-demand, making variable costs (state transitions) more economical than fixed costs (server uptime).

### **Centralized Model Registry with MLflow**

MLflow hosted on Databricks serves as the single source of truth for model metadata, bridging feature engineering and model training workflows.

Feature engineers log feature sets to Delta Lake, capturing schema and lineage. ML engineers log experiments to MLflow, tracking hyperparameters, metrics, and artifacts. SageMaker training jobs register models remotely via MLflow REST APIs, maintaining consistency across environments. Production deployments pull models from S3 paths registered in MLflow, ensuring FastAPI always references the correct "Production" stage model.

This unified registry prevents model-feature mismatches: the same MLflow run links to both the Delta Lake feature version and the trained model artifact. SageMaker Model Registry could serve this role, but keeping the registry in Databricks alongside features simplifies governance and provides better visibility for data science teams already working in that environment.

### **Kubernetes Secrets Management with External Secrets Operator**

Production Kubernetes deployments require secure, auditable credential management. External Secrets Operator syncs AWS Secrets Manager to Kubernetes secrets, centralizing secret lifecycle in AWS.

Secrets (Databricks tokens, Bedrock API keys, MLflow credentials) are stored in AWS Secrets Manager with encryption at rest, access logging, and automatic rotation policies. External Secrets Operator polls Secrets Manager every 1 hour, creating or updating Kubernetes secrets automatically. Pods reference these secrets as environment variables or mounted files.

This pattern eliminates hardcoded secrets in Helm charts or ConfigMaps, which would expose credentials in version control. Centralized secrets enable compliance requirements (SOC 2, HIPAA): audit logs track who accessed secrets, rotation policies enforce security hygiene, and IAM policies restrict access to specific pods via IRSA (IAM Roles for Service Accounts).

### **Multi-AZ Deployment for High Availability**

The system deploys across two availability zones (us-east-1a, us-east-1b) to meet production reliability requirements and minimize blast radius during failures.

Each AZ hosts EKS worker nodes, with pods distributed via Kubernetes anti-affinity rules. The Application Load Balancer routes traffic across zones, automatically removing unhealthy targets. This configuration tolerates single-AZ failures (outages, network partitions, planned maintenance) without service disruption.

Multi-AZ deployment provides additional benefits beyond fault tolerance: load distribution reduces per-zone network saturation, and Pod Disruption Budgets ensure at least 2 replicas remain available during voluntary disruptions (cluster upgrades, node drains). While costlier than single-AZ (cross-AZ data transfer fees), this architecture demonstrates understanding of production SLA requirements (99.9%+ uptime).

### **Infrastructure as Code with Terraform**

All AWS resources (VPC, EKS, Lambda, S3, IAM) are defined in Terraform HCL, enabling version-controlled, reproducible infrastructure.

Terraform's declarative syntax describes desired state, with the state file tracking actual deployed resources. The plan-before-apply workflow shows changes before execution, preventing accidental deletions. Modular design (VPC module, EKS module, Lambda module) promotes reusability across projects.

This approach eliminates manual AWS Console operations, which lack audit trails and reproducibility. Infrastructure changes follow the same code review process as application code—pull requests, automated Terraform plan in CI, manual approval, Terraform apply on merge. State stored in S3 with DynamoDB locking enables team collaboration without state conflicts.

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

**What you're seeing**: Case study documentation with architecture diagrams and screenshots.

---

## Contact

#### **Hamid Hirsi**
**Senior Platform Engineer - AI/ML**

- **GitHub**: [@hamidhirsi](https://github.com/hamidhirsi)
- **LinkedIn**: [linkedin.com/in/hamidhirsi](https://linkedin.com/in/hamidhirsi)
- **Email**: hamidhirsi7@gmail.com

#### **Interested in discussing this project?** 

I'm happy to walk through the technical architecture, design decisions, and lessons learned in an interview setting.

---

## Acknowledgments

- **Dataset**: [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- **Inspiration**: Real-world MLOps best practices from AWS, Databricks, and Netflix engineering blogs
- **Community**: AWS, Kubernetes, and MLOps communities for documentation and support

---

*Last Updated: November 2025*
