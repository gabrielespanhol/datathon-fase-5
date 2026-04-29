# System Card — Fraud Detection System

## Overview

This System Card provides a comprehensive overview of the Fraud Detection System, including its intended use, limitations, ethical considerations, and deployment context. The system integrates traditional machine learning, Retrieval-Augmented Generation (RAG), and a ReAct agent to detect fraudulent financial transactions.

## Intended Use

### Primary Use Cases
- Real-time fraud detection for online financial transactions
- Risk assessment and decision support for payment processors
- Automated flagging of suspicious activities in banking systems
- Integration with existing fraud prevention workflows

### Target Users
- Financial institutions and payment processors
- Risk management teams
- Compliance officers
- Data scientists and ML engineers

### Out-of-Scope Uses
- Direct decision-making without human oversight
- Use in non-financial domains
- Real-time trading or high-frequency applications

## System Architecture

### Components
1. **Data Pipeline**: DVC-managed synthetic data generation and feature engineering
2. **ML Model**: Logistic Regression baseline with MLflow tracking and model registry
3. **RAG Pipeline**: SentenceTransformer embeddings with semantic search over documentation
4. **Agent**: ReAct-based agent with custom tools for prediction, explanation, and knowledge retrieval
5. **API Layer**: FastAPI endpoints for prediction, agent interaction, and metrics
6. **Monitoring**: Prometheus metrics, drift detection, and Grafana dashboards
7. **Security**: Input guardrails and basic output validation

### Data Flow
1. Raw transaction data → Feature engineering → ML prediction
2. User query → Agent classification → Tool execution → Response generation
3. System metrics → Prometheus → Grafana visualization

## Performance Characteristics

### Accuracy and Reliability
- ML Model: ROC AUC ~0.85 on synthetic test data
- RAG Retrieval: Context availability ~90%, semantic similarity ~0.75
- Agent: Correctness ~8.5/10, Clarity ~8.0/10, Business alignment ~7.5/10 (based on LLM judge evaluation)

### Latency
- Prediction endpoint: <100ms per request
- Agent response: <5s per query (including LLM generation)
- RAG retrieval: <500ms per query

### Scalability
- Designed for containerized deployment (Docker)
- Supports horizontal scaling via Kubernetes
- Memory usage: ~2GB for model + embeddings + LLM

## Limitations and Biases

### Technical Limitations
- Trained exclusively on synthetic data; performance on real-world data unvalidated
- Limited to 5 features; may miss complex fraud patterns
- LLM responses depend on prompt quality and context availability
- No real-time retraining or online learning

### Data Biases
- Synthetic data may not capture all real-world fraud patterns
- Potential bias in feature engineering (e.g., log transformations may favor certain transaction ranges)
- Lack of demographic or geographic diversity in synthetic generation

### Operational Limitations
- Requires manual intervention for model updates
- Drift detection is basic (KS test); may miss subtle changes
- No automated alerting or incident response

## Ethical Considerations

### Fairness
- System may disproportionately flag certain transaction types
- Potential for false positives affecting legitimate users
- No explicit fairness constraints in training

### Privacy
- Processes transaction data; no PII handling in current implementation
- Synthetic data generation avoids real user data exposure
- See LGPD_PLAN.md for privacy compliance details

### Safety
- Guardrails prevent prompt injection but may not catch all adversarial inputs
- Agent responses are not guaranteed to be safe or appropriate
- No content moderation for generated text

## Deployment and Maintenance

### Deployment
- Containerized via Docker Compose
- Requires GPU for LLM inference (CUDA support)
- Environment variables for configuration
- Health checks and graceful shutdown implemented

### Monitoring and Maintenance
- Prometheus metrics for operational monitoring
- MLflow for model versioning and experiments
- Drift detection reports in JSON format
- Regular retraining recommended (monthly basis)

### Incident Response
- Logs available via application logging
- Metrics dashboard for real-time monitoring
- Manual intervention required for model updates
- Backup procedures via DVC and MLflow

## Recommendations

### For Users
- Always combine with human review for high-risk decisions
- Monitor performance metrics regularly
- Validate on real data before production deployment

### For Developers
- Implement more robust guardrails and adversarial testing
- Add automated retraining pipelines
- Enhance monitoring with alerting rules
- Conduct thorough fairness and bias audits

## References

- Mitchell, M. et al. (2019). Model Cards for Model Reporting. FAT* Conference.
- Sculley, D. et al. (2015). Hidden Technical Debt in Machine Learning Systems. NeurIPS.
- OWASP Top 10 for Large Language Model Applications (2025)

---

*Last updated: April 2026*