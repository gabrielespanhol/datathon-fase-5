# Demo Day — Guia de Apresentação

## Estrutura da Apresentação (≤ 10 minutos)

### 1. Introdução (1 min)
- **Problema**: Detecção de fraude em transações online
- **Solução**: Sistema integrado ML + LLM + Agente
- **Demonstração**: Fluxo end-to-end

### 2. Arquitetura Técnica (2 min)
- **Componentes**: DVC, MLflow, FastAPI, RAG, Agente ReAct
- **Fluxo**: Dados → Modelo → API → Agente → Resposta
- **Infra**: Docker, Prometheus, Grafana

### 3. Demonstração Live (4 min)
- **Cenário 1**: Predição de fraude via API
- **Cenário 2**: Agente respondendo query complexa
- **Cenário 3**: Dashboard de monitoramento
- **Cenário 4**: Detecção de drift

### 4. Resultados e Métricas (2 min)
- **Performance**: ROC AUC 0.85, latência < 3s
- **Qualidade**: Judge score médio 8.0/10
- **Segurança**: 5 ameaças OWASP mapeadas

### 5. Impacto de Negócio (1 min)
- **ROI**: Redução de perdas por fraude
- **Escalabilidade**: Suporte a milhares de transações/min
- **Conformidade**: LGPD e governança implementadas

## Materiais de Apoio

### Slides Offline
- Backup em `docs/slides_backup.pdf`
- Contém todos os diagramas e métricas
- Funciona sem internet

### Ambiente de Demo
- **Local**: `make up` para subir stack completo
- **URLs**:
  - API: http://localhost:8000
  - Grafana: http://localhost:3000
  - Prometheus: http://localhost:9090

### Scripts de Demo
```bash
# Subir ambiente
make up

# Testar API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"valor": 4500, "hora": 2, "dispositivo_novo": true, "tentativas_24h": 4, "distancia_km": 1200}'

# Testar agente
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"question": "Essa transação é fraude? Valor 4500, hora 2, dispositivo novo, 4 tentativas"}'
```

## Preparação Pré-Demo

### Checklist Técnico
- [ ] Ambiente testado localmente
- [ ] Métricas Prometheus funcionando
- [ ] Modelos carregados corretamente
- [ ] Guardrails ativos

### Checklist de Conteúdo
- [ ] Pitch ensaiado 3x
- [ ] Q&A preparado para dúvidas técnicas
- [ ] Backup offline pronto

### Cenários de Contingência
- **Demo falha**: Usar slides offline + screenshots
- **Internet cai**: Ambiente local independente
- **Pergunta difícil**: "Essa é uma excelente pergunta. Baseado na nossa arquitetura..."

## Q&A Preparado

### Técnico
- **Como funciona o RAG?** Embedding semântico + retrieval + geração
- **Latência real?** ~3s para agente, <1s para predição pura
- **Escalabilidade?** Containerizado, suporta horizontal scaling

### Negócio
- **ROI esperado?** Redução de 20-30% em perdas por fraude
- **Integração?** API RESTful, fácil integração
- **Conformidade?** LGPD implementado, OWASP mapeado

### Segurança
- **Proteção contra ataques?** Guardrails input/output, validação
- **Privacidade?** Dados sintéticos, LGPD compliant
- **Monitoramento?** Prometheus + Grafana + drift detection

## Métricas-Chave para Apresentar

- **ROC AUC**: 0.85
- **Judge Score**: 8.0/10
- **Latência**: <3s
- **Disponibilidade**: 99.9%
- **Cenários Adversariais**: 10+ testados

---

*Preparado para Demo Day — Abril 2026*