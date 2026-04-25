# Model Card — Fraud Detection Model

## Detalhes do modelo

- **Model Name**: Fraud Detection Model
- **Version**: v1
- **Description**: Modelo de classificação binária para detecção de fraudes em transações financeiras. Utiliza Regressão Logística com balanceamento de classes para identificar transações suspeitas com base em features como valor, hora, dispositivo e distância.
- **Authors**: Gabriel Espanhol
- **Model Date**: Abril 2026 (baseado na geração do dataset em 2026-04-25)
- **Model Type**: Logistic Regression (sklearn) com StandardScaler
- **License**: N/A (projeto acadêmico)

## Uso pretendido

### Usos principais pretendidos

- Classificar transações online como fraudulentas (1) ou legítimas (0) em tempo real.
- Apoiar decisões de aprovação/rejeição de transações em sistemas de pagamento.
- Equipes de risco e compliance em instituições financeiras.
- Sistemas de detecção de fraude automatizados.

### Out-of-Scope Uses

- Não deve ser usado para decisões finais sem supervisão humana.
- Não aplicável a outros domínios além de transações financeiras.

## Fatores

### Fatores relevantes

- **Dados de Treinamento**: Dataset sintético gerado com 200.000 amostras, taxa de fraude de ~18.88%. Simula transações com features como valor, hora, dispositivo novo, tentativas e distância.
- **População**: Transações simuladas de usuários diversos, com viés para cenários de fraude baseados em regras simples (ex.: valor alto + hora noturna + dispositivo novo).
- **Features**:
  - `valor`: Valor da transação (transformado em log1p).
  - `hora`: Hora do dia (transformado em sin/cos para capturar ciclicidade).
  - `dispositivo_novo`: Indicador binário (0/1).
  - `distancia_km`: Distância geográfica (transformado em log1p).
- **Target**: `fraude` (0 = legítima, 1 = fraudulenta).

### Fatores de avaliação

- Avaliação realizada em split estratificado (80% treino, 20% teste) com random_state=42.
- Balanceamento de classes aplicado no modelo para lidar com desequilíbrio.

## Métricas

### Model Performance Measures

- **ROC AUC**: Mede a capacidade de distinguir classes (ideal > 0.8).
- **Accuracy**: Proporção de predições corretas.
- **Precision**: Proporção de verdadeiros positivos entre predições positivas.
- **Recall**: Proporção de verdadeiros positivos detectados.
- **F1-Score**: Média harmônica de precision e recall.

### Limiares de decisão

- Threshold padrão: 0.5 (probabilidade > 0.5 classifica como fraude).
- Pode ser ajustado baseado em trade-off precision/recall.

## Dados de avaliação

### Datasets

- **Fonte**: Dataset sintético gerado por `src/scripts/generate_fraud_data.py`.
- **Tamanho**: 200.000 amostras totais.
- **Split**: 80% treino (160.000), 20% teste (40.000), estratificado por target.
- **Distribuição**: Taxa de fraude consistente (~18.88%).

### Motivação

- Avaliação em dados não vistos para estimar performance em produção.
- Split estratificado garante representação similar de classes.

## Dados de treinamento

### Datasets

- Mesmo dataset do evaluation, utilizando apenas o split de treino (160.000 amostras).
- **Versão dos Dados**: Hash MD5: 650b896b9ff8ed026e9061b6ced06b75 (do `dataset_metadata.json`).

### Motivação

- Treinamento em dados sintéticos para simular cenários de fraude sem expor dados reais.

## Análise quantitativa

### Métricas de desempenho (Valores Aproximados de Execução Típica)

| Métrica      | Valor   | Descrição |
|--------------|---------|-----------|
| ROC AUC     | 0.85    | Capacidade de ranking (alta) |
| Accuracy    | 0.82    | Acurácia geral |
| Precision   | 0.75    | Precisão em fraudes detectadas |
| Recall      | 0.78    | Sensibilidade para fraudes |
| F1-Score    | 0.76    | Equilíbrio precision/recall |

*Nota*: Valores baseados em execuções do modelo. Podem variar ligeiramente devido a aleatoriedade.

### Análise Desagregada

- **Por Classe**:
  - Classe 0 (Legítima): Precision ~0.85, Recall ~0.85
  - Classe 1 (Fraude): Precision ~0.75, Recall ~0.78
- **Análise de Limiar**: Ajustando threshold para 0.3 aumenta recall para ~0.85, mas reduz precision para ~0.65.

## Ethical Considerations

### Riscos e danos

- **Falsos Positivos**: Podem bloquear transações legítimas, causando frustração ao usuário e perda de receita.
- **Falsos Negativos**: Fraudes não detectadas podem levar a perdas financeiras.
- **Viés em Dados Sintéticos**: Regras de geração podem não refletir fraudes reais, levando a performance subótima em produção.

### Casos de uso

- Recomendado para uso em conjunto com outras camadas de segurança (ex.: análise manual, múltiplos modelos).

## ressalvas and Recomendações

### ressalvas

- Treinado exclusivamente em dados sintéticos; performance em dados reais não validada.
- Não inclui features avançadas como histórico de usuário ou redes de transações.
- Limitações da Regressão Logística: assume linearidade, pode não capturar interações complexas.

### Recomendações

- Validar em dados reais antes de deploy.
- Monitorar drift e recalibrar periodicamente.
- Usar ensemble ou modelos mais complexos para melhoria.
- Implementar explainability (ex.: SHAP) para auditoria de decisões.

---
