# 🛡️ Base Operacional – Detecção de Fraudes

## 📌 Visão Geral

A detecção de fraudes identifica transações suspeitas com base em desvios de comportamento do usuário.

Os principais fatores analisados são:

* valor da transação
* horário
* dispositivo utilizado
* tentativas recentes
* distância geográfica

O sistema combina esses sinais para estimar o risco de fraude e explicar a decisão.

---

## 🧠 Fatores de Risco

### 💰 Valor

* Valores elevados aumentam o risco
* Fraudadores tendem a maximizar ganhos

### 🌙 Horário

* Madrugada (00h–05h) é mais suspeita
* Horário comercial é padrão esperado

### 📱 Dispositivo

* Dispositivo novo aumenta o risco
* Pode indicar acesso indevido

### 🔁 Tentativas

* Muitas tentativas indicam:

  * invasão
  * automação
  * teste de limites

### 🌍 Distância

* Distância elevada pode indicar uso indevido
* Principalmente se incompatível com comportamento recente

---

## 🚨 Regras de Decisão

### 🔴 Alto Risco

Quando há combinação forte de fatores:

* valor alto + horário incomum
* dispositivo novo + múltiplas tentativas
* valor alto + dispositivo novo
* múltiplos fatores simultâneos

---

### 🟡 Risco Moderado

Quando há sinais mistos:

* valor médio + dispositivo novo
* múltiplas tentativas isoladas
* valor alto isolado
* distância elevada isolada

---

### 🟢 Baixo Risco

Quando o comportamento é normal:

* valor baixo
* dispositivo conhecido
* horário comum
* sem tentativas recentes

---

## ⚠️ Regras Importantes

* Um único fator raramente define fraude
* A combinação de fatores é o principal sinal
* Nem toda anomalia é fraude
* Fraudes podem ocorrer sem sinais evidentes

---

## 🧪 Casos Especiais

* Valor baixo + madrugada → geralmente não é fraude
* Valor alto isolado → risco moderado
* Todos sinais normais → baixo risco, mas não impossível

---

## 🎯 Objetivo

O sistema deve:

1. Classificar o risco (baixo, moderado, alto)
2. Explicar claramente a decisão
3. Manter consistência nas respostas

---

## 💬 Tipos de Resposta

### Predição

Classificar risco + justificar

### Explicação

Explicar um fator isolado

### Conhecimento

Responder sobre o sistema

### Híbrido

Classificação + explicação

### Edge Case

Responder com cautela

---

## 🧾 Formato de Resposta por Tipo de Pergunta

### Perguntas sobre uma transação específica
Responder no formato:

**[Nível de risco]. Motivo: [fatores relevantes].**

Exemplos:
- Alta probabilidade de fraude. Motivo: valor elevado, dispositivo novo e horário incomum.
- Risco moderado. Motivo: há sinais suspeitos, mas não suficientes para alto risco.
- Baixo risco de fraude. Motivo: comportamento consistente com o padrão normal.

### Perguntas conceituais sobre fraude
Explicar o conceito diretamente, sem classificar risco de uma transação inexistente.

Exemplo:
"Transações de madrugada são mais suspeitas porque fogem do padrão típico de uso."

### Perguntas sobre o sistema, modelo ou dataset
Responder com base na documentação recuperada.

Exemplo:
"O sistema utiliza valor, horário, dispositivo, tentativas recentes e distância geográfica para estimar risco de fraude."

### Perguntas híbridas
Classificar o risco e explicar os fatores usados.

Exemplo:
"Risco moderado. Motivo: apesar do valor baixo, múltiplas tentativas recentes indicam comportamento suspeito."

---

## ❌ Evitar

* Respostas vagas
* Falta de justificativa
* Ambiguidade
* "Pode ser fraude" sem explicação

---

## 🧠 Limitações

* Modelo baseado em dados sintéticos
* Nem todos padrões reais estão representados

---

# 🧪 Exemplos de Raciocínio (Generalizados)

Os exemplos abaixo ensinam padrões, não casos específicos.

---

### 🔴 Alto Risco

Situação:
Múltiplos fatores fortes combinados (valor alto, dispositivo novo, horário incomum)

Resposta:
Alta probabilidade de fraude. Motivo: combinação de fatores fortes de risco.

---

### 🟢 Baixo Risco

Situação:
Comportamento normal (valor baixo, horário comum, dispositivo conhecido)

Resposta:
Baixo risco de fraude. Motivo: ausência de sinais relevantes de risco.

---

### 🟡 Risco Moderado

Situação:
Sinais mistos (ex: valor médio + comportamento parcialmente suspeito)

Resposta:
Risco moderado. Motivo: presença de alguns fatores de risco, mas sem combinação forte.

---

### ⚠️ Edge Case

Situação:
Valor baixo em horário incomum

Resposta:
Provavelmente não é fraude. Motivo: valor baixo reduz significativamente o risco.

---

## 🧠 Regra Final

Sempre avaliar a combinação de fatores.

Quanto mais fatores fortes presentes, maior a probabilidade de fraude.
