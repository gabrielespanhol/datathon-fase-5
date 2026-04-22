# 📊 Dataset Sintético de Fraude

## 🎯 Objetivo

Este dataset foi criado para simular transações financeiras com o objetivo de desenvolver e avaliar um sistema de detecção de fraude.

Os dados são **sintéticos** (gerados artificialmente), mas seguem **regras de negócio plausíveis**, inspiradas em padrões comuns de sistemas antifraude reais.

---

## 🧠 Visão Geral

Cada linha do dataset representa uma **transação financeira**.

O dataset contém variáveis comportamentais e contextuais que ajudam a identificar possíveis fraudes.

A variável alvo (`fraude`) indica se a transação foi considerada fraudulenta com base em regras de risco.

---

## 📂 Estrutura do Dataset

| Coluna              | Tipo      | Descrição |
|--------------------|----------|----------|
| valor              | float    | Valor da transação |
| hora               | int      | Hora do dia (0–23) |
| dispositivo_novo   | bool     | Indica se o dispositivo é novo |
| tentativas_24h     | int      | Número de tentativas nas últimas 24h |
| distancia_km       | float    | Distância em relação ao padrão do usuário |
| fraude             | int      | Variável alvo (0 = normal, 1 = fraude) |

---

## 🔍 Descrição das Variáveis

### 💰 valor
Representa o valor monetário da transação.

- Transações de valor elevado podem indicar maior risco de fraude.
- Gerado aleatoriamente entre 10 e 5000.

---

### 🕒 hora
Hora do dia em que a transação foi realizada.

- Fraudes são mais comuns em horários incomuns (madrugada).
- Valores entre 0 e 23.

---

### 📱 dispositivo_novo
Indica se a transação foi realizada em um dispositivo não reconhecido.

- `True`: dispositivo novo
- `False`: dispositivo conhecido

Uso de novos dispositivos pode indicar tentativa de fraude.

---

### 🔁 tentativas_24h
Número de tentativas de transação nas últimas 24 horas.

- Muitos acessos em curto período podem indicar comportamento suspeito.
- Valores entre 0 e 5.

---

### 📍 distancia_km
Distância entre a localização da transação e o comportamento habitual do usuário.

- Grandes distâncias podem indicar uso indevido da conta.
- Valores entre 0 e 2000 km.

---

### 🎯 fraude (target)
Variável alvo do modelo.

- `0`: transação normal
- `1`: transação fraudulenta

---

## ⚙️ Lógica de Geração de Fraude

A variável `fraude` é definida com base em um **score de risco** calculado a partir das variáveis.

Cada condição suspeita contribui para o aumento do risco:

- Valor alto
- Horário incomum
- Dispositivo novo
- Muitas tentativas recentes
- Grande distância geográfica

A fraude é atribuída quando múltiplos sinais de risco estão presentes.

---

## 🧠 Uso no Projeto

Este dataset é utilizado para:

- Treinar o modelo baseline de classificação
- Simular entradas da API
- Alimentar a camada LLM para explicação das decisões
- Testar monitoramento e detecção de drift

---

## 💬 Exemplo de Transação

```json
{
  "valor": 2450.90,
  "hora": 2,
  "dispositivo_novo": true,
  "tentativas_24h": 4,
  "distancia_km": 1200.5,
  "fraude": 1
}