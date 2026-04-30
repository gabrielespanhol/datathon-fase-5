# Fraud Detection — Projeto de Detecção de Fraudes

## Visão geral

Este projeto implementa um sistema de detecção de fraudes em transações financeiras sintéticas.
O objetivo é gerar dados, processá-los, treinar um modelo de classificação, expor uma API de inferência e oferecer monitoramento e controle de qualidade.

O sistema resolve o problema de identificar padrões suspeitos em transações usando:
- geração de dados sintéticos (`src/scripts/generate_fraud_data.py`);
- engenharia de features (`src/features/feature_engineering.py`);
- modelo de ML baseline com `sklearn` (`src/models/baseline.py`);
- rastreamento de experimentos com MLflow e versionamento de artefatos;
- API REST com FastAPI (`src/serving/app.py`);
- pipeline RAG e agente de suporte a perguntas (`src/agent/`).

## Arquitetura do projeto

### Principais módulos
- `src/scripts`: scripts de geração e processamento de dados.
- `src/features`: transformação e engenharia de atributos.
- `src/models`: treino do modelo, baseline e salvamento de champion.
- `src/serving`: aplicação FastAPI para inferência e agente.
- `src/agent`: componentes do agente ReAct, herramientas e pipeline RAG.
- `src/security`: guardrails de entrada e sanitização de saída.
- `src/monitoring`: métricas Prometheus.

### Função da pasta `src/`
Contém toda a lógica do projeto:
- ingestão e geração de dados;
- preprocessamento e transformação;
- treino e avaliação do modelo;
- API de inferência e agentes de consulta;
- segurança e métricas.

### Função da pasta `docs/`
Reúne documentação de suporte e compliance:
- `docs/SYSTEM_CARD.md`: visão geral do sistema, arquitetura e limitações.
- `docs/MODEL_CARD.md`: detalhes do modelo, métricas e recomendações.
- `docs/LGPD_PLAN.md`: plano de conformidade com LGPD.
- `docs/OWASP_MAPPING.md`: mapeamento de ameaças LLM/segurança.
- `docs/FRAUD_KNOWLEDGE_BASE.md`: base de conhecimento de fraude.
- `docs/DATASET.md`: descrição do dataset sintético.
- `docs/benchmark.md`: benchmark de configurações.

### Papel dos dados, datasets, modelos, API e pipelines
- `data/raw`: dados brutos gerados sinteticamente e metadados.
- `data/processed`: features processadas em formato parquet.
- `mlruns`: artefatos e histórico do MLflow.
- `src/models/fraud_detection`: modelo champion salvo localmente.
- API expõe endpoints de predição, saúde, métricas e agente.
- DVC controla as etapas de geração, processamento e treino.

## Estrutura de diretórios resumida

```text
.
├── Dockerfile
├── Makefile
├── docker-compose.yml
├── dvc.yaml
├── mlflow.db
├── pyproject.toml
├── requirements.txt
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── SYSTEM_CARD.md
│   ├── MODEL_CARD.md
│   ├── LGPD_PLAN.md
│   ├── OWASP_MAPPING.md
│   ├── FRAUD_KNOWLEDGE_BASE.md
│   ├── DATASET.md
│   └── benchmark.md
├── src/
│   ├── agent/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   ├── scripts/
│   ├── security/
│   └── serving/
└── tests/
```

### Pastas importantes
- `data/`: dados de entrada, saída e artefatos de pipeline.
- `docs/`: documentação técnica e regulatória.
- `src/`: código-fonte principal.
- `tests/`: testes automatizados.
- `mlruns/`: rastreamento de experimentos MLflow.
- `docker-compose.yml`: orquestração de `api`, `prometheus` e `grafana`.
- `Dockerfile`: imagem de container para a API.
- `dvc.yaml`: definição dos estágios DVC.

## Pré-requisitos

- Python 3.11
- Docker
- Docker Compose
- Make
- DVC
- MLflow
- GPU compatível CUDA/PyTorch para o pipeline LLM/Docker com `cu121` (se for utilizar inferência local de LLM)
- Ambiente virtual recomendado: `venv` ou similar

## Instalação local

1. Crie e ative ambiente virtual:

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows PowerShell
# ou
source .venv/bin/activate       # macOS/Linux
```

2. Atualize instaladores:

```bash
pip install --upgrade pip setuptools wheel
```

3. Instale todas as dependências com o Make:

```bash
make install-all
```

Esse comando instala o PyTorch `2.3.1+cu121` e os extras necessários ao projeto.

## Como gerar e processar os dados

### `make data`

Executa:

```bash
python -m src.scripts.generate_fraud_data
```

O script gera um dataset sintético em `data/raw/fraud_dataset.csv` e salva metadados em `data/raw/dataset_metadata.json`.

### `make process`

Executa:

```bash
python -m src.scripts.process_data
```

Esse comando lê `data/raw/fraud_dataset.csv`, aplica a transformação de features definida em `src/features/feature_engineering.py` e salva os dados processados em `data/processed/features.parquet`.

## Como treinar o modelo de IA

### `make train`

Executa o pipeline DVC:

```bash
make train
```

No projeto, isso chama:

```bash
dvc repro
```

O DVC reproduz os estágios definidos em `dvc.yaml`, incluindo geração de dados, processamento e treino.

### `make retrain`

Executa:

```bash
make retrain
```

Isso força a reprodução completa do pipeline com:

```bash
dvc repro -f
```

Use quando quiser forçar a reexecução, mesmo que o DVC considere as saídas atualizadas.

### `make dvc-status`

Verifica o estado do pipeline DVC:

```bash
make dvc-status
```

O comando mostra se há mudanças pendentes em arquivos/artefatos e se o pipeline está desatualizado.

### Observações sobre DVC
- `dvc.yaml` define os estágios `generate_data`, `process` e `train`.
- O DVC controla dependências e saídas para garantir reprodutibilidade.
- O estágio `train` roda `python -m src.models.train`.

## Como rodar a API localmente

### `make api`

Executa:

```bash
make api
```

Isso equivale a:

```bash
uvicorn src.serving.app:app --reload
```

Por padrão, a API deve ficar disponível em:

```text
http://127.0.0.1:8000
```

### Endpoints principais
- `GET /`: status básico
- `GET /health`: verificação de saúde
- `POST /predict`: predição de fraude
- `POST /agent`: consulta ao agente RAG
- `GET /metrics`: métricas Prometheus

### Exemplo de payload para `/predict`

```json
{
  "valor": 1234.56,
  "hora": 22,
  "dispositivo_novo": true,
  "tentativas_24h": 2,
  "distancia_km": 120.5
}
```

## Como rodar com Docker

### `make build`

Constói a imagem Docker:

```bash
make build
```

### `make up`

Inicia os containers em segundo plano:

```bash
make up
```

A configuração `docker-compose.yml` sobe os serviços:
- `api`: aplicação FastAPI
- `prometheus`: coleta de métricas
- `grafana`: dashboard

### `make docker-down`

Desliga os containers:

```bash
make docker-down
```

### `make docker-restart`

Reinicia o ambiente Docker:

```bash
make docker-restart
```

### `make docker-logs`

Acompanha os logs:

```bash
make docker-logs
```

### Notas Docker
- O `docker-compose.yml` usa recursos de GPU (`nvidia`), então verifique se o ambiente Docker tem suporte a NVIDIA Container Toolkit.
- Os volumes montam `./mlruns` e `./data` para persistência local.

## MLflow

### `make mlflow`

Executa:

```bash
make mlflow
```

Isso inicia a interface do MLflow para acompanhar experimentos e modelos.

### Papel do MLflow no projeto
- registra métricas e parâmetros de treino;
- salva o modelo e seus artefatos;
- permite consultar histórico de execuções.
- o tracking URI usado no código é `sqlite:///mlflow.db`.

## Testes e qualidade de código

### `make test`

Executa os testes com `pytest`:

```bash
make test
```

### `make lint`

Executa o lint com `ruff` e aplica correções automáticas:

```bash
make lint
```

## Dados e datasets

### Estrutura dos dados
- `data/raw/fraud_dataset.csv`: dataset sintético de transações.
- `data/raw/dataset_metadata.json`: metadados do dataset e hash MD5.
- `data/processed/features.parquet`: dados já transformados para treino.

### Como os datasets são gerados e consumidos
- `make data`: gera dados sintéticos com regras de fraude.
- `make process`: aplica engenheira de atributos e salva parquet.
- `make train`: usa o dataframe processado para treinar o modelo.

### Base de dados e features
- `valor`: transação, transformada com `log1p`
- `hora`: transformada em `hora_sin` e `hora_cos`
- `dispositivo_novo`: convertido para inteiro
- `distancia_km`: transformado com `log1p`
- `fraude`: target binário

Para detalhes, consulte `docs/DATASET.md`.

## Modelo de IA

### Tipo de modelo
- Regressão Logística (`sklearn.pipeline.Pipeline`) com `StandardScaler`.
- Modelo de classificação binária para detecção de fraude.

### Fluxo de treinamento
1. Carregar dados processados em `data/processed/features.parquet`.
2. Dividir em treino/teste estratificado em `src/models/baseline.py`.
3. Treinar o pipeline de regressão logística.
4. Logar parâmetros, métricas e modelo no MLflow.
5. Salvar o modelo champion em `src/models/fraud_detection`.

### Artefatos do modelo
- registro MLflow em `mlruns/`
- banco de dados `mlflow.db`
- modelo champion local em `src/models/fraud_detection`

Para mais informações, consulte `docs/MODEL_CARD.md` e `docs/SYSTEM_CARD.md`.

## Segurança, privacidade e conformidade

### Principais cuidados
- validação de entrada e sanitização de saída em `src/security/guardrails.py`;
- pipeline de agente com RAG e embeddeds em `src/agent/rag_pipeline.py`;
- métricas de API expostas em `src/monitoring/metrics.py`.

### LGPD
- O sistema usa dados sintéticos e não processa PII reais no código atual.
- A documentação de conformidade está em `docs/LGPD_PLAN.md`.
- As medidas abordam finalidade, minimização, segurança e auditoria.

### OWASP e segurança LLM
- O projeto contém mitigação contra prompt injection e sanitização de output.
- A análise de ameaças LLM está em `docs/OWASP_MAPPING.md`.
- O serviço ainda exige melhorias para produção, especialmente em rate limiting e hardening.

## Base de conhecimento de fraude

A documentação `docs/FRAUD_KNOWLEDGE_BASE.md` descreve os fatores de risco usados pelo agente e pelo modelo:
- valor alto
- horário incomum
- dispositivo novo
- tentativas recentes
- distância geográfica

O agente usa esse conhecimento para explicar decisões de fraude e auxiliar respostas híbridas.

## Benchmarks

O benchmark em `docs/benchmark.md` compara configurações de latência, qualidade e uso de memória para diferentes ajustes do pipeline RAG e LLM.

Principais pontos:
- configuração baseline é recomendada para equilíbrio entre qualidade e performance;
- configuração velocidade reduz latência às custas de qualidade;
- configuração qualidade melhora respostas com maior custo de recursos.

## Comandos úteis

| Comando | Descrição |
|---|---|
| `make data` | Gera dataset sintético em `data/raw` |
| `make process` | Processa dados brutos e salva features em `data/processed` |
| `make train` | Roda `dvc repro` para reproduzir o pipeline de treino |
| `make retrain` | Roda `dvc repro -f` para forçar a reprodução do pipeline |
| `make dvc-status` | Mostra status do DVC |
| `make test` | Executa testes com `pytest` |
| `make lint` | Checa e corrige código com `ruff` |
| `make api` | Inicia a API local com Uvicorn |
| `make freeze` | Atualiza `requirements.txt` com dependências instaladas |
| `make install-all` | Instala dependências do projeto e PyTorch CUDA |
| `make mlflow` | Inicia MLflow UI |
| `make build` | Constrói imagem Docker |
| `make up` | Sobe containers Docker em background |
| `make docker-down` | Derruba containers Docker |
| `make docker-restart` | Reinicia o ambiente Docker |
| `make docker-logs` | Exibe logs dos containers |

## Troubleshooting

### Docker não sobe
- Verifique se Docker Compose está instalado.
- Confirme se o runtime NVIDIA está disponível para GPU.
- Veja logs com `make docker-logs`.
- Se houver erro de build, execute `docker-compose build --no-cache`.

### Porta ocupada
- A API usa `8000`; Prometheus usa `9090`; Grafana usa `3000`.
- Libere a porta ou altere no `docker-compose.yml`.

### Erro de dependências
- Ative o ambiente virtual correto.
- Reinstale com `make install-all`.
- Se necessário, use `pip install -r requirements.txt`.

### Erro com CUDA/PyTorch
- Confirme a versão compatível: `torch==2.3.1+cu121`.
- No Docker, verifique se o host tem NVIDIA Container Toolkit configurado.
- Em máquinas sem GPU, o container pode não funcionar corretamente.

### Pipeline DVC desatualizado
- Rode `make train` para atualizar as saídas.
- Use `make retrain` para forçar reexecução.
- Consulte `dvc status` para identificar arquivos modificados.

### MLflow não abre
- Verifique se `mlflow` está instalado no ambiente.
- Execute `make mlflow` e abra `http://127.0.0.1:5000`.
- Confirme se `mlflow.db` está presente na raiz.

### API não responde
- Verifique logs do servidor.
- Confirme que o modelo está salvo em `src/models/fraud_detection`.
- No caso de `make api`, confirme se o `uvicorn` iniciou sem erro.
- Use `GET /health` para checar disponibilidade.

## Referências internas consultadas

- `docs/SYSTEM_CARD.md`
- `docs/OWASP_MAPPING.md`
- `docs/MODEL_CARD.md`
- `docs/LGPD_PLAN.md`
- `docs/FRAUD_KNOWLEDGE_BASE.md`
- `docs/DATASET.md`
- `docs/benchmark.md`
