# Cartão do Sistema — Sistema de Detecção de Fraudes

## Visão Geral

Este Cartão do Sistema fornece uma visão abrangente do Sistema de Detecção de Fraudes, incluindo seu uso pretendido, limitações, considerações éticas e contexto de implantação. O sistema integra aprendizado de máquina tradicional, Geração Aumentada por Recuperação (RAG) e um agente ReAct para detectar transações financeiras fraudulentas.

## Uso Pretendido

### Casos de Uso Principais
- Detecção de fraudes em tempo real para transações financeiras online
- Avaliação de risco e suporte à decisão para processadores de pagamento
- Sinalização automática de atividades suspeitas em sistemas bancários
- Integração com fluxos de trabalho existentes de prevenção a fraudes

### Usuários-Alvo
- Instituições financeiras e processadores de pagamento
- Equipes de gestão de risco
- Oficiais de conformidade
- Cientistas de dados e engenheiros de ML

### Usos Fora do Escopo
- Tomada de decisão direta sem supervisão humana
- Uso em domínios não financeiros
- Negociação em tempo real ou aplicações de alta frequência

## Arquitetura do Sistema

### Componentes
1. **Pipeline de Dados**: Geração de dados sintéticos e engenharia de atributos gerenciadas pelo DVC
2. **Modelo de ML**: Regressão Logística como baseline com rastreamento via MLflow e registro de modelos
3. **Pipeline RAG**: Embeddings com SentenceTransformer e busca semântica sobre documentação
4. **Agente**: Agente baseado em ReAct com ferramentas personalizadas para predição, explicação e recuperação de conhecimento
5. **Camada de API**: Endpoints FastAPI para predição, interação com o agente e métricas
6. **Monitoramento**: Métricas com Prometheus, detecção de drift e dashboards no Grafana
7. **Segurança**: Guardrails de entrada e validação básica de saída

### Fluxo de Dados
1. Dados brutos de transação → Engenharia de atributos → Predição do modelo de ML
2. Consulta do usuário → Classificação pelo agente → Execução de ferramenta → Geração de resposta
3. Métricas do sistema → Prometheus → Visualização no Grafana

## Características de Desempenho

### Precisão e Confiabilidade
- Modelo de ML: ROC AUC ~0,85 em dados sintéticos de teste
- Recuperação RAG: Disponibilidade de contexto ~90%, similaridade semântica ~0,75
- Agente: Correção ~8,5/10, Clareza ~8,0/10, Alinhamento com o negócio ~7,5/10 (com base em avaliação por LLM)

### Latência
- Endpoint de predição: <100ms por requisição
- Resposta do agente: <5s por consulta (incluindo geração por LLM)
- Recuperação RAG: <500ms por consulta

### Escalabilidade
- Projetado para implantação em contêineres (Docker)
- Suporta escalabilidade horizontal via Kubernetes
- Uso de memória: ~2GB para modelo + embeddings + LLM

## Limitações e Vieses

### Limitações Técnicas
- Treinado exclusivamente com dados sintéticos; desempenho em dados reais não validado
- Limitado a 5 atributos; pode não capturar padrões complexos de fraude
- Respostas do LLM dependem da qualidade do prompt e da disponibilidade de contexto
- Sem retreinamento em tempo real ou aprendizado online

### Vieses de Dados
- Dados sintéticos podem não capturar todos os padrões reais de fraude
- Possível viés na engenharia de atributos (ex.: transformações logarítmicas podem favorecer certos intervalos de transação)
- Falta de diversidade demográfica ou geográfica na geração sintética

### Limitações Operacionais
- Requer intervenção manual para atualizações do modelo
- Detecção de drift é básica (teste KS); pode não identificar mudanças sutis
- Sem alertas automáticos ou resposta a incidentes

## Considerações Éticas

### Equidade
- O sistema pode sinalizar desproporcionalmente certos tipos de transação
- Possibilidade de falsos positivos afetando usuários legítimos
- Sem restrições explícitas de equidade no treinamento

### Privacidade
- Processa dados de transações; não há tratamento de PII na implementação atual
- Geração de dados sintéticos evita exposição de dados reais
- Consulte LGPD_PLAN.md para detalhes de conformidade com privacidade

### Segurança
- Guardrails previnem injeção de prompt, mas podem não capturar todos os inputs adversariais
- Respostas do agente não são garantidamente seguras ou apropriadas
- Sem moderação de conteúdo para texto gerado

## Implantação e Manutenção

### Implantação
- Containerizado via Docker Compose
- Requer GPU para inferência do LLM (suporte CUDA)
- Variáveis de ambiente para configuração
- Verificações de saúde e desligamento gracioso implementados

### Monitoramento e Manutenção
- Métricas do Prometheus para monitoramento operacional
- MLflow para versionamento de modelos e experimentos
- Relatórios de detecção de drift em formato JSON
- Retreinamento regular recomendado (mensalmente)

### Resposta a Incidentes
- Logs disponíveis via logging da aplicação
- Dashboard de métricas para monitoramento em tempo real
- Intervenção manual necessária para atualizações do modelo
- Procedimentos de backup via DVC e MLflow

## Recomendações

### Para Usuários
- Sempre combinar com revisão humana para decisões de alto risco
- Monitorar métricas de desempenho regularmente
- Validar com dados reais antes da implantação em produção

### Para Desenvolvedores
- Implementar guardrails mais robustos e testes adversariais
- Adicionar pipelines automatizados de retreinamento
- Melhorar monitoramento com regras de alerta
- Conduzir auditorias completas de equidade e viés