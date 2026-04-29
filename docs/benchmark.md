# Benchmark de Configurações — Sistema de Detecção de Fraude

## Introdução

Este documento apresenta o benchmark comparativo de diferentes configurações do sistema de detecção de fraude. Foram testadas 3 configurações principais variando parâmetros do RAG, LLM e agente para avaliar impacto na performance.

## Configurações Testadas

### Configuração 1: Baseline
- **LLM**: Qwen2.5-0.5B-Instruct-AWQ (padrão)
- **RAG**: chunk_size=800, chunk_overlap=100, top_k=3
- **Agente**: Temperatura 0.0, max_tokens=256
- **Objetivo**: Configuração conservadora, foco em consistência

### Configuração 2: Otimizada para Velocidade
- **LLM**: Mesmo modelo, mas max_tokens=128
- **RAG**: chunk_size=400, chunk_overlap=50, top_k=2
- **Agente**: Temperatura 0.0, max_tokens=128
- **Objetivo**: Reduzir latência mantendo qualidade

### Configuração 3: Otimizada para Qualidade
- **LLM**: Mesmo modelo, temperatura=0.1
- **RAG**: chunk_size=1200, chunk_overlap=200, top_k=5
- **Agente**: Temperatura 0.1, max_tokens=512
- **Objetivo**: Melhorar qualidade de resposta com mais contexto

## Métricas Avaliadas

### Métricas Técnicas
- **Latência**: Tempo médio de resposta (ms)
- **Throughput**: Requests por segundo
- **Memória**: Uso de RAM (GB)

### Métricas de Qualidade (RAGAS)
- **Context Availability**: % de queries com contexto retornado
- **Context Similarity**: Similaridade semântica entre query+esperado e contexto
- **Answer Similarity**: Similaridade entre resposta esperada e gerada

### Métricas de Judge (LLM-as-Judge)
- **Correctness**: Correção da resposta (0-10)
- **Clarity**: Clareza da resposta (0-10)
- **Explanation**: Qualidade da explicação (0-10)
- **Business Alignment**: Alinhamento com contexto de negócio (0-10)

## Resultados

### Performance Técnica

| Configuração | Latência (ms) | Throughput (req/s) | Memória (GB) |
|-------------|---------------|-------------------|--------------|
| Baseline    | 3200         | 0.3              | 2.1         |
| Velocidade  | 1800         | 0.5              | 1.8         |
| Qualidade   | 4500         | 0.2              | 2.5         |

### Qualidade RAGAS (Média sobre 20 queries)

| Configuração | Context Availability | Context Similarity | Answer Similarity |
|-------------|---------------------|-------------------|------------------|
| Baseline    | 0.95               | 0.72             | 0.68           |
| Velocidade  | 0.90               | 0.65             | 0.62           |
| Qualidade   | 0.98               | 0.78             | 0.74           |

### Qualidade Judge (Média sobre 20 queries)

| Configuração | Correctness | Clarity | Explanation | Business Alignment |
|-------------|-------------|---------|-------------|-------------------|
| Baseline    | 8.2        | 7.8    | 7.5        | 7.2              |
| Velocidade  | 7.8        | 7.5    | 7.0        | 6.8              |
| Qualidade   | 8.5        | 8.2    | 8.0        | 7.8              |

## Análise

### Trade-offs Identificados
- **Velocidade vs Qualidade**: Configuração otimizada para velocidade reduz latência em ~44% mas diminui qualidade em ~10-15%
- **Qualidade vs Recursos**: Configuração de qualidade melhora métricas em ~10% mas aumenta uso de memória em ~19% e latência em ~40%
- **Baseline**: Equilíbrio razoável entre performance e qualidade

### Recomendações
- **Produção**: Usar configuração Baseline como padrão
- **Desenvolvimento**: Configuração Velocidade para testes rápidos
- **Pesquisa**: Configuração Qualidade para experimentos

## Metodologia de Teste

### Ambiente
- Hardware: CPU Intel i7, GPU RTX 3060 12GB
- Software: Python 3.11, PyTorch 2.3.1+cu121
- Dataset: Golden set com 20 queries de teste

### Procedimento
1. Carregamento completo do sistema por configuração
2. Execução sequencial de todas as 20 queries
3. Medição de latência e recursos por query
4. Cálculo de métricas RAGAS e Judge
5. Agregação de resultados

### Limitações
- Testes em ambiente controlado, não produção
- Queries fixas, não representam distribuição real
- Métricas LLM-as-Judge têm variabilidade inerente

## Conclusão

O benchmark demonstra que pequenas variações nos parâmetros podem impactar significativamente performance e qualidade. A configuração Baseline oferece o melhor equilíbrio para uso em produção, com latência aceitável e qualidade adequada para detecção de fraude.

---