# Projeto Traduz

**Modernização do Legado KNIME — Tradução Automatizada para Python**

Área responsável: Tecnologia / Controles Internos
Data: Fevereiro/2026
Classificação: Interno — Uso Executivo

---

## 1. Resumo Executivo

A área mantém atualmente **~500 workflows** desenvolvidos na plataforma KNIME, responsáveis por indicadores operacionais executados mensal e diariamente. Esses processos encontram-se **defasados**, com execução manual unitária e sem integração com a esteira moderna de dados da organização.

O **Projeto Traduz** propõe a conversão automatizada desses workflows para **Python**, utilizando Inteligência Artificial generativa para traduzir a lógica de negócio embarcada em cada fluxo KNIME. A solução já em desenvolvimento realiza a leitura estruturada do workflow, interpreta as conexões e regras, e gera código Python funcional — reduzindo de **horas para minutos** o tempo de conversão por processo.

Com uma equipe de **2 pessoas** e prazo alvo de **junho/2026**, o projeto elimina a dependência de uma plataforma legada, moderniza a base de indicadores e viabiliza a automação futura de sua execução.

**Decisão solicitada:** Aprovação para execução do projeto até junho/2026.

---

## 2. Problema e Oportunidade de Negócio

### Situação atual

| Aspecto | Estado |
|---------|--------|
| Plataforma | KNIME (legado, sem evolução) |
| Volume | ~500 workflows de indicadores |
| Execução | Manual, um por um, sem automação |
| Manutenção | Trabalhosa — interface gráfica dificulta versionamento e auditoria |
| Integração | Limitada — não conversa nativamente com a esteira de dados moderna |
| Documentação | Inexistente ou incompleta na maioria dos fluxos |

### Riscos de não agir

- **Obsolescência crescente**: workflows cada vez mais difíceis de manter e adaptar a mudanças regulatórias.
- **Dependência de conhecimento tácito**: regras de negócio embarcadas nos fluxos sem documentação formal.
- **Ineficiência operacional**: execução manual consome tempo de analistas em tarefas repetitivas.
- **Barreira à automação**: enquanto os processos estiverem em KNIME, a área não consegue integrá-los a pipelines automatizados.

### Oportunidade

Converter a base legada para Python permite:

- Integrar com orquestradores e pipelines modernos.
- Versionar código com controle de mudanças (Git).
- Viabilizar execuções agendadas e automatizadas.
- Tornar a lógica de negócio auditável e documentada.

---

## 3. Solução Proposta

### Visão geral

O Projeto Traduz é uma **ferramenta de tradução automatizada** que converte workflows KNIME em scripts Python utilizando IA generativa (Google Gemini).

### Como funciona

```
┌────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  Workflow   │ ──▸ │  Mapeamento  │ ──▸ │  Tradução IA   │ ──▸ │ Script Python│
│   KNIME     │     │ Estruturado  │     │  (Gemini)      │     │  Funcional   │
└────────────┘     └──────────────┘     └────────────────┘     └──────────────┘
```

1. **Mapeamento**: leitura automatizada do workflow KNIME, extraindo nodes, conexões, regras de negócio, parâmetros de banco de dados e fluxo de execução.
2. **Tradução via IA**: a estrutura mapeada é enviada ao modelo de linguagem com instruções especializadas para gerar código Python idiomático, com tratamento de erros, logging e documentação.
3. **Validação automática**: o código gerado passa por validação de sintaxe e, se necessário, auto-correção.
4. **Entrega**: script Python pronto, com relatório técnico e log de execução para rastreabilidade.

### Diferencial

- **Não é uma conversão genérica** — a ferramenta foi calibrada para o ambiente específico da organização: bancos Sybase, SQL Server, Oracle, BigQuery e padrões internos de conexão.
- **Revisão mínima** — o analista precisa apenas inserir suas credenciais de acesso aos bancos; a lógica de negócio já vem traduzida.

---

## 4. Benefícios e Indicadores Esperados

### Benefícios diretos

| Benefício | Impacto |
|-----------|---------|
| **Velocidade de conversão** | De horas/dias (manual) para minutos (automatizado) |
| **Eliminação do legado KNIME** | 500 workflows migrados para linguagem moderna |
| **Padronização** | Código uniforme, documentado e versionável |
| **Autonomia da equipe** | Python é amplamente dominado; KNIME depende de especialistas |
| **Base para automação** | Scripts Python podem ser agendados e orquestrados |

### KPIs do projeto

| Indicador | Meta |
|-----------|------|
| Workflows convertidos | 500 até junho/2026 |
| Taxa de conversão automatizada (sem reescrita manual) | ≥ 85% |
| Tempo médio de conversão por workflow | ≤ 15 minutos |
| Workflows em produção (executando via Python) | 100% até junho/2026 |
| Redução de esforço manual na manutenção | ≥ 60% |

---

## 5. Riscos e Mitigação

| # | Risco | Probabilidade | Impacto | Mitigação |
|---|-------|:---:|:---:|-----------|
| 1 | Workflows com lógica muito complexa que a IA não traduz com fidelidade | Média | Alto | Revisão humana nos casos críticos; ajuste iterativo das instruções da IA |
| 2 | Conectividade com bancos legados (Sybase, Oracle) exige configurações específicas | Baixa | Médio | Arquivo de referência de conexões já mapeado e integrado à ferramenta |
| 3 | Prazo insuficiente para 500 workflows com 2 pessoas | Média | Alto | Priorização por criticidade; workflows simples em lote, complexos individualizados |
| 4 | Mudanças nos workflows KNIME durante a migração | Baixa | Médio | Congelamento de alterações em workflows já em fila de conversão |
| 5 | Indisponibilidade do serviço de IA (Vertex AI / Google Cloud) | Baixa | Baixo | Retry automático e possibilidade de execução em lote fora do horário comercial |

---

## 6. Cronograma Macro

```
FEV/2026          MAR              ABR              MAI              JUN/2026
───┬──────────────┬────────────────┬────────────────┬────────────────┬───
   │              │                │                │                │
   ▼              ▼                ▼                ▼                ▼
 FASE 1        FASE 2           FASE 3           FASE 4           FASE 5
 Ferramenta    Piloto           Conversão        Conversão        Validação
 & Calibração  (50 workflows)   em escala        em escala        & Encerramento
                                (200 wf)         (+250 wf)
```

| Fase | Período | Entrega |
|------|---------|---------|
| **1 — Ferramenta e Calibração** | Fev/2026 | Ferramenta de tradução calibrada e testada |
| **2 — Piloto** | Mar/2026 | 50 workflows convertidos e validados |
| **3 — Escala (lote 1)** | Abr/2026 | +200 workflows convertidos |
| **4 — Escala (lote 2)** | Mai/2026 | +250 workflows convertidos |
| **5 — Validação e Encerramento** | Jun/2026 | 100% migrado, relatório final, desativação KNIME |

### Marcos de decisão

- **Final de março**: avaliação do piloto — go/no-go para escala.
- **Final de maio**: 90% convertido — decisão sobre desativação do KNIME.
- **Junho**: entrega final e encerramento formal.

---

## 7. Investimentos Necessários

| Item | Tipo | Observação |
|------|------|------------|
| **Equipe dedicada** | 2 profissionais | Alocação parcial ou integral conforme fase |
| **Google Cloud (Vertex AI)** | Consumo sob demanda | Custo proporcional ao volume traduzido |
| **Infraestrutura adicional** | Nenhuma | Utiliza ambiente já existente |
| **Licenças** | Nenhuma | Python e bibliotecas são open source |

> **Nota:** O projeto opera com investimento incremental baixo — os maiores custos são de alocação de equipe, sem necessidade de contratações ou aquisições.

---

## 8. Próximos Passos e Decisão Esperada

### Solicitação à diretoria

> **Aprovar a execução do Projeto Traduz** com equipe de 2 pessoas e prazo de entrega até junho/2026, para conversão automatizada dos ~500 workflows KNIME para Python.

### Próximos passos imediatos (pós-aprovação)

| # | Ação | Responsável | Prazo |
|---|------|-------------|-------|
| 1 | Inventariar e classificar os 500 workflows por criticidade | Equipe do projeto | 2 semanas |
| 2 | Finalizar calibração da ferramenta de tradução | Equipe do projeto | Fev/2026 |
| 3 | Executar piloto com 50 workflows prioritários | Equipe do projeto | Mar/2026 |
| 4 | Apresentar resultados do piloto à superintendência | Equipe do projeto | Final Mar/2026 |

---

*Documento elaborado pela equipe do Projeto Traduz — Fevereiro/2026*
