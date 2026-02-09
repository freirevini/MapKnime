# MapKnime — KNIME Workflow Analyzer & AI Transpiler

Ferramenta CLI para analisar workflows KNIME (`.knwf`) e transpilá-los para Python usando Vertex AI Gemini 2.5 Pro.

## Estrutura do Projeto

```
2ChatKnime/
├── knime_parser.py              # Parser XML → JSON (workflow.knime + settings.xml)
├── MapKnime/
│   ├── __init__.py              # Package exports
│   ├── __main__.py              # Entry point: python -m MapKnime
│   ├── run_analysis.py          # CLI unificado (extração → parsing → mappers)
│   ├── temporal_mapper.py       # Mapeador de padrões temporais (datas, timestamps)
│   ├── loop_mapper.py           # Mapeador de estruturas de loop
│   ├── logic_mapper.py          # Mapeador de lógica (regras, expressões, snippets)
│   ├── avaliacao_IA.py          # Transpiler IA (Vertex AI Gemini 2.5 Pro)
│   └── config.yaml              # Credenciais Vertex AI
└── docs/
    └── PLAN-knime-workflow-analysis.md
```

## Pré-requisitos

- Python 3.10+
- Dependências:

```bash
pip install pyyaml google-cloud-aiplatform
```

## Uso

### 1. Análise Completa (Extraction + Parsing + Mappers)

```bash
# Uso básico — gera outputs no mesmo diretório do .knwf
python -m MapKnime <workflow.knwf>

# Especificar diretório de saída
python -m MapKnime <workflow.knwf> --output-dir ./resultado

# Pular geração de HTML (mais rápido)
python -m MapKnime <workflow.knwf> --skip-html

# Manter arquivos temporários da extração (debug)
python -m MapKnime <workflow.knwf> --keep-extract
```

### 2. Somente Mappers (usando JSON existente)

```bash
python -m MapKnime --json-only <KNIME_WORKFLOW_ANALYSIS.json>
```

### 3. Transpilação com IA (Vertex AI Gemini 2.5 Pro)

**Configuração** — preencha `MapKnime/config.yaml`:

```yaml
vertex_ai:
  project_id: "seu-projeto-gcp"
  region: "us-central1"
  model: "gemini-2.5-pro"
```

> **Nota**: As credenciais de banco de dados são geradas automaticamente como
> placeholders no arquivo Python transpilado (`fluxo_transpilado.py`).
> O usuário preenche diretamente no código gerado.

**Execução**:

```bash
# Standalone — a partir de um diretório com os JSONs de análise
python -m MapKnime.avaliacao_IA <diretorio_com_jsons>

# Config customizado
python -m MapKnime.avaliacao_IA <diretorio> --config meu_config.yaml

# Saída customizada
python -m MapKnime.avaliacao_IA <diretorio> --output resultado.py

# Pipeline completo (análise + transpilação)
python -m MapKnime <workflow.knwf> --transpile
```

## Outputs Gerados

### Análise (steps 1-5)

| Arquivo | Descrição |
|---------|-----------|
| `KNIME_WORKFLOW_ANALYSIS.json` | Estrutura completa do workflow (nodes, conexões, settings) |
| `KNIME_WORKFLOW_ANALYSIS.md` | Relatório legível em Markdown |
| `KNIME_WORKFLOW_ANALYSIS.html` | Visualização interativa no navegador |
| `temporal_map.json` | Padrões temporais (datas, timestamps, variáveis) |
| `loop_map.json` | Estruturas de loop (pares, standalone, tipos) |
| `logic_map.json` | Lógica (Rule Engine, expressões, code snippets) |
| `analysis_summary.txt` | Resumo consolidado |

### Transpilação IA (step 6)

| Arquivo | Descrição |
|---------|-----------|
| `fluxo_transpilado.py` | Código Python executável gerado pela IA |
| `transpilation_report.md` | Relatório com status por nó, tokens usados, warnings |
| `transpilation.log` | Log detalhado de toda operação |

## Pipeline

```
┌─────────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────┐
│  .knwf file │───▶│ Extract  │───▶│ Parse XML    │───▶│ 3 Mappers │
│  (ZIP)      │    │ (Step 1) │    │ (Step 2)     │    │ (3/4/5)   │
└─────────────┘    └──────────┘    └──────────────┘    └─────┬─────┘
                                                             │
                                                             ▼
                                                   ┌─────────────────┐
                                                   │ AI Transpiler   │
                                                   │ Gemini 2.5 Pro  │
                                                   │ (Step 6)        │
                                                   └────────┬────────┘
                                                            │
                                              ┌─────────────▼──────────────┐
                                              │  fluxo_transpilado.py      │
                                              │  transpilation_report.md   │
                                              └────────────────────────────┘
```

## Autenticação Vertex AI

O projeto utiliza [Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/provide-credentials-adc):

```bash
# Opção 1: Login pessoal
gcloud auth application-default login

# Opção 2: Service Account
export GOOGLE_APPLICATION_CREDENTIALS="/caminho/para/service-account.json"
```

## Exemplo Completo

```bash
# 1. Análise básica
python -m MapKnime fluxo_knime_exemplo.knwf -o ./output

# 2. Verificar resultados
cat ./output/analysis_summary.txt

# 3. Transpilação IA (requer config.yaml preenchido)
python -m MapKnime.avaliacao_IA ./output

# 4. Verificar código gerado
python -c "import ast; ast.parse(open('./output/fluxo_transpilado.py').read()); print('OK')"
```
