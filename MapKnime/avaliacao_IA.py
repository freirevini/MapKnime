"""
avaliacao_IA.py — KNIME-to-Python AI Transpiler

Reads MapKnime analysis outputs and generates executable Python code
using Vertex AI Gemini 2.5 Pro.

Usage:
    python -m MapKnime.avaliacao_IA <analysis_dir>
    python -m MapKnime.avaliacao_IA <analysis_dir> --config config.yaml
    python -m MapKnime.avaliacao_IA <analysis_dir> --output resultado.py

The <analysis_dir> must contain:
    - KNIME_WORKFLOW_ANALYSIS.json
    - temporal_map.json
    - loop_map.json
    - logic_map.json
"""

import argparse
import ast
import json
import logging
import os
import re
import sys
import time
import textwrap
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "[%(asctime)s] %(levelname)-7s %(message)s"
DATE_FORMAT = "%H:%M:%S"

logger = logging.getLogger("MapKnime.transpiler")


def setup_logging(log_path: str, verbose: bool = False):
    """Configure file + console logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    fh = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(ch)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VertexConfig:
    project_id: str = ""
    region: str = "us-central1"
    model: str = "gemini-2.5-pro"


@dataclass
class OutputConfig:
    max_line_length: int = 120
    include_comments: bool = True
    include_type_hints: bool = True


@dataclass
class AppConfig:
    vertex: VertexConfig = field(default_factory=VertexConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: str) -> AppConfig:
    """Load and validate config.yaml."""
    cfg = AppConfig()

    if not os.path.exists(config_path):
        logger.warning("Config não encontrado: %s — usando padrões", config_path)
        return cfg

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    v = raw.get("vertex_ai", {})
    cfg.vertex = VertexConfig(
        project_id=v.get("project_id", ""),
        region=v.get("region", "us-central1"),
        model=v.get("model", "gemini-2.5-pro"),
    )



    o = raw.get("output", {})
    cfg.output = OutputConfig(
        max_line_length=int(o.get("max_line_length", 120)),
        include_comments=bool(o.get("include_comments", True)),
        include_type_hints=bool(o.get("include_type_hints", True)),
    )

    return cfg


# ---------------------------------------------------------------------------
# Analysis Loader
# ---------------------------------------------------------------------------

REQUIRED_FILES = [
    "KNIME_WORKFLOW_ANALYSIS.json",
    "temporal_map.json",
    "loop_map.json",
    "logic_map.json",
]


@dataclass
class AnalysisBundle:
    """Unified container for all analysis outputs."""
    workflow: dict = field(default_factory=dict)
    temporal: dict = field(default_factory=dict)
    loops: dict = field(default_factory=dict)
    logic: dict = field(default_factory=dict)
    source_dir: str = ""
    total_chars: int = 0
    estimated_tokens: int = 0


def load_analysis(analysis_dir: str) -> AnalysisBundle:
    """Load all 4 JSON files and compute token estimates."""
    bundle = AnalysisBundle(source_dir=analysis_dir)

    for fname in REQUIRED_FILES:
        fpath = os.path.join(analysis_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {fpath}")

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if fname == "KNIME_WORKFLOW_ANALYSIS.json":
            bundle.workflow = data
        elif fname == "temporal_map.json":
            bundle.temporal = data
        elif fname == "loop_map.json":
            bundle.loops = data
        elif fname == "logic_map.json":
            bundle.logic = data

    # Estimate total tokens (~4 chars per token)
    total = sum(
        os.path.getsize(os.path.join(analysis_dir, f))
        for f in REQUIRED_FILES
    )
    bundle.total_chars = total
    bundle.estimated_tokens = total // 4

    logger.info("Análise carregada: %s chars (~%s tokens)",
                f"{total:,}", f"{bundle.estimated_tokens:,}")

    return bundle


def count_nodes(wf: dict) -> int:
    """Count total nodes recursively including MetaNodes."""
    total = len(wf.get("nodes", {}))
    for n in wf.get("nodes", {}).values():
        if n.get("sub_workflow"):
            total += count_nodes(n["sub_workflow"])
    return total


# ---------------------------------------------------------------------------
# Chunking Engine
# ---------------------------------------------------------------------------

TOKEN_THRESHOLD = 800_000  # Safety margin below Gemini's 1M


@dataclass
class Chunk:
    """A segment of the workflow for individual AI processing."""
    index: int
    total: int
    nodes: dict
    connections: list
    temporal_nodes: list = field(default_factory=list)
    loop_nodes: list = field(default_factory=list)
    logic_nodes: list = field(default_factory=list)
    context_from_previous: list = field(default_factory=list)
    estimated_tokens: int = 0


def build_chunks(bundle: AnalysisBundle) -> list[Chunk]:
    """Split workflow into chunks if it exceeds token threshold."""

    if bundle.estimated_tokens < TOKEN_THRESHOLD:
        logger.info("Workflow cabe em um único prompt (%s < %s tokens)",
                     f"{bundle.estimated_tokens:,}", f"{TOKEN_THRESHOLD:,}")
        return [Chunk(
            index=0,
            total=1,
            nodes=bundle.workflow.get("nodes", {}),
            connections=bundle.workflow.get("connections", []),
            temporal_nodes=bundle.temporal.get("nodes", []),
            loop_nodes=_extract_loop_nodes(bundle.loops),
            logic_nodes=bundle.logic.get("nodes", []),
            estimated_tokens=bundle.estimated_tokens,
        )]

    logger.info("Workflow excede threshold — ativando chunking por MetaNode")
    return _split_by_metanode(bundle)


def _extract_loop_nodes(loops: dict) -> list:
    """Flatten loop pairs and standalone into a node list."""
    nodes = []
    for pair in loops.get("loop_pairs", []):
        nodes.append(pair)
    for standalone in loops.get("standalone_loops", []):
        nodes.append(standalone)
    return nodes


def _split_by_metanode(bundle: AnalysisBundle) -> list[Chunk]:
    """Split the workflow at MetaNode boundaries."""
    wf = bundle.workflow
    all_nodes = wf.get("nodes", {})
    connections = wf.get("connections", [])

    # Separate root nodes from MetaNode children
    root_nodes = {}
    meta_groups = []

    for nid, node in all_nodes.items():
        if node.get("is_meta") and node.get("sub_workflow"):
            meta_groups.append((nid, node))
        else:
            root_nodes[nid] = node

    # Build chunks
    chunks = []
    total = 1 + len(meta_groups)

    # Chunk 0: root non-meta nodes
    root_conns = [
        c for c in connections
        if str(c["source_id"]) in root_nodes or str(c["dest_id"]) in root_nodes
    ]
    chunks.append(Chunk(
        index=0, total=total,
        nodes=root_nodes, connections=root_conns,
        temporal_nodes=[], loop_nodes=[], logic_nodes=[],
        estimated_tokens=bundle.estimated_tokens // total,
    ))

    # Chunk 1..N: one per MetaNode
    context_vars = []
    for i, (nid, meta) in enumerate(meta_groups, start=1):
        sub_wf = meta["sub_workflow"]
        sub_nodes = sub_wf.get("nodes", {})
        sub_conns = sub_wf.get("connections", [])
        meta_name = meta.get("settings", {}).get("node_name", f"MetaNode_{nid}")

        chunks.append(Chunk(
            index=i, total=total,
            nodes=sub_nodes, connections=sub_conns,
            temporal_nodes=[], loop_nodes=[], logic_nodes=[],
            context_from_previous=context_vars.copy(),
            estimated_tokens=bundle.estimated_tokens // total,
        ))
        context_vars.append(meta_name)

    logger.info("Workflow dividido em %d chunks", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert KNIME-to-Python transpiler. Your task is to convert
    a KNIME workflow into a single, executable Python script.

    RULES:
    1. Output ONLY valid Python code inside a single ```python code block.
    2. Use pandas for data manipulation, numpy for numeric ops, sqlalchemy for DB.
    3. Preserve the exact execution order from the workflow DAG.
    4. Each KNIME node becomes a clearly labeled step with a comment showing
       the original node name and ID.
    5. Use functions to group MetaNode logic (one function per MetaNode).
    6. Include proper error handling with try/except per major step.
    7. Add a main() function that orchestrates the full pipeline.
    8. Include all necessary import statements at the top.
    9. Use logging (not print) for status messages.
    10. For database connections, use sqlalchemy with configurable parameters.

    DATABASE CREDENTIALS:
    For EACH database connection found in the workflow, generate a clearly
    labeled configuration section at the top of the file where the user can
    fill in their credentials. Use this pattern:

        # ═══════════════════════════════════════════════════════════
        # DATABASE CREDENTIALS — Fill in before running
        # ═══════════════════════════════════════════════════════════
        DB_CONFIG = {
            "driver": "postgresql",   # postgresql | mysql | mssql | oracle
            "host": "",               # <-- INSERT DB HOST
            "port": 5432,             # <-- INSERT DB PORT
            "user": "",               # <-- INSERT DB USERNAME
            "password": "",           # <-- INSERT DB PASSWORD
            "database": "",           # <-- INSERT DB NAME
        }

    If the workflow uses MULTIPLE database connections, generate
    separate config dicts for each (e.g. DB_CONFIG_SOURCE, DB_CONFIG_TARGET).

    IMPORTANT:
    - If a KNIME node has no direct Python equivalent, implement the closest
      approximation and add a comment: # APPROXIMATION: <explanation>
    - For flow variables, use Python variables with descriptive names.
    - For temporal operations, use pd.to_datetime() and timedelta.
    - For Rule Engine nodes, translate rules to np.where() or pd.eval().
    - For loops, use standard Python for/while with pandas operations.
    - Generate COMPLETE code — no placeholders like "..." or "pass".
""")


def build_prompt(
    chunk: Chunk, bundle: AnalysisBundle, config: AppConfig
) -> str:
    """Build the full prompt for Vertex AI."""

    parts = []

    # --- Section 1: Workflow Structure ---
    parts.append("## WORKFLOW STRUCTURE\n")
    parts.append("Execution order: " + json.dumps(
        bundle.workflow.get("execution_order", [])
    ))
    parts.append("\n\n### Nodes\n")
    parts.append(json.dumps(chunk.nodes, ensure_ascii=False, indent=1))
    parts.append("\n\n### Connections\n")
    parts.append(json.dumps(chunk.connections, ensure_ascii=False, indent=1))

    # --- Section 2: Temporal Enrichment ---
    if chunk.temporal_nodes or bundle.temporal.get("nodes"):
        parts.append("\n\n## TEMPORAL PATTERNS\n")
        temporal_data = chunk.temporal_nodes or bundle.temporal.get("nodes", [])
        parts.append(json.dumps(temporal_data, ensure_ascii=False, indent=1))

    # --- Section 3: Loop Enrichment ---
    if chunk.loop_nodes or bundle.loops.get("loop_pairs"):
        parts.append("\n\n## LOOP STRUCTURES\n")
        loop_data = chunk.loop_nodes or _extract_loop_nodes(bundle.loops)
        parts.append(json.dumps(loop_data, ensure_ascii=False, indent=1))

    # --- Section 4: Logic Enrichment ---
    if chunk.logic_nodes or bundle.logic.get("nodes"):
        parts.append("\n\n## LOGIC / RULES / EXPRESSIONS\n")
        logic_data = chunk.logic_nodes or bundle.logic.get("nodes", [])
        parts.append(json.dumps(logic_data, ensure_ascii=False, indent=1))

    # --- Section 5: Previous chunk context ---
    if chunk.context_from_previous:
        parts.append("\n\n## CONTEXT FROM PREVIOUS CHUNKS\n")
        parts.append("The following functions/variables are already defined "
                      "from previous chunk processing:\n")
        for var in chunk.context_from_previous:
            parts.append(f"  - {var}()\n")

    # --- Section 6: Chunk info ---
    if chunk.total > 1:
        parts.append(f"\n\n## CHUNK INFO\n")
        parts.append(f"This is chunk {chunk.index + 1} of {chunk.total}.\n")
        if chunk.index == 0:
            parts.append("Generate: imports, DB config, utility functions, "
                          "and root-level node processing.\n")
        elif chunk.index == chunk.total - 1:
            parts.append("Generate: this MetaNode's function + the main() "
                          "orchestrator that calls all functions.\n")
        else:
            parts.append("Generate: this MetaNode's function only.\n")

    return "".join(parts)


def build_system_prompt(config: AppConfig) -> str:
    """Return the system prompt (no dynamic formatting needed)."""
    return SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Vertex AI Caller
# ---------------------------------------------------------------------------

@dataclass
class AIResponse:
    """Container for a single Vertex AI response."""
    code: str = ""
    raw_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0
    attempt: int = 1


def call_vertex(
    system_prompt: str,
    user_prompt: str,
    config: VertexConfig,
    max_retries: int = 3,
) -> AIResponse:
    """Call Vertex AI Gemini and extract Python code from response."""

    import vertexai
    from vertexai.generative_models import GenerativeModel, Part

    vertexai.init(project=config.project_id, location=config.region)
    model = GenerativeModel(
        config.model,
        system_instruction=[system_prompt],
    )

    generation_config = {
        "max_output_tokens": 65536,
        "temperature": 0.1,
        "top_p": 0.95,
    }

    for attempt in range(1, max_retries + 1):
        logger.info("Vertex AI call — tentativa %d/%d", attempt, max_retries)
        t0 = time.time()

        try:
            response = model.generate_content(
                [user_prompt],
                generation_config=generation_config,
            )

            elapsed = time.time() - t0
            raw_text = response.text

            # Extract token usage
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0)
            output_tokens = getattr(usage, "candidates_token_count", 0)

            logger.info(
                "Resposta recebida: %d tokens in, %d tokens out, %.1fs",
                input_tokens, output_tokens, elapsed,
            )

            code = _extract_python_code(raw_text)

            return AIResponse(
                code=code,
                raw_text=raw_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_s=elapsed,
                attempt=attempt,
            )

        except Exception as e:
            elapsed = time.time() - t0
            logger.error("Tentativa %d falhou (%.1fs): %s", attempt, elapsed, e)
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.info("Aguardando %ds antes de retry...", wait)
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Vertex AI falhou após {max_retries} tentativas: {e}"
                ) from e

    # Unreachable but satisfies type checker
    return AIResponse()


def _extract_python_code(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    # Try to find ```python ... ``` blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # If multiple blocks, join them (chunking scenario)
        return "\n\n".join(m.strip() for m in matches)

    # Fallback: try generic code blocks
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(m.strip() for m in matches)

    # Last resort: return raw text (might be code without fences)
    logger.warning("Nenhum bloco de código encontrado — usando texto bruto")
    return text.strip()


# ---------------------------------------------------------------------------
# Syntax Validator
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    is_valid: bool = False
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    corrected_code: str = ""
    correction_attempts: int = 0


def validate_syntax(
    code: str,
    config: VertexConfig,
    system_prompt: str,
    max_corrections: int = 2,
) -> ValidationResult:
    """Validate Python syntax and attempt self-correction via AI."""
    result = ValidationResult()

    for attempt in range(max_corrections + 1):
        try:
            ast.parse(code)
            result.is_valid = True
            result.corrected_code = code
            result.correction_attempts = attempt
            if attempt == 0:
                logger.info("✅ Sintaxe Python válida (primeira tentativa)")
            else:
                logger.info("✅ Sintaxe corrigida após %d tentativa(s)", attempt)
            break

        except SyntaxError as e:
            error_msg = f"Linha {e.lineno}: {e.msg}"
            result.errors.append(error_msg)
            logger.warning("❌ Erro de sintaxe: %s", error_msg)

            if attempt < max_corrections:
                logger.info("Enviando erro ao Gemini para auto-correção...")
                code = _request_correction(code, e, config, system_prompt)
            else:
                logger.error("Sintaxe inválida após %d correções", max_corrections)
                result.corrected_code = code

    # Static analysis warnings
    result.warnings = _static_analysis(result.corrected_code)
    if result.warnings:
        logger.info("⚠️  %d warnings de análise estática", len(result.warnings))

    return result


def _request_correction(
    code: str, error: SyntaxError, config: VertexConfig, system_prompt: str
) -> str:
    """Send syntax error to Gemini for auto-correction."""
    correction_prompt = textwrap.dedent(f"""\
        The following Python code has a syntax error:

        ERROR: Line {error.lineno}: {error.msg}
        {f'Near: {error.text.strip()}' if error.text else ''}

        Fix ONLY the syntax error. Return the COMPLETE corrected code
        inside a ```python code block. Do not change any logic.

        ```python
        {code}
        ```
    """)

    try:
        response = call_vertex(system_prompt, correction_prompt, config, max_retries=1)
        return response.code
    except Exception as e:
        logger.error("Auto-correção falhou: %s", e)
        return code


def _static_analysis(code: str) -> list[str]:
    """Basic static analysis for common issues."""
    warnings = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["Código com erro de sintaxe — análise estática não executada"]

    # Check for undefined main()
    func_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "main" not in func_names:
        warnings.append("Função main() não encontrada no código gerado")

    # Check for missing common imports
    import_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                import_names.add(node.module.split(".")[0])

    required = {"pandas", "logging"}
    missing = required - import_names
    for m in missing:
        warnings.append(f"Import '{m}' esperado mas não encontrado")

    return warnings


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

def generate_report(
    bundle: AnalysisBundle,
    responses: list[AIResponse],
    validation: ValidationResult,
    output_path: str,
    config: AppConfig,
    elapsed: float,
) -> str:
    """Generate transpilation_report.md."""
    total_nodes = count_nodes(bundle.workflow)

    total_input = sum(r.input_tokens for r in responses)
    total_output = sum(r.output_tokens for r in responses)
    total_ai_time = sum(r.duration_s for r in responses)

    # Count generated code lines
    code = validation.corrected_code or responses[-1].code if responses else ""
    code_lines = len(code.splitlines())

    lines = [
        "# 📋 Transpilation Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Duration**: {elapsed:.1f}s (AI: {total_ai_time:.1f}s)",
        "",
        "---",
        "",
        "## Workflow Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Nodes | {total_nodes} |",
        f"| Temporal Nodes | {bundle.temporal.get('temporal_nodes_count', 0)} |",
        f"| Loop Nodes | {bundle.loops.get('total_loop_nodes', 0)} |",
        f"| Logic Nodes | {bundle.logic.get('logic_nodes_count', 0)} |",
        f"| MetaNodes | {sum(1 for n in bundle.workflow.get('nodes', {}).values() if n.get('is_meta'))} |",
        "",
        "## AI Processing",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Model | {config.vertex.model} |",
        f"| Chunks | {len(responses)} |",
        f"| Input Tokens | {total_input:,} |",
        f"| Output Tokens | {total_output:,} |",
        f"| AI Time | {total_ai_time:.1f}s |",
        f"| Attempts (total) | {sum(r.attempt for r in responses)} |",
        "",
        "## Validation",
        "",
        f"| Check | Result |",
        f"|-------|--------|",
        f"| Syntax Valid | {'✅ Yes' if validation.is_valid else '❌ No'} |",
        f"| Correction Attempts | {validation.correction_attempts} |",
        f"| Static Warnings | {len(validation.warnings)} |",
        "",
    ]

    if validation.errors:
        lines.append("### Errors")
        lines.append("")
        for err in validation.errors:
            lines.append(f"- ❌ {err}")
        lines.append("")

    if validation.warnings:
        lines.append("### Warnings")
        lines.append("")
        for warn in validation.warnings:
            lines.append(f"- ⚠️  {warn}")
        lines.append("")

    lines.extend([
        "## Output",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Output File | `{os.path.basename(output_path)}` |",
        f"| Code Lines | {code_lines:,} |",
        f"| File Size | {len(code):,} chars |",
        "",
        "---",
        "",
        f"*Generated by MapKnime AI Transpiler*",
    ])

    report_path = os.path.join(
        os.path.dirname(output_path),
        "transpilation_report.md",
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Relatório salvo: %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def transpile(
    analysis_dir: str,
    config_path: str = "",
    output_path: str = "",
) -> str:
    """Full transpilation pipeline. Returns path to generated .py file."""

    start_time = time.time()

    # Resolve paths
    analysis_dir = os.path.abspath(analysis_dir)
    if not config_path:
        config_path = os.path.join(
            Path(__file__).resolve().parent, "config.yaml"
        )
    if not output_path:
        output_path = os.path.join(analysis_dir, "fluxo_transpilado.py")

    # Setup logging
    log_path = os.path.join(analysis_dir, "transpilation.log")
    setup_logging(log_path)

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        MapKnime — AI Transpiler (Gemini 2.5 Pro)       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Step 1: Load config
    logger.info("[1/6] Carregando configuração...")
    config = load_config(config_path)

    if not config.vertex.project_id:
        logger.error("project_id não configurado em %s", config_path)
        raise ValueError(
            f"Preencha o project_id no arquivo de configuração: {config_path}"
        )

    # Step 2: Load analysis
    logger.info("[2/6] Carregando análise de %s...", analysis_dir)
    bundle = load_analysis(analysis_dir)
    total_nodes = count_nodes(bundle.workflow)
    logger.info("      %d nodes total, ~%s tokens",
                total_nodes, f"{bundle.estimated_tokens:,}")

    # Step 3: Build chunks
    logger.info("[3/6] Preparando chunks...")
    chunks = build_chunks(bundle)
    logger.info("      %d chunk(s) criado(s)", len(chunks))

    # Step 4: Generate code via Vertex AI
    logger.info("[4/6] Gerando código Python via Vertex AI...")
    system_prompt = build_system_prompt(config)
    responses: list[AIResponse] = []
    code_parts: list[str] = []

    for chunk in chunks:
        user_prompt = build_prompt(chunk, bundle, config)

        if len(chunks) > 1:
            logger.info("  Processando chunk %d/%d...", chunk.index + 1, chunk.total)

        resp = call_vertex(system_prompt, user_prompt, config.vertex)
        responses.append(resp)
        code_parts.append(resp.code)

    # Merge code from all chunks
    if len(code_parts) == 1:
        generated_code = code_parts[0]
    else:
        generated_code = _merge_chunks(code_parts)

    logger.info("      Código gerado: %d linhas", len(generated_code.splitlines()))

    # Step 5: Validate syntax
    logger.info("[5/6] Validando sintaxe...")
    validation = validate_syntax(
        generated_code, config.vertex, system_prompt
    )
    final_code = validation.corrected_code or generated_code

    # Save generated code
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_code)
    logger.info("      Código salvo: %s", output_path)

    # Step 6: Generate report
    elapsed = time.time() - start_time
    logger.info("[6/6] Gerando relatório...")
    report_path = generate_report(
        bundle, responses, validation, output_path, config, elapsed,
    )

    # Final summary
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                TRANSPILATION COMPLETE                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"  Nodes:      {total_nodes}")
    print(f"  Code:       {len(final_code.splitlines())} linhas")
    print(f"  Syntax:     {'✅ Válida' if validation.is_valid else '❌ Inválida'}")
    print(f"  Warnings:   {len(validation.warnings)}")
    print(f"  Duration:   {elapsed:.1f}s")
    print(f"  Output:     {output_path}")
    print(f"  Report:     {report_path}")
    print(f"  Log:        {log_path}")
    print()

    return output_path


def _merge_chunks(parts: list[str]) -> str:
    """Merge multiple code chunks into a single file."""
    # First chunk: keep everything (imports, config, etc.)
    # Subsequent chunks: skip duplicate imports
    if not parts:
        return ""

    merged = [parts[0]]
    import_lines = set()

    # Extract imports from first chunk
    for line in parts[0].splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            import_lines.add(stripped)

    # Add subsequent chunks, skipping duplicate imports
    for part in parts[1:]:
        filtered = []
        for line in part.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and stripped in import_lines:
                continue
            filtered.append(line)
        merged.append("\n".join(filtered))

    return "\n\n".join(merged)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="MapKnime.avaliacao_IA",
        description="Transpile KNIME workflow analysis to Python using Vertex AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m MapKnime.avaliacao_IA ./analysis_output
              python -m MapKnime.avaliacao_IA ./analysis_output --config my_config.yaml
              python -m MapKnime.avaliacao_IA ./analysis_output --output resultado.py
        """),
    )
    parser.add_argument(
        "analysis_dir",
        help="Directory containing the MapKnime analysis JSONs",
    )
    parser.add_argument(
        "--config", "-c",
        default="",
        help="Path to config.yaml (default: MapKnime/config.yaml)",
    )
    parser.add_argument(
        "--output", "-o",
        default="",
        help="Output Python file path (default: <analysis_dir>/fluxo_transpilado.py)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()

    try:
        transpile(
            analysis_dir=args.analysis_dir,
            config_path=args.config,
            output_path=args.output,
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
