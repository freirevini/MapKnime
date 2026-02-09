"""
run_analysis.py — Unified KNIME Workflow Analyzer CLI

End-to-end pipeline: .knwf → extract → parse → map (temporal + loop + logic)

Usage:
    python -m MapKnime.run_analysis <workflow.knwf> [--output-dir <dir>]

Example:
    python -m MapKnime.run_analysis fluxo_knime_exemplo.knwf
    python -m MapKnime.run_analysis fluxo_knime_exemplo.knwf --output-dir ./resultado

Output (in output dir):
    KNIME_WORKFLOW_ANALYSIS.json   — Full workflow structure
    KNIME_WORKFLOW_ANALYSIS.md     — Human-readable report
    KNIME_WORKFLOW_ANALYSIS.html   — Interactive HTML visualization
    temporal_map.json              — Temporal pattern analysis
    loop_map.json                  — Loop structure analysis
    logic_map.json                 — Logic/rules/expressions analysis
    analysis_summary.txt           — Consolidated summary
"""

import argparse
import json
import os
import sys
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

# Ensure parent dir is on path for knime_parser import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import knime_parser
from temporal_mapper import scan_workflow as scan_temporal
from loop_mapper import scan_workflow as scan_loops
from logic_mapper import scan_workflow as scan_logic


# =============================================================================
# STEP 1 — EXTRACT .knwf (ZIP)
# =============================================================================

def extract_knwf(knwf_path: str, extract_dir: str) -> str:
    """Extract .knwf file (ZIP format) and return path to workflow.knime.

    Extracts to a short temp directory to avoid Windows MAX_PATH limits.
    """
    print(f"[1/5] Extraindo {Path(knwf_path).name}...")

    if not zipfile.is_zipfile(knwf_path):
        raise ValueError(f"Arquivo não é um ZIP válido: {knwf_path}")

    with zipfile.ZipFile(knwf_path, "r") as zf:
        zf.extractall(extract_dir)

    # Find the first (root) workflow.knime
    for root, dirs, files in os.walk(extract_dir):
        if "workflow.knime" in files:
            wf_path = os.path.join(root, "workflow.knime")
            print(f"      workflow.knime encontrado")
            return wf_path

    raise FileNotFoundError(
        f"workflow.knime não encontrado dentro de {knwf_path}"
    )


# =============================================================================
# STEP 2 — PARSE WORKFLOW XML → JSON
# =============================================================================

def parse_workflow(workflow_knime_path: str, output_dir: str) -> str:
    """Parse workflow.knime and generate JSON/MD/HTML. Returns JSON path."""
    print("[2/5] Parseando workflow.knime (recursivo)...")

    base_dir = os.path.dirname(workflow_knime_path)
    workflow = knime_parser.parse_workflow_knime(workflow_knime_path, base_dir)

    # Count stats
    def count_all(wf):
        total = len(wf.get("nodes", {}))
        for n in wf.get("nodes", {}).values():
            if n.get("sub_workflow"):
                total += count_all(n["sub_workflow"])
        return total

    total = count_all(workflow)
    root_nodes = len(workflow["nodes"])
    root_conn = len(workflow["connections"])

    print(f"      {total} nodes total | {root_nodes} raiz | {root_conn} conexões")

    # Generate outputs
    json_path = os.path.join(output_dir, "KNIME_WORKFLOW_ANALYSIS.json")
    md_path = os.path.join(output_dir, "KNIME_WORKFLOW_ANALYSIS.md")
    html_path = os.path.join(output_dir, "KNIME_WORKFLOW_ANALYSIS.html")

    knime_parser.generate_json(workflow, json_path)
    knime_parser.generate_markdown(workflow, md_path)
    knime_parser.generate_html(workflow, html_path)

    print(f"      JSON: {json_path}")
    print(f"      MD:   {md_path}")
    print(f"      HTML: {html_path}")

    return json_path


# =============================================================================
# STEP 3-5 — RUN MAPPERS
# =============================================================================

def run_temporal_mapper(json_path: str, output_dir: str):
    """Run temporal pattern mapper."""
    print("[3/5] Mapeando padrões temporais...")
    tmap = scan_temporal(json_path)
    out = os.path.join(output_dir, "temporal_map.json")
    tmap.to_json(out)
    print(f"      {tmap.temporal_nodes_count} nodes temporais "
          f"({tmap.confirmed_count} confirmados)")
    return tmap


def run_loop_mapper(json_path: str, output_dir: str):
    """Run loop structure mapper."""
    print("[4/5] Mapeando estruturas de loop...")
    lmap = scan_loops(json_path)
    out = os.path.join(output_dir, "loop_map.json")
    lmap.to_json(out)
    pairs = len(lmap.loop_pairs)
    standalone = len(lmap.standalone_loops)
    print(f"      {lmap.total_loop_nodes} nodes de loop | "
          f"{pairs} pares | {standalone} standalone")
    return lmap


def run_logic_mapper(json_path: str, output_dir: str):
    """Run logic/rules/expression mapper."""
    print("[5/5] Mapeando lógica (regras, expressões, snippets)...")
    lgmap = scan_logic(json_path)
    out = os.path.join(output_dir, "logic_map.json")
    lgmap.to_json(out)
    print(f"      {lgmap.logic_nodes_count} nodes de lógica | "
          f"{lgmap.rule_engine_count} regras | "
          f"{lgmap.expression_count} expressões | "
          f"{lgmap.snippet_count} snippets")
    return lgmap


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_summary(
    json_path: str, tmap, lmap, lgmap, output_dir: str, elapsed: float
):
    """Generate consolidated analysis_summary.txt."""
    with open(json_path, "r", encoding="utf-8") as f:
        wf = json.load(f)

    def count_all(wf_data):
        total = len(wf_data.get("nodes", {}))
        for n in wf_data.get("nodes", {}).values():
            if n.get("sub_workflow"):
                total += count_all(n["sub_workflow"])
        return total

    total_nodes = count_all(wf)

    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║              KNIME WORKFLOW ANALYSIS — COMPLETE             ║
╚══════════════════════════════════════════════════════════════╝

  Workflow:    {wf.get('name', 'Unknown')}
  Analyzed:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Duration:    {elapsed:.1f}s
  Total Nodes: {total_nodes}

┌──────────────────────────────────────────────────────────────┐
│ TEMPORAL MAPPER                                              │
├──────────────────────────────────────────────────────────────┤
│  Temporal nodes:  {tmap.temporal_nodes_count:<6} ({tmap.confirmed_count} confirmed, {tmap.probable_count} probable)│
│  Variable chains: {len(tmap.variable_chains):<6}                                        │
│  Generators:      {len(tmap.generators):<6}                                        │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ LOOP MAPPER                                                  │
├──────────────────────────────────────────────────────────────┤
│  Loop nodes:      {lmap.total_loop_nodes:<6}                                        │
│  Loop pairs:      {len(lmap.loop_pairs):<6}                                        │
│  Standalone:      {len(lmap.standalone_loops):<6}                                        │
│  Types: {', '.join(lmap.by_type.keys()):<52}│
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ LOGIC MAPPER                                                 │
├──────────────────────────────────────────────────────────────┤
│  Logic nodes:     {lgmap.logic_nodes_count:<6}                                        │
│  Rule Engine:     {lgmap.rule_engine_count:<6}                                        │
│  Expressions:     {lgmap.expression_count:<6}                                        │
│  Code Snippets:   {lgmap.snippet_count:<6}                                        │
│  Unique columns:  {len(lgmap.all_columns):<6}                                        │
└──────────────────────────────────────────────────────────────┘

  Output Directory: {output_dir}
  Files:
    - KNIME_WORKFLOW_ANALYSIS.json  (full workflow structure)
    - KNIME_WORKFLOW_ANALYSIS.md    (human-readable report)
    - KNIME_WORKFLOW_ANALYSIS.html  (interactive visualization)
    - temporal_map.json             (temporal analysis)
    - loop_map.json                 (loop analysis)
    - logic_map.json                (logic analysis)
"""
    summary_path = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    return summary


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="MapKnime",
        description="Analyze KNIME workflows (.knwf) for AI transpilation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m MapKnime.run_analysis workflow.knwf
  python -m MapKnime.run_analysis workflow.knwf --output-dir ./results
  python -m MapKnime.run_analysis workflow.knwf --skip-html
  python -m MapKnime.run_analysis workflow.knwf --transpile
        """,
    )
    parser.add_argument(
        "knwf_file",
        nargs="?",
        default=None,
        help="Path to .knwf file (KNIME workflow archive)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: same as .knwf file location)",
    )
    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="Skip HTML report generation (faster)",
    )
    parser.add_argument(
        "--keep-extract",
        action="store_true",
        help="Keep extracted temp files (for debugging)",
    )
    parser.add_argument(
        "--json-only",
        default=None,
        help="Skip extraction, use existing JSON file directly",
    )
    parser.add_argument(
        "--transpile",
        action="store_true",
        help="Also transpile to Python via Vertex AI Gemini (requires config.yaml)",
    )

    args = parser.parse_args()
    start_time = time.time()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           MapKnime — KNIME Workflow Analyzer            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Resolve paths
    if args.json_only:
        # Skip extraction, use existing JSON
        json_path = os.path.abspath(args.json_only)
        if not os.path.exists(json_path):
            print(f"[ERROR] JSON não encontrado: {json_path}")
            sys.exit(1)
        output_dir = args.output_dir or os.path.dirname(json_path)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[1/5] SKIP — usando JSON existente: {json_path}")
        print(f"[2/5] SKIP — JSON já parseado")
    else:
        if not args.knwf_file:
            parser.error("Informe o arquivo .knwf ou use --json-only <path>")
        knwf_path = os.path.abspath(args.knwf_file)
        if not os.path.exists(knwf_path):
            print(f"[ERROR] Arquivo .knwf não encontrado: {knwf_path}")
            sys.exit(1)

        output_dir = args.output_dir or os.path.dirname(knwf_path)
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Extract to short temp path (avoids MAX_PATH)
        extract_dir = tempfile.mkdtemp(prefix="knime_")
        workflow_knime_path = extract_knwf(knwf_path, extract_dir)

        # Step 2: Parse
        json_path = parse_workflow(workflow_knime_path, output_dir)

        # Cleanup extraction if not requested to keep
        if not args.keep_extract:
            shutil.rmtree(extract_dir, ignore_errors=True)

    # Step 3: Temporal Mapper
    tmap = run_temporal_mapper(json_path, output_dir)

    # Step 4: Loop Mapper
    lmap = run_loop_mapper(json_path, output_dir)

    # Step 5: Logic Mapper
    lgmap = run_logic_mapper(json_path, output_dir)

    # Step 6 (optional): AI Transpilation
    if args.transpile:
        from .avaliacao_IA import transpile
        print()
        transpile(analysis_dir=output_dir)

    # Summary
    elapsed = time.time() - start_time
    summary = generate_summary(json_path, tmap, lmap, lgmap, output_dir, elapsed)
    print(summary)


if __name__ == "__main__":
    main()
