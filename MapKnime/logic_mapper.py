"""
logic_mapper.py — Logic Pattern Mapper for KNIME Workflow JSON.

Extracts, classifies, and maps all embedded logic within a KNIME
workflow: Rule Engine rules, JEP/String expressions, and Java/Python
code snippets. Produces a structured `logic_map.json` with:
  - Parsed rules with column references and outcomes
  - Expressions with identified columns and functions
  - Code snippets with input/output variable mappings
  - Python equivalence suggestions

Usage:
    python logic_mapper.py [workflow.json]
    Default: KNIME_WORKFLOW_ANALYSIS.json
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════
# ENUMS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════

class LogicType(str, Enum):
    RULE_ENGINE = "RULE_ENGINE"
    RULE_FILTER = "RULE_FILTER"
    JEP_EXPRESSION = "JEP_EXPRESSION"
    STRING_MANIPULATION = "STRING_MANIPULATION"
    JAVA_SNIPPET = "JAVA_SNIPPET"
    JAVA_EDIT_VAR = "JAVA_EDIT_VAR"
    PYTHON_SCRIPT = "PYTHON_SCRIPT"


class Complexity(str, Enum):
    SIMPLE = "SIMPLE"          # 1-2 rules/lines
    MODERATE = "MODERATE"      # 3-10 rules/lines
    COMPLEX = "COMPLEX"        # 11+ rules/lines or nested logic


# Factory → LogicType mapping
LOGIC_FACTORY_MAP: dict[str, LogicType] = {
    "RuleEngineNodeFactory": LogicType.RULE_ENGINE,
    "RuleEngineFilterNodeFactory": LogicType.RULE_FILTER,
    "JEPNodeFactory": LogicType.JEP_EXPRESSION,
    "StringManipulationNodeFactory": LogicType.STRING_MANIPULATION,
    "JavaSnippetNodeFactory": LogicType.JAVA_SNIPPET,
    "JavaEditVarNodeFactory": LogicType.JAVA_EDIT_VAR,
    "Python2ScriptNodeFactory2": LogicType.PYTHON_SCRIPT,
    "PythonScriptNodeFactory": LogicType.PYTHON_SCRIPT,
    "Python3ScriptNodeFactory": LogicType.PYTHON_SCRIPT,
}

# Python equivalence templates
PYTHON_EQUIV: dict[LogicType, str] = {
    LogicType.RULE_ENGINE: "np.select(conditions, choices, default)",
    LogicType.RULE_FILTER: "df[condition]  # or df.query()",
    LogicType.JEP_EXPRESSION: "df['col'] = expression  # pandas vectorized",
    LogicType.STRING_MANIPULATION: "df['col'] = df['col'].str.method()",
    LogicType.JAVA_SNIPPET: "# Translate Java logic to Python",
    LogicType.JAVA_EDIT_VAR: "# Translate Java variable logic to Python",
    LogicType.PYTHON_SCRIPT: "# Adapt Python 2 → Python 3 if needed",
}

# KNIME column reference pattern: $column_name$
COL_REF_PATTERN = re.compile(r'\$([A-Za-z_][A-Za-z0-9_]*)\$')

# KNIME flow variable pattern: $${Svar}$$ or $${Ivar}$$
FLOW_VAR_PATTERN = re.compile(r'\$\$\{([SID])([^}]+)\}\$\$')

# JEP function pattern
JEP_FUNC_PATTERN = re.compile(r'([a-zA-Z_]\w*)\s*\(')

# KNIME Rule Engine operators
RULE_OPERATORS = {'>', '<', '>=', '<=', '=', '!=', 'AND', 'OR', 'NOT',
                  'LIKE', 'IN', 'MATCHES', 'MISSING', 'TRUE', 'FALSE'}


# ═══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ParsedRule:
    """A single parsed KNIME rule."""
    raw: str
    condition: str
    outcome: str
    columns_used: list = field(default_factory=list)
    operators: list = field(default_factory=list)
    is_default: bool = False


@dataclass
class LogicNode:
    """A node containing embedded logic."""
    ref: str
    node_id: str
    factory_short: str
    logic_type: str
    folder: str
    annotation: str
    level: int
    parent_metanode: str
    complexity: str
    python_hint: str
    # Rule Engine fields
    rules: list = field(default_factory=list)
    rule_count: int = 0
    output_column: str = ""
    replaces_column: str = ""
    is_filter: bool = False
    # Expression fields
    expression: str = ""
    expression_columns: list = field(default_factory=list)
    expression_functions: list = field(default_factory=list)
    appends_column: bool = False
    target_column: str = ""
    return_type: str = ""
    # Code Snippet fields
    code: str = ""
    code_language: str = ""
    input_columns: list = field(default_factory=list)
    output_columns: list = field(default_factory=list)
    input_variables: list = field(default_factory=list)
    output_variables: list = field(default_factory=list)
    imports: list = field(default_factory=list)
    # Common
    all_columns_referenced: list = field(default_factory=list)
    flow_variables_used: list = field(default_factory=list)


@dataclass
class LogicMap:
    """Complete logic mapping for a workflow."""
    workflow_name: str
    scan_timestamp: str
    total_nodes: int
    logic_nodes_count: int
    rule_engine_count: int
    expression_count: int
    snippet_count: int
    nodes: list = field(default_factory=list)
    by_type: dict = field(default_factory=dict)
    all_columns: list = field(default_factory=list)
    all_flow_variables: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False, default=str)

    def to_summary(self) -> str:
        lines = [
            f"=== Logic Map: {self.workflow_name} ===",
            f"Scan: {self.scan_timestamp}",
            f"Total nodes: {self.total_nodes}",
            f"Logic nodes: {self.logic_nodes_count}",
            f"  Rule Engine: {self.rule_engine_count}",
            f"  Expressions (JEP+String): {self.expression_count}",
            f"  Code Snippets (Java+Python): {self.snippet_count}",
            "",
            "By Type:",
        ]
        for lt, count in sorted(self.by_type.items()):
            hint = PYTHON_EQUIV.get(LogicType(lt), "")
            lines.append(f"  {lt}: {count} nodes")
            if hint:
                lines.append(f"    Python: {hint}")

        lines.append("")
        lines.append(f"Unique columns referenced: {len(self.all_columns)}")
        if self.all_flow_variables:
            lines.append(f"Flow variables used: {self.all_flow_variables}")

        lines.append("")
        lines.append("Nodes:")
        for node in self.nodes:
            detail = ""
            if node["logic_type"] in (LogicType.RULE_ENGINE.value, LogicType.RULE_FILTER.value):
                detail = f"{node['rule_count']} rules"
                if node["output_column"]:
                    detail += f" -> {node['output_column']}"
            elif node["logic_type"] in (LogicType.JEP_EXPRESSION.value, LogicType.STRING_MANIPULATION.value):
                expr_preview = node["expression"][:60] + "..." if len(node["expression"]) > 60 else node["expression"]
                detail = f"expr: {expr_preview}"
            elif node["logic_type"] in (LogicType.JAVA_SNIPPET.value, LogicType.JAVA_EDIT_VAR.value, LogicType.PYTHON_SCRIPT.value):
                detail = f"{node['code_language']}, {len(node['code'])} chars"

            lines.append(f"  [{node['logic_type']}] {node['ref']}: {node['folder']} ({detail}) [{node['complexity']}]")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# RULE ENGINE PARSING
# ═══════════════════════════════════════════════════════════════════

def parse_rules(rules_text: str) -> list[dict]:
    """Parse KNIME Rule Engine syntax into structured rules."""
    if not rules_text:
        return []

    parsed = []
    for line in rules_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("//"):
            continue

        # Split at =>
        parts = line.split("=>", 1)
        if len(parts) != 2:
            continue

        condition = parts[0].strip()
        outcome = parts[1].strip()

        # Extract column references
        columns = COL_REF_PATTERN.findall(condition)

        # Extract operators
        operators = [op for op in RULE_OPERATORS if f" {op} " in f" {condition} "]

        is_default = condition.upper() == "TRUE"

        parsed.append(asdict(ParsedRule(
            raw=line,
            condition=condition,
            outcome=outcome,
            columns_used=columns,
            operators=operators,
            is_default=is_default,
        )))

    return parsed


def _assess_complexity(rule_count: int = 0, expr_len: int = 0, code_len: int = 0) -> str:
    """Assess logic complexity."""
    if rule_count > 10 or expr_len > 200 or code_len > 500:
        return Complexity.COMPLEX.value
    if rule_count > 2 or expr_len > 50 or code_len > 100:
        return Complexity.MODERATE.value
    return Complexity.SIMPLE.value


# ═══════════════════════════════════════════════════════════════════
# EXPRESSION PARSING
# ═══════════════════════════════════════════════════════════════════

def parse_expression(expr: str) -> tuple[list[str], list[str], list[str]]:
    """Parse expressions to extract columns, functions, and flow variables."""
    columns = list(set(COL_REF_PATTERN.findall(expr)))
    functions = list(set(JEP_FUNC_PATTERN.findall(expr)))
    flow_vars = [(prefix, name) for prefix, name in FLOW_VAR_PATTERN.findall(expr)]

    # Filter out known keywords from functions
    noise = {'if', 'else', 'true', 'false', 'null', 'missing', 'not',
             'and', 'or', 'TRUE', 'FALSE', 'MISSING', 'NaN'}
    functions = [f for f in functions if f.lower() not in {n.lower() for n in noise}]

    flow_var_names = [f"{p}{n}" for p, n in flow_vars]
    return columns, functions, flow_var_names


# ═══════════════════════════════════════════════════════════════════
# CODE SNIPPET PARSING
# ═══════════════════════════════════════════════════════════════════

def parse_java_snippet(model: dict) -> dict:
    """Extract Java snippet details from model."""
    code = model.get("scriptBody", "")
    imports = model.get("scriptImports", "")

    # Parse input/output columns
    in_cols = []
    out_cols = []
    raw_in = model.get("inCols", {})
    raw_out = model.get("outCols", {})

    if isinstance(raw_in, dict):
        for key, val in raw_in.items():
            if isinstance(val, dict):
                in_cols.append(val.get("colName", val.get("name", key)))
            elif isinstance(val, str):
                in_cols.append(val)
    elif isinstance(raw_in, list):
        for item in raw_in:
            if isinstance(item, dict):
                in_cols.append(item.get("colName", item.get("name", "")))

    if isinstance(raw_out, dict):
        for key, val in raw_out.items():
            if isinstance(val, dict):
                out_cols.append(val.get("colName", val.get("name", key)))
            elif isinstance(val, str):
                out_cols.append(val)
    elif isinstance(raw_out, list):
        for item in raw_out:
            if isinstance(item, dict):
                out_cols.append(item.get("colName", item.get("name", "")))

    # Parse input/output variables
    in_vars = []
    out_vars = []
    raw_iv = model.get("inVars", {})
    raw_ov = model.get("outVars", {})

    if isinstance(raw_iv, dict):
        for key, val in raw_iv.items():
            if isinstance(val, dict):
                in_vars.append(val.get("name", key))
    if isinstance(raw_ov, dict):
        for key, val in raw_ov.items():
            if isinstance(val, dict):
                out_vars.append(val.get("name", key))

    return {
        "code": code,
        "imports": imports if isinstance(imports, str) else str(imports),
        "in_cols": in_cols,
        "out_cols": out_cols,
        "in_vars": in_vars,
        "out_vars": out_vars,
    }


def parse_python_script(model: dict) -> dict:
    """Extract Python script details from model."""
    code = model.get("sourceCode", "")
    return {
        "code": code,
        "python_version": "2" if "python2" in model.get("pythonVersionOption", "python2").lower() else "3",
        "chunk_size": model.get("chunkSize", -1),
    }


# ═══════════════════════════════════════════════════════════════════
# SCANNING ENGINE
# ═══════════════════════════════════════════════════════════════════

def _get_factory_short(factory: str) -> str:
    return factory.split(".")[-1] if factory else ""


def _scan_nodes_recursive(
    nodes: dict,
    prefix: str = "",
    level: int = 0,
    parent_meta: str = "",
) -> list[LogicNode]:
    """Recursively scan all nodes for logic factories."""
    found: list[LogicNode] = []

    for nid, node in nodes.items():
        settings = node.get("settings", {})
        factory = settings.get("factory", "")
        factory_short = _get_factory_short(factory)
        model = settings.get("model", {}) or {}
        annotation = str(settings.get("annotation", "") or "").strip()[:200]
        ref = f"{prefix}{nid}" if prefix else nid

        logic_type = LOGIC_FACTORY_MAP.get(factory_short)

        if logic_type is not None:
            logic_node = _build_logic_node(
                ref=ref,
                node_id=nid,
                factory_short=factory_short,
                logic_type=logic_type,
                model=model,
                folder=node.get("folder", ""),
                annotation=annotation,
                level=level,
                parent_meta=parent_meta,
            )
            found.append(logic_node)

        # Recurse into sub-workflows
        sub = node.get("sub_workflow")
        if sub and isinstance(sub, dict):
            sub_nodes = sub.get("nodes", {})
            new_prefix = f"{prefix}{nid}:" if prefix else f"{nid}:"
            found.extend(_scan_nodes_recursive(sub_nodes, new_prefix, level + 1, ref))

    return found


def _build_logic_node(
    ref: str,
    node_id: str,
    factory_short: str,
    logic_type: LogicType,
    model: dict,
    folder: str,
    annotation: str,
    level: int,
    parent_meta: str,
) -> LogicNode:
    """Build a LogicNode from raw data."""
    node = LogicNode(
        ref=ref,
        node_id=node_id,
        factory_short=factory_short,
        logic_type=logic_type.value,
        folder=folder,
        annotation=annotation,
        level=level,
        parent_metanode=parent_meta,
        complexity=Complexity.SIMPLE.value,
        python_hint=PYTHON_EQUIV.get(logic_type, ""),
    )

    all_cols = set()
    all_flow_vars = set()

    # ── Rule Engine ──
    if logic_type in (LogicType.RULE_ENGINE, LogicType.RULE_FILTER):
        rules_text = model.get("rules", "")
        if isinstance(rules_text, list):
            rules_text = "\n".join(str(r) for r in rules_text)

        parsed = parse_rules(rules_text)
        node.rules = parsed
        node.rule_count = len(parsed)
        node.is_filter = logic_type == LogicType.RULE_FILTER
        node.output_column = model.get("new-column-name", "")
        node.replaces_column = model.get("replace-column-name", "")
        node.complexity = _assess_complexity(rule_count=len(parsed))

        for rule in parsed:
            all_cols.update(rule.get("columns_used", []))

    # ── JEP / String Expression ──
    elif logic_type in (LogicType.JEP_EXPRESSION, LogicType.STRING_MANIPULATION):
        expr = model.get("expression", model.get("Expression", ""))
        if isinstance(expr, list):
            expr = "\n".join(str(e) for e in expr)

        node.expression = expr
        columns, functions, flow_vars = parse_expression(expr)
        node.expression_columns = columns
        node.expression_functions = functions
        node.appends_column = bool(model.get("append_column", False))
        node.target_column = model.get("replaced_column", "")
        node.return_type = model.get("return_type", "")
        node.complexity = _assess_complexity(expr_len=len(expr))

        all_cols.update(columns)
        all_flow_vars.update(flow_vars)

    # ── Java Snippet ──
    elif logic_type in (LogicType.JAVA_SNIPPET, LogicType.JAVA_EDIT_VAR):
        parsed = parse_java_snippet(model)
        node.code = parsed["code"]
        node.code_language = "java"
        node.imports = [parsed["imports"]] if parsed["imports"] else []
        node.input_columns = parsed["in_cols"]
        node.output_columns = parsed["out_cols"]
        node.input_variables = parsed["in_vars"]
        node.output_variables = parsed["out_vars"]
        node.complexity = _assess_complexity(code_len=len(parsed["code"]))

        all_cols.update(parsed["in_cols"])
        all_cols.update(parsed["out_cols"])

    # ── Python Script ──
    elif logic_type == LogicType.PYTHON_SCRIPT:
        parsed = parse_python_script(model)
        node.code = parsed["code"]
        node.code_language = f"python{parsed['python_version']}"
        node.complexity = _assess_complexity(code_len=len(parsed["code"]))

        # Extract column refs from Python code
        py_col_refs = re.findall(r'\[[\"\'](\w+)[\"\']\]', parsed["code"])
        all_cols.update(py_col_refs)

    node.all_columns_referenced = sorted(all_cols)
    node.flow_variables_used = sorted(all_flow_vars)

    return node


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def scan_workflow(json_path: str) -> LogicMap:
    """Scan a KNIME workflow JSON and produce a LogicMap."""
    path = Path(json_path)
    with open(path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    wf_name = path.stem
    total_nodes = _count_nodes(workflow.get("nodes", {}))

    logic_nodes = _scan_nodes_recursive(workflow.get("nodes", {}))

    # Counts by category
    rule_types = {LogicType.RULE_ENGINE.value, LogicType.RULE_FILTER.value}
    expr_types = {LogicType.JEP_EXPRESSION.value, LogicType.STRING_MANIPULATION.value}
    snippet_types = {LogicType.JAVA_SNIPPET.value, LogicType.JAVA_EDIT_VAR.value, LogicType.PYTHON_SCRIPT.value}

    rule_count = sum(1 for n in logic_nodes if n.logic_type in rule_types)
    expr_count = sum(1 for n in logic_nodes if n.logic_type in expr_types)
    snippet_count = sum(1 for n in logic_nodes if n.logic_type in snippet_types)

    # Type distribution
    by_type: dict[str, int] = {}
    for n in logic_nodes:
        by_type[n.logic_type] = by_type.get(n.logic_type, 0) + 1

    # Aggregate columns and flow vars
    all_columns = set()
    all_flow_vars = set()
    for n in logic_nodes:
        all_columns.update(n.all_columns_referenced)
        all_flow_vars.update(n.flow_variables_used)

    # Complexity distribution
    complexity_dist = {}
    for n in logic_nodes:
        complexity_dist[n.complexity] = complexity_dist.get(n.complexity, 0) + 1

    logic_map = LogicMap(
        workflow_name=wf_name,
        scan_timestamp=datetime.now().isoformat(),
        total_nodes=total_nodes,
        logic_nodes_count=len(logic_nodes),
        rule_engine_count=rule_count,
        expression_count=expr_count,
        snippet_count=snippet_count,
        nodes=[asdict(n) for n in logic_nodes],
        by_type=by_type,
        all_columns=sorted(all_columns),
        all_flow_variables=sorted(all_flow_vars),
        summary={
            "total_logic_nodes": len(logic_nodes),
            "rule_engine_nodes": rule_count,
            "expression_nodes": expr_count,
            "snippet_nodes": snippet_count,
            "total_rules_parsed": sum(n.rule_count for n in logic_nodes),
            "unique_columns": len(all_columns),
            "unique_flow_variables": len(all_flow_vars),
            "complexity_distribution": complexity_dist,
            "has_java_code": any(n.logic_type in (LogicType.JAVA_SNIPPET.value, LogicType.JAVA_EDIT_VAR.value) for n in logic_nodes),
            "has_python_code": any(n.logic_type == LogicType.PYTHON_SCRIPT.value for n in logic_nodes),
        },
    )

    return logic_map


def _count_nodes(nodes: dict) -> int:
    count = len(nodes)
    for node in nodes.values():
        sub = node.get("sub_workflow")
        if sub and isinstance(sub, dict):
            count += _count_nodes(sub.get("nodes", {}))
    return count


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else "KNIME_WORKFLOW_ANALYSIS.json"
    print(f"Scanning: {json_path}")

    result = scan_workflow(json_path)

    output_path = "logic_map.json"
    result.to_json(output_path)

    print(result.to_summary())
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
