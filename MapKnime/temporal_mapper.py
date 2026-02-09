"""
temporal_mapper.py — KNIME Temporal Pattern Mapper

Scans KNIME workflow JSON (from knime_parser.py), identifies all nodes/MetaNodes
with temporal references, classifies them hierarchically, traces variable chains,
and produces a structured temporal_map.json for AI-driven transpilation.

Usage:
    python temporal_mapper.py [workflow_json_path]
    # Defaults to KNIME_WORKFLOW_ANALYSIS.json in current directory

Author: Generated for 2ChatKnime project
"""

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from pathlib import Path
from typing import Optional


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class TemporalType(str, Enum):
    RELATIVE_DYNAMIC = "RELATIVE_DYNAMIC"
    ABSOLUTE_FIXED = "ABSOLUTE_FIXED"
    RELATIVE_CALCULATED = "RELATIVE_CALCULATED"
    TEMPORAL_TRANSFORMER = "TEMPORAL_TRANSFORMER"
    TYPE_CONVERTER = "TYPE_CONVERTER"
    TEMPORAL_CONSUMER = "TEMPORAL_CONSUMER"
    VARIABLE_PROPAGATOR = "VARIABLE_PROPAGATOR"
    TEMPORAL_GENERATOR = "TEMPORAL_GENERATOR"


class HierarchyLevel(str, Enum):
    METANODE_PURE = "METANODE_PURE"
    NODE_PURE = "NODE_PURE"
    HYBRID = "HYBRID"


class DateType(str, Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    SEQUENTIAL = "sequential"


class ConfidenceLevel(str, Enum):
    CONFIRMED = "CONFIRMED"
    PROBABLE = "PROBABLE"
    REFERENCE = "REFERENCE"


@dataclass
class TemporalSignal:
    ref: str
    node_id: int
    name: str
    factory: str
    temporal_type: TemporalType
    hierarchy: HierarchyLevel
    confidence: int
    confidence_level: ConfidenceLevel
    date_type: DateType
    signals: list = field(default_factory=list)
    variables_produced: list = field(default_factory=list)
    variables_consumed: list = field(default_factory=list)
    model_config: dict = field(default_factory=dict)
    children: list = field(default_factory=list)
    annotation: str = ""


@dataclass
class VariableChain:
    variable_name: str
    var_type: str
    producer_ref: str
    consumer_refs: list = field(default_factory=list)
    sql_usage: list = field(default_factory=list)


@dataclass
class TemporalMap:
    workflow_name: str
    scan_timestamp: str
    total_nodes: int
    temporal_nodes_count: int
    confirmed_count: int
    probable_count: int
    reference_count: int
    signals: list = field(default_factory=list)
    variable_chains: list = field(default_factory=list)
    generators: list = field(default_factory=list)
    by_type: dict = field(default_factory=dict)
    by_hierarchy: dict = field(default_factory=dict)

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2, default=str)

    def to_summary(self) -> str:
        lines = [
            f"═══ Temporal Map: {self.workflow_name} ═══",
            f"Scan: {self.scan_timestamp}",
            f"Total nodes: {self.total_nodes}",
            f"Temporal nodes: {self.temporal_nodes_count}",
            f"  CONFIRMED (≥60%): {self.confirmed_count}",
            f"  PROBABLE (30-59%): {self.probable_count}",
            f"  REFERENCE (<30%): {self.reference_count}",
            "",
            "By Type:",
        ]
        for t, refs in self.by_type.items():
            lines.append(f"  {t}: {len(refs)} nodes")
        lines.append("")
        lines.append("By Hierarchy:")
        for h, refs in self.by_hierarchy.items():
            lines.append(f"  {h}: {len(refs)} nodes")
        if self.generators:
            lines.append("")
            lines.append(f"Temporal Generators: {', '.join(self.generators)}")
        if self.variable_chains:
            lines.append("")
            lines.append("Variable Chains:")
            for vc in self.variable_chains:
                chain = vc if isinstance(vc, dict) else asdict(vc)
                consumers = " → ".join(chain.get("consumer_refs", []))
                lines.append(
                    f"  {chain['variable_name']}: {chain['producer_ref']} → {consumers}"
                )
        return "\n".join(lines)

    def get_by_type(self, temporal_type: TemporalType) -> list:
        return [s for s in self.signals if s.temporal_type == temporal_type.value
                or s.get("temporal_type") == temporal_type.value]

    def get_by_hierarchy(self, hierarchy: HierarchyLevel) -> list:
        return [s for s in self.signals if s.hierarchy == hierarchy.value
                or s.get("hierarchy") == hierarchy.value]

    def get_variable_origin(self, var_name: str) -> Optional[dict]:
        for vc in self.variable_chains:
            chain = vc if isinstance(vc, dict) else asdict(vc)
            if chain["variable_name"] == var_name:
                return chain
        return None


# =============================================================================
# FACTORY PATTERN REGISTRY
# =============================================================================

FACTORY_PATTERNS = {
    "CreateDateTimeNodeFactory": (TemporalType.RELATIVE_DYNAMIC, 40),
    "CreateDateTimeNodeFactory2": (TemporalType.RELATIVE_DYNAMIC, 40),
    "DateTimeShiftNodeFactory": (TemporalType.RELATIVE_CALCULATED, 40),
    "DateTimeDifferenceNodeFactory": (TemporalType.RELATIVE_CALCULATED, 40),
    "ExtractDateTimeFieldsNodeFactory": (TemporalType.TEMPORAL_TRANSFORMER, 40),
    "ExtractDateTimeFieldsNodeFactory2": (TemporalType.TEMPORAL_TRANSFORMER, 40),
    "ModifyTimeNodeFactory": (TemporalType.TEMPORAL_TRANSFORMER, 40),
    "ModifyTimeNodeFactory2": (TemporalType.TEMPORAL_TRANSFORMER, 40),
    "LegacyDateTimeToNewDateTimeNodeFactory": (TemporalType.TYPE_CONVERTER, 40),
    "DateTimeToStringNodeFactory": (TemporalType.TYPE_CONVERTER, 35),
    "StringToDateTimeNodeFactory": (TemporalType.TYPE_CONVERTER, 35),
    "DBReaderNodeFactory": (None, 0),
    "DatabaseLoopingNodeFactory": (None, 0),
    "TableToVariable3NodeFactory": (TemporalType.VARIABLE_PROPAGATOR, 20),
    "TableToVariable4NodeFactory": (TemporalType.VARIABLE_PROPAGATOR, 20),
}

TEMPORAL_MODEL_KEYS = {
    "start_use_exec_time": 30,
    "end_use_exec_time": 15,
    "duration": 20,
    "granularity": 20,
    "numerical_granularity": 20,
    "period_selection": 15,
    "period_value": 10,
    "fixed_date_time": 15,
    "col_select1": 15,
    "col_select2": 15,
    "modus": 10,
    "modify_select": 15,
    "newTypeEnum": 15,
}

SQL_TEMPORAL_VARIABLE_RE = re.compile(r"\$\$\{([SIDB])([^}]+)\}\$\$")
DATE_COLUMN_RE = re.compile(r"\bDt\w+", re.IGNORECASE)
BETWEEN_DATE_RE = re.compile(
    r"BETWEEN\s+['\"]?\$\$\{[SIDB][^}]+\}\$\$", re.IGNORECASE
)


# =============================================================================
# CORE CLASSIFICATION ENGINE
# =============================================================================

def detect_factory_pattern(factory: str):
    """Match factory class name against known temporal patterns."""
    if not factory:
        return None, 0
    class_name = factory.split(".")[-1]
    for pattern, (temporal_type, score) in FACTORY_PATTERNS.items():
        if pattern in class_name:
            return temporal_type, score
    return None, 0


def analyze_model_keys(model: dict):
    """Score model configuration for temporal indicators."""
    signals = []
    score = 0
    model_str = json.dumps(model, default=str)

    for key, points in TEMPORAL_MODEL_KEYS.items():
        if key in model_str:
            val = model.get(key)
            if key == "start_use_exec_time" and val is True:
                signals.append(f"model:{key}=true (DYNAMIC)")
                score += points
            elif key == "start_use_exec_time" and val is False:
                signals.append(f"model:{key}=false (FIXED)")
                score += 5
            elif val is not None:
                signals.append(f"model:{key}={val}")
                score += points

    return signals, score


def extract_sql_variables(sql: str):
    """Extract temporal variable references from SQL statements."""
    produced = []
    consumed = []
    sql_usage = []

    matches = SQL_TEMPORAL_VARIABLE_RE.findall(sql)
    for var_type, var_name in matches:
        full_var = f"{var_type}{var_name}"
        consumed.append(full_var)

        between_match = BETWEEN_DATE_RE.search(sql)
        if between_match:
            sql_usage.append(f"BETWEEN clause with ${{{full_var}}}$$")

    date_cols = DATE_COLUMN_RE.findall(sql)
    for col in date_cols:
        if col not in [v[1:] for v in consumed]:
            sql_usage.append(f"date_column:{col}")

    return produced, consumed, sql_usage


def classify_hierarchy(ref: str) -> HierarchyLevel:
    """Determine hierarchy level from reference string."""
    parts = ref.split(":")
    if len(parts) == 1:
        return HierarchyLevel.NODE_PURE
    if len(parts) == 2:
        return HierarchyLevel.METANODE_PURE
    return HierarchyLevel.HYBRID


def determine_date_type(
    temporal_type: TemporalType, model: dict
) -> DateType:
    """Determine if temporal reference is absolute, relative, or sequential."""
    if temporal_type in (
        TemporalType.RELATIVE_DYNAMIC,
        TemporalType.RELATIVE_CALCULATED,
    ):
        if model.get("start_use_exec_time") is True:
            return DateType.RELATIVE
        if "duration" in model or "granularity" in model:
            return DateType.RELATIVE
    if temporal_type == TemporalType.ABSOLUTE_FIXED:
        return DateType.ABSOLUTE
    if temporal_type == TemporalType.TEMPORAL_CONSUMER:
        return DateType.RELATIVE
    return DateType.SEQUENTIAL


def confidence_level(score: int) -> ConfidenceLevel:
    if score >= 60:
        return ConfidenceLevel.CONFIRMED
    if score >= 30:
        return ConfidenceLevel.PROBABLE
    return ConfidenceLevel.REFERENCE


def extract_temporal_model_config(model: dict) -> dict:
    """Extract only the temporal-relevant parts of the model config."""
    relevant_keys = set(TEMPORAL_MODEL_KEYS.keys()) | {
        "statement", "column_name", "type", "start", "end",
        "duration_or_end", "nr_rows", "col_select", "col_select1",
        "col_select2", "new_col_name", "output", "replace_or_append",
        "suffix", "numerical_col_select", "numerical_value",
        "time_zone", "time_zone_select", "time",
    }
    result = {}
    for key in relevant_keys:
        if key in model:
            result[key] = model[key]
    return result


# =============================================================================
# NODE ANALYSIS
# =============================================================================

def classify_node(
    node: dict,
    all_connections: list,
    prefix: str = "",
    level: int = 0,
) -> Optional[TemporalSignal]:
    """Analyze a single node for temporal characteristics."""
    nid = node.get("id", 0)
    folder = node.get("folder", "")
    settings = node.get("settings", {})
    factory = settings.get("factory", "")
    model = settings.get("model", {}) or {}
    annotation = str(settings.get("annotation", "") or "")

    signals = []
    score = 0

    # 1. Factory pattern matching (strongest signal)
    factory_type, factory_score = detect_factory_pattern(factory)
    if factory_score > 0:
        signals.append(f"factory:{factory.split('.')[-1]}={factory_type}")
        score += factory_score

    # 2. Model key analysis
    model_signals, model_score = analyze_model_keys(model)
    signals.extend(model_signals)
    score += model_score

    # 3. SQL temporal variable analysis
    vars_produced = []
    vars_consumed = []
    sql_usage = []
    if "statement" in model:
        _, vars_consumed, sql_usage = extract_sql_variables(model["statement"])
        if vars_consumed:
            temporal_vars = [v for v in vars_consumed if v.startswith("SDt") or "Date" in v]
            if temporal_vars:
                signals.append(f"sql_temporal_vars:{temporal_vars}")
                score += 25
            non_temporal = [v for v in vars_consumed if v not in temporal_vars]
            if non_temporal:
                signals.append(f"sql_other_vars:{non_temporal}")
                score += 5
        if sql_usage:
            signals.extend([f"sql:{u}" for u in sql_usage])
            score += 10

    # 4. Refine factory type based on model analysis
    temporal_type = factory_type
    if temporal_type == TemporalType.RELATIVE_DYNAMIC:
        if model.get("start_use_exec_time") is not True:
            temporal_type = TemporalType.ABSOLUTE_FIXED
            signals.append("reclassified:ABSOLUTE_FIXED (no exec_time)")

    if temporal_type is None and vars_consumed:
        temporal_type = TemporalType.TEMPORAL_CONSUMER
    if temporal_type is None and score > 0:
        temporal_type = TemporalType.TEMPORAL_TRANSFORMER

    # 5. Skip if no temporal signals
    if score == 0:
        return None

    # Build reference
    ref = f"{prefix}{nid}" if prefix else str(nid)

    # Determine hierarchy
    hierarchy = classify_hierarchy(ref)

    # Override for MetaNodes containing temporal children
    is_meta = node.get("is_meta", False)
    if is_meta:
        hierarchy = (
            HierarchyLevel.METANODE_PURE
            if level == 0
            else HierarchyLevel.HYBRID
        )

    return TemporalSignal(
        ref=ref,
        node_id=nid,
        name=folder,
        factory=factory,
        temporal_type=temporal_type,
        hierarchy=hierarchy,
        confidence=min(score, 100),
        confidence_level=confidence_level(min(score, 100)),
        date_type=determine_date_type(temporal_type, model),
        signals=signals,
        variables_produced=vars_produced,
        variables_consumed=vars_consumed,
        model_config=extract_temporal_model_config(model),
        annotation=annotation,
    )


# =============================================================================
# METANODE ANALYSIS
# =============================================================================

def analyze_metanode_temporal(
    node: dict, prefix: str, level: int
) -> tuple:
    """Analyze a MetaNode's internals for temporal patterns.

    Returns (children_signals, is_temporal_generator, produced_vars).
    """
    sub = node.get("sub_workflow")
    if not sub or not isinstance(sub, dict):
        return [], False, []

    sub_nodes = sub.get("nodes", {})
    sub_conns = sub.get("connections", [])
    nid = node.get("id", 0)
    new_prefix = f"{prefix}{nid}:" if prefix else f"{nid}:"

    children_signals = []
    has_variable_producer = False
    has_temporal_source = False
    output_vars = []

    for snid, snode in sub_nodes.items():
        sig = classify_node(snode, sub_conns, new_prefix, level + 1)
        if sig:
            children_signals.append(sig)
            if sig.temporal_type == TemporalType.VARIABLE_PROPAGATOR:
                has_variable_producer = True
            if sig.temporal_type in (
                TemporalType.RELATIVE_DYNAMIC,
                TemporalType.ABSOLUTE_FIXED,
            ):
                has_temporal_source = True

        # Recurse into sub-MetaNodes
        if snode.get("sub_workflow") and isinstance(snode["sub_workflow"], dict):
            sub_children, sub_gen, sub_vars = analyze_metanode_temporal(
                snode, new_prefix, level + 1
            )
            children_signals.extend(sub_children)
            if sub_gen:
                has_variable_producer = True
            output_vars.extend(sub_vars)

    # Detect output port connections (node → port -1)
    for conn in sub_conns:
        if conn.get("dest_id") == -1:
            src_id = str(conn.get("source_id"))
            src_node = sub_nodes.get(src_id, {})
            src_settings = src_node.get("settings", {})
            src_factory = src_settings.get("factory", "")
            if "TableToVariable" in src_factory:
                has_variable_producer = True
                # Trace what columns this Table Row to Variable receives
                output_vars.extend(
                    _trace_columns_to_output(src_id, sub_nodes, sub_conns)
                )

    is_generator = has_variable_producer and has_temporal_source
    temporal_child_count = sum(
        1
        for s in children_signals
        if s.confidence >= 30
    )
    if temporal_child_count >= 3:
        is_generator = True

    return children_signals, is_generator, output_vars


def _trace_columns_to_output(
    var_node_id: str, nodes: dict, connections: list
) -> list:
    """Trace column names arriving at a Table Row to Variable node
    by following the rename chain backwards."""
    output_vars = []

    # Find what connects INTO this node
    for conn in connections:
        if str(conn.get("dest_id")) == var_node_id:
            src_id = str(conn.get("source_id"))
            src_node = nodes.get(src_id, {})
            src_model = src_node.get("settings", {}).get("model", {}) or {}

            # If source is Column Rename, extract renamed column names
            if "all_columns" in src_model:
                for _, mapping in src_model["all_columns"].items():
                    if isinstance(mapping, dict):
                        new_name = mapping.get("new_column_name", "")
                        if new_name:
                            output_vars.append(f"S{new_name}")

            # If source is Extract Date&Time Fields, check which fields
            elif src_model.get("col_select"):
                for field_name in [
                    "Year", "Month (number)", "Day of month",
                    "Hour", "Minute", "Second",
                ]:
                    if src_model.get(field_name) is True:
                        output_vars.append(f"I{field_name}")

    return output_vars


# =============================================================================
# VARIABLE CHAIN TRACER
# =============================================================================

def trace_variable_chains(
    signals: list, root_connections: list, root_nodes: dict
) -> list:
    """Trace variable propagation chains across the workflow."""
    chains = {}

    # Find all variable port connections (port 0 is typically variable port)
    var_connections = []
    for conn in root_connections:
        if conn.get("dest_port") == 0 or conn.get("source_port") == 0:
            var_connections.append(conn)

    # Build adjacency for variable flow
    var_flow = {}
    for conn in var_connections:
        src = str(conn["source_id"])
        dst = str(conn["dest_id"])
        src_port = conn.get("source_port", 0)
        if src_port == 0:
            var_flow.setdefault(src, []).append(dst)

    # Find generators (signals with variables_produced)
    for sig in signals:
        sig_dict = sig if isinstance(sig, dict) else asdict(sig)
        if sig_dict.get("variables_produced"):
            for var_name in sig_dict["variables_produced"]:
                ref = sig_dict["ref"]
                node_id = str(sig_dict["node_id"])
                consumers = _follow_var_chain(node_id, var_flow)

                # Check each consumer's SQL for this variable
                sql_refs = []
                for consumer_id in consumers:
                    consumer_node = root_nodes.get(consumer_id, {})
                    consumer_model = (
                        consumer_node.get("settings", {}).get("model", {}) or {}
                    )
                    if "statement" in consumer_model:
                        if var_name in consumer_model["statement"]:
                            sql_refs.append(
                                f"Node #{consumer_id}: "
                                f"{consumer_node.get('folder', '?')}"
                            )

                var_type = "String" if var_name.startswith("S") else (
                    "Integer" if var_name.startswith("I") else (
                        "Double" if var_name.startswith("D") else "Unknown"
                    )
                )

                chains[var_name] = VariableChain(
                    variable_name=var_name,
                    var_type=var_type,
                    producer_ref=ref,
                    consumer_refs=consumers,
                    sql_usage=sql_refs,
                )

    # Also detect chains from MetaNode generators via root connections
    for sig in signals:
        sig_dict = sig if isinstance(sig, dict) else asdict(sig)
        if sig_dict.get("temporal_type") == TemporalType.TEMPORAL_GENERATOR.value:
            meta_id = str(sig_dict["node_id"])
            downstream = var_flow.get(meta_id, [])

            # Check downstream SQLs for any $${S/I...}$$ variables
            for consumer_id in downstream:
                consumer_node = root_nodes.get(consumer_id, {})
                consumer_model = (
                    consumer_node.get("settings", {}).get("model", {}) or {}
                )
                if "statement" in consumer_model:
                    sql = consumer_model["statement"]
                    matches = SQL_TEMPORAL_VARIABLE_RE.findall(sql)
                    for var_type_char, var_name in matches:
                        full_var = f"{var_type_char}{var_name}"
                        if full_var not in chains:
                            # Follow chain further from this consumer
                            further = _follow_var_chain(consumer_id, var_flow)
                            sql_refs = [
                                f"Node #{consumer_id}: "
                                f"{consumer_node.get('folder', '?')}"
                            ]
                            for fid in further:
                                fn = root_nodes.get(fid, {})
                                fm = fn.get("settings", {}).get("model", {}) or {}
                                if "statement" in fm and full_var in fm["statement"]:
                                    sql_refs.append(
                                        f"Node #{fid}: {fn.get('folder', '?')}"
                                    )

                            chains[full_var] = VariableChain(
                                variable_name=full_var,
                                var_type="String" if var_type_char == "S" else (
                                    "Integer" if var_type_char == "I" else "Double"
                                ),
                                producer_ref=sig_dict["ref"],
                                consumer_refs=[consumer_id] + further,
                                sql_usage=sql_refs,
                            )

    return list(chains.values())


def _follow_var_chain(start_id: str, var_flow: dict) -> list:
    """Follow variable port connections downstream."""
    visited = set()
    result = []
    queue = var_flow.get(start_id, [])[:]
    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        result.append(nid)
        queue.extend(var_flow.get(nid, []))
    return result


# =============================================================================
# MAIN SCANNER
# =============================================================================

def scan_workflow(json_path: str) -> TemporalMap:
    """Main entry point — scans KNIME workflow JSON for temporal patterns."""
    with open(json_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    root_nodes = workflow.get("nodes", {})
    root_connections = workflow.get("connections", [])
    workflow_name = workflow.get("name", Path(json_path).stem)

    all_signals = []
    generators = []

    # Count total nodes recursively
    total_nodes = _count_nodes_recursive(workflow)

    for nid, node in root_nodes.items():
        # Classify the node itself
        sig = classify_node(node, root_connections, "", 0)

        if node.get("is_meta"):
            # Analyze MetaNode internals
            children_sigs, is_generator, produced_vars = (
                analyze_metanode_temporal(node, "", 0)
            )

            # If MetaNode is a temporal generator, create/update its signal
            if is_generator:
                if sig is None:
                    sig = TemporalSignal(
                        ref=str(node["id"]),
                        node_id=node["id"],
                        name=node.get("folder", ""),
                        factory="",
                        temporal_type=TemporalType.TEMPORAL_GENERATOR,
                        hierarchy=HierarchyLevel.METANODE_PURE,
                        confidence=85,
                        confidence_level=ConfidenceLevel.CONFIRMED,
                        date_type=DateType.RELATIVE,
                        signals=["metanode:contains_temporal_chain"],
                        annotation=str(
                            node.get("settings", {}).get("annotation", "") or ""
                        ),
                    )
                else:
                    sig.temporal_type = TemporalType.TEMPORAL_GENERATOR
                    sig.confidence = max(sig.confidence, 85)
                    sig.confidence_level = ConfidenceLevel.CONFIRMED
                    sig.signals.append("metanode:contains_temporal_chain")

                sig.variables_produced = produced_vars
                sig.children = [
                    asdict(c)["ref"] for c in children_sigs if c.confidence >= 30
                ]
                generators.append(sig.ref)

            # Add children signals
            all_signals.extend(children_sigs)

        if sig:
            all_signals.append(sig)

    # Sort by confidence descending, then by ref
    all_signals.sort(
        key=lambda s: (-s.confidence, s.ref)
    )

    # Trace variable chains
    variable_chains = trace_variable_chains(
        [asdict(s) for s in all_signals],
        root_connections,
        root_nodes,
    )

    # Build aggregations
    by_type = {}
    by_hierarchy = {}
    confirmed = 0
    probable = 0
    reference = 0

    for sig in all_signals:
        tt = sig.temporal_type.value if isinstance(sig.temporal_type, TemporalType) else sig.temporal_type
        hl = sig.hierarchy.value if isinstance(sig.hierarchy, HierarchyLevel) else sig.hierarchy
        by_type.setdefault(tt, []).append(sig.ref)
        by_hierarchy.setdefault(hl, []).append(sig.ref)
        if sig.confidence >= 60:
            confirmed += 1
        elif sig.confidence >= 30:
            probable += 1
        else:
            reference += 1

    temporal_map = TemporalMap(
        workflow_name=workflow_name,
        scan_timestamp=datetime.now().isoformat(),
        total_nodes=total_nodes,
        temporal_nodes_count=len(all_signals),
        confirmed_count=confirmed,
        probable_count=probable,
        reference_count=reference,
        signals=[asdict(s) for s in all_signals],
        variable_chains=[asdict(vc) for vc in variable_chains],
        generators=generators,
        by_type=by_type,
        by_hierarchy=by_hierarchy,
    )

    return temporal_map


def _count_nodes_recursive(workflow: dict) -> int:
    """Count all nodes recursively including MetaNode internals."""
    total = len(workflow.get("nodes", {}))
    for node in workflow.get("nodes", {}).values():
        sub = node.get("sub_workflow")
        if sub and isinstance(sub, dict):
            total += _count_nodes_recursive(sub)
    return total


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else "KNIME_WORKFLOW_ANALYSIS.json"

    if not Path(json_path).exists():
        print(f"ERROR: File not found: {json_path}")
        sys.exit(1)

    print(f"Scanning: {json_path}")
    temporal_map = scan_workflow(json_path)

    output_path = str(Path(json_path).parent / "temporal_map.json")
    temporal_map.to_json(output_path)

    print(temporal_map.to_summary())
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
