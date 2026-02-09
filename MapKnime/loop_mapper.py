"""
loop_mapper.py — Loop Pattern Mapper for KNIME Workflow JSON.

Identifies, classifies, and maps all loop structures within a KNIME
workflow. Produces a structured `loop_map.json` with:
  - Loop pairs (Start ↔ End) with body nodes
  - Internal connections within each loop body
  - Python equivalence hints
  - Nested loop detection
  - Variable flow inside loops (Table↔Variable)

Usage:
    python loop_mapper.py [workflow.json]
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

class LoopType(str, Enum):
    GROUP_LOOP = "GROUP_LOOP"
    DATABASE_LOOPING = "DATABASE_LOOPING"
    PARAM_OPTIMIZATION = "PARAM_OPTIMIZATION"
    COUNTING_LOOP = "COUNTING_LOOP"
    COLUMN_LIST_LOOP = "COLUMN_LIST_LOOP"
    TABLE_ROW_LOOP = "TABLE_ROW_LOOP"
    RECURSIVE_LOOP = "RECURSIVE_LOOP"
    WHILE_LOOP = "WHILE_LOOP"
    CHUNK_LOOP = "CHUNK_LOOP"
    GENERIC_LOOP = "GENERIC_LOOP"


class LoopRole(str, Enum):
    START = "START"
    END = "END"
    BODY = "BODY"
    STANDALONE = "STANDALONE"  # DatabaseLooping is self-contained


# Factory → (LoopType, LoopRole) mapping
LOOP_FACTORY_MAP: dict[str, tuple[LoopType, LoopRole]] = {
    # Group Loop
    "GroupLoopStartNodeFactory": (LoopType.GROUP_LOOP, LoopRole.START),
    # Database Looping (self-contained: iterates internally)
    "DatabaseLoopingNodeFactory": (LoopType.DATABASE_LOOPING, LoopRole.STANDALONE),
    # Parameter Optimization
    "LoopStartParOptNodeFactory": (LoopType.PARAM_OPTIMIZATION, LoopRole.START),
    "LoopEndParOptNodeFactory": (LoopType.PARAM_OPTIMIZATION, LoopRole.END),
    # Loop End variants (generic ends paired with any start)
    "LoopEndDynamicNodeFactory": (LoopType.GENERIC_LOOP, LoopRole.END),
    "LoopEndNodeFactory": (LoopType.GENERIC_LOOP, LoopRole.END),
    "LoopEnd2NodeFactory": (LoopType.GENERIC_LOOP, LoopRole.END),
    # Counting / Table Row / Column List
    "LoopStartCountNodeFactory": (LoopType.COUNTING_LOOP, LoopRole.START),
    "TableRowToRowLoopStartNodeFactory": (LoopType.TABLE_ROW_LOOP, LoopRole.START),
    "ColumnListLoopStartNodeFactory": (LoopType.COLUMN_LIST_LOOP, LoopRole.START),
    # Recursive
    "RecursiveLoopStartNodeFactory": (LoopType.RECURSIVE_LOOP, LoopRole.START),
    "RecursiveLoopEndNodeFactory": (LoopType.RECURSIVE_LOOP, LoopRole.END),
    # While
    "WhileLoopStartNodeFactory": (LoopType.WHILE_LOOP, LoopRole.START),
    "WhileLoopEndNodeFactory": (LoopType.WHILE_LOOP, LoopRole.END),
    # Chunk
    "ChunkLoopStartNodeFactory": (LoopType.CHUNK_LOOP, LoopRole.START),
    "ChunkLoopEndNodeFactory": (LoopType.CHUNK_LOOP, LoopRole.END),
}

# Python equivalence hints by type
PYTHON_HINTS: dict[LoopType, str] = {
    LoopType.GROUP_LOOP: "df.groupby(columns).apply(func)",
    LoopType.DATABASE_LOOPING: "for val in column: execute_sql(sql, val)",
    LoopType.PARAM_OPTIMIZATION: "scipy.optimize.minimize_scalar(func, bounds, method)",
    LoopType.COUNTING_LOOP: "for i in range(n): ...",
    LoopType.TABLE_ROW_LOOP: "for idx, row in df.iterrows(): ...",
    LoopType.COLUMN_LIST_LOOP: "for col in df.columns: ...",
    LoopType.RECURSIVE_LOOP: "while not converged: result = func(result)",
    LoopType.WHILE_LOOP: "while condition: ...",
    LoopType.CHUNK_LOOP: "for chunk in np.array_split(df, n): ...",
    LoopType.GENERIC_LOOP: "for item in iterable: ...",
}


# ═══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LoopNode:
    """A single loop-related node."""
    ref: str
    node_id: str
    factory: str
    factory_short: str
    loop_type: str
    role: str
    folder: str
    annotation: str
    level: int
    parent_metanode: str
    config: dict = field(default_factory=dict)


@dataclass
class LoopPair:
    """A matched Start→End loop pair with body."""
    pair_id: str
    loop_type: str
    python_hint: str
    parent_metanode: str
    level: int
    start_node: dict | None = None
    end_node: dict | None = None
    body_nodes: list = field(default_factory=list)
    body_connections: list = field(default_factory=list)
    nested_loops: list = field(default_factory=list)
    config_summary: dict = field(default_factory=dict)


@dataclass
class StandaloneLoop:
    """A self-contained loop (e.g. DatabaseLooping)."""
    ref: str
    loop_type: str
    python_hint: str
    parent_metanode: str
    level: int
    config: dict = field(default_factory=dict)
    annotation: str = ""
    folder: str = ""


@dataclass
class LoopMap:
    """Complete loop mapping for a workflow."""
    workflow_name: str
    scan_timestamp: str
    total_nodes: int
    total_loop_nodes: int
    loop_pairs: list = field(default_factory=list)
    standalone_loops: list = field(default_factory=list)
    by_type: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False, default=str)

    def to_summary(self) -> str:
        lines = [
            f"=== Loop Map: {self.workflow_name} ===",
            f"Scan: {self.scan_timestamp}",
            f"Total nodes: {self.total_nodes}",
            f"Loop-related nodes: {self.total_loop_nodes}",
            f"  Loop pairs (Start+End): {len(self.loop_pairs)}",
            f"  Standalone loops: {len(self.standalone_loops)}",
            "",
            "By Type:",
        ]
        for lt, count in sorted(self.by_type.items()):
            hint = PYTHON_HINTS.get(LoopType(lt), "")
            lines.append(f"  {lt}: {count} {'pair' if count == 1 else 'pairs'}")
            if hint:
                lines.append(f"    Python: {hint}")

        lines.append("")
        lines.append("Loop Pairs:")
        for pair in self.loop_pairs:
            start_ref = pair.get("start_node", {}).get("ref", "?") if pair.get("start_node") else "?"
            end_ref = pair.get("end_node", {}).get("ref", "?") if pair.get("end_node") else "?"
            body_count = len(pair.get("body_nodes", []))
            nested = len(pair.get("nested_loops", []))
            lines.append(
                f"  [{pair['loop_type']}] {start_ref} -> {end_ref} "
                f"({body_count} body nodes"
                f"{f', {nested} nested' if nested else ''})"
            )
            cfg = pair.get("config_summary", {})
            if cfg:
                for k, v in cfg.items():
                    lines.append(f"    {k}: {v}")

        if self.standalone_loops:
            lines.append("")
            lines.append("Standalone Loops:")
            for sl in self.standalone_loops:
                lines.append(f"  [{sl['loop_type']}] {sl['ref']}: {sl['folder']}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# SCANNING ENGINE
# ═══════════════════════════════════════════════════════════════════

def _get_factory_short(factory: str) -> str:
    return factory.split(".")[-1] if factory else ""


def _classify_factory(factory: str) -> tuple[LoopType | None, LoopRole | None]:
    short = _get_factory_short(factory)
    if short in LOOP_FACTORY_MAP:
        return LOOP_FACTORY_MAP[short]
    # Fallback: check if any key substring matches
    for key, val in LOOP_FACTORY_MAP.items():
        if key.lower() in factory.lower():
            return val
    return None, None


def _extract_loop_config(loop_type: LoopType, role: LoopRole, model: dict) -> dict:
    """Extract the semantically important config for each loop type."""
    if not model:
        return {}

    if loop_type == LoopType.GROUP_LOOP and role == LoopRole.START:
        group_cols = model.get("GroupColNames", {})
        included = group_cols.get("included_names", []) if isinstance(group_cols, dict) else []
        return {
            "group_columns": included,
            "sorted_input": model.get("SortedInput", False),
        }

    if loop_type == LoopType.DATABASE_LOOPING:
        sql = model.get("statement", "")
        col_sel = model.get("column_selection", "")
        return {
            "sql": sql,
            "placeholder": "#PLACE_HOLDER_DO_NOT_EDIT#" if "#PLACE_HOLDER_DO_NOT_EDIT#" in sql else None,
            "column_selection": col_sel,
            "aggregate_by_row": model.get("aggregate_by_row", None),
            "values_per_query": model.get("values_per_query", None),
            "database": model.get("database", ""),
            "driver": model.get("driver", ""),
        }

    if loop_type == LoopType.PARAM_OPTIMIZATION and role == LoopRole.START:
        param = model.get("parameter_0", {})
        return {
            "method": model.get("method", ""),
            "num_trials": model.get("numTrials", 0),
            "early_stopping": model.get("useEarlyStopping", False),
            "tolerance": model.get("tolerance", None),
            "parameter": {
                "name": param.get("name", ""),
                "from": param.get("from", 0),
                "to": param.get("to", 0),
                "step": param.get("step", 0),
                "integer": param.get("integer", False),
            } if param else {},
        }

    if loop_type == LoopType.PARAM_OPTIMIZATION and role == LoopRole.END:
        return {
            "objective_variable": model.get("flowVariableName", ""),
            "maximize": model.get("maximize", False),
        }

    if loop_type == LoopType.GENERIC_LOOP and role == LoopRole.END:
        return {
            "add_iteration_column": model.get("addIterationColumn", False),
            "row_key_policy": model.get("rowKeyPolicy", ""),
            "propagate_loop_variables": model.get("propagateLoopVariables", False),
            "ignore_empty_tables": model.get("ignoreEmptyTables", None),
        }

    return {k: v for k, v in model.items() if not isinstance(v, dict) or len(str(v)) < 200}


def _scan_nodes_recursive(
    nodes: dict,
    connections: list,
    prefix: str = "",
    level: int = 0,
    parent_meta: str = "",
) -> tuple[list[LoopNode], list[dict]]:
    """Recursively scan all nodes for loop factories."""
    found: list[LoopNode] = []
    all_connections: list[dict] = []

    for nid, node in nodes.items():
        settings = node.get("settings", {})
        factory = settings.get("factory", "")
        model = settings.get("model", {}) or {}
        annotation = str(settings.get("annotation", "") or "")

        loop_type, role = _classify_factory(factory)
        ref = f"{prefix}{nid}" if prefix else nid

        if loop_type is not None:
            config = _extract_loop_config(loop_type, role, model)
            found.append(LoopNode(
                ref=ref,
                node_id=nid,
                factory=factory,
                factory_short=_get_factory_short(factory),
                loop_type=loop_type.value,
                role=role.value,
                folder=node.get("folder", ""),
                annotation=annotation.strip()[:200],
                level=level,
                parent_metanode=parent_meta,
                config=config,
            ))

        # Recurse into sub-workflows
        sub = node.get("sub_workflow")
        if sub and isinstance(sub, dict):
            sub_nodes = sub.get("nodes", {})
            sub_conns = sub.get("connections", [])
            new_prefix = f"{prefix}{nid}:" if prefix else f"{nid}:"

            # Tag connections with their scope
            for conn in sub_conns:
                all_connections.append({
                    "scope": ref,
                    "source_id": str(conn.get("source_id", "")),
                    "source_port": conn.get("source_port", 0),
                    "dest_id": str(conn.get("dest_id", "")),
                    "dest_port": conn.get("dest_port", 0),
                    "scope_prefix": new_prefix,
                })

            sub_found, sub_conns_deep = _scan_nodes_recursive(
                sub_nodes, sub_conns, new_prefix, level + 1, ref,
            )
            found.extend(sub_found)
            all_connections.extend(sub_conns_deep)

    return found, all_connections


def _build_pair_id(start_ref: str, end_ref: str, loop_type: str) -> str:
    return f"{loop_type}:{start_ref}->{end_ref}"


def _find_body_between(
    start_id: str,
    end_id: str,
    scope_conns: list[dict],
    all_nodes_in_scope: dict,
) -> tuple[list[dict], list[dict]]:
    """Find nodes between loop start and end via connection traversal."""
    # Build adjacency from connections in this scope
    adj: dict[str, list[str]] = {}
    for conn in scope_conns:
        src = conn["source_id"]
        dst = conn["dest_id"]
        adj.setdefault(src, []).append(dst)

    # BFS from start, stop at end
    visited = set()
    queue = [start_id]
    body_ids = set()

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        if current != start_id and current != end_id:
            body_ids.add(current)
        if current == end_id:
            continue  # Don't traverse past the end
        for neighbor in adj.get(current, []):
            queue.append(neighbor)

    # Collect body node info
    body_nodes = []
    for bid in sorted(body_ids):
        node = all_nodes_in_scope.get(bid, {})
        settings = node.get("settings", {})
        body_nodes.append({
            "id": bid,
            "folder": node.get("folder", ""),
            "factory": _get_factory_short(settings.get("factory", "")),
            "annotation": str(settings.get("annotation", "") or "").strip()[:100],
        })

    # Collect connections within body (including start→body and body→end)
    body_connections = []
    all_ids = body_ids | {start_id, end_id}
    for conn in scope_conns:
        if conn["source_id"] in all_ids and conn["dest_id"] in all_ids:
            body_connections.append({
                "from": conn["source_id"],
                "from_port": conn["source_port"],
                "to": conn["dest_id"],
                "to_port": conn["dest_port"],
            })

    return body_nodes, body_connections


def _make_config_summary(pair_type: LoopType, start_config: dict, end_config: dict) -> dict:
    """Create a human-readable config summary for the loop pair."""
    summary = {}

    if pair_type == LoopType.GROUP_LOOP:
        cols = start_config.get("group_columns", [])
        summary["group_by"] = cols
        summary["sorted"] = start_config.get("sorted_input", False)

    elif pair_type == LoopType.PARAM_OPTIMIZATION:
        summary["method"] = start_config.get("method", "?")
        param = start_config.get("parameter", {})
        if param:
            summary["parameter"] = param.get("name", "?")
            summary["range"] = f"{param.get('from', 0)} to {param.get('to', 0)}, step {param.get('step', 0)}"
        summary["trials"] = start_config.get("num_trials", 0)
        summary["objective"] = end_config.get("objective_variable", "?")
        summary["maximize"] = end_config.get("maximize", False)

    return summary


def _pair_loops(
    loop_nodes: list[LoopNode],
    all_connections: list[dict],
    workflow: dict,
) -> tuple[list[LoopPair], list[StandaloneLoop]]:
    """Match loop starts with their corresponding ends."""
    pairs: list[LoopPair] = []
    standalones: list[StandaloneLoop] = []
    used_ends: set[str] = set()

    # Separate by role
    starts = [n for n in loop_nodes if n.role == LoopRole.START.value]
    ends = [n for n in loop_nodes if n.role == LoopRole.END.value]
    standalone_nodes = [n for n in loop_nodes if n.role == LoopRole.STANDALONE.value]

    # Handle standalone (DatabaseLooping)
    for sn in standalone_nodes:
        standalones.append(StandaloneLoop(
            ref=sn.ref,
            loop_type=sn.loop_type,
            python_hint=PYTHON_HINTS.get(LoopType(sn.loop_type), ""),
            parent_metanode=sn.parent_metanode,
            level=sn.level,
            config=sn.config,
            annotation=sn.annotation,
            folder=sn.folder,
        ))

    # Match starts to ends within the same scope (parent_metanode)
    for start in starts:
        # Find matching end in same scope
        candidates = [
            e for e in ends
            if e.ref not in used_ends
            and e.parent_metanode == start.parent_metanode
        ]

        # Prefer type-specific match, then generic
        best_end = None
        start_lt = LoopType(start.loop_type)

        # First: exact type match
        for c in candidates:
            if c.loop_type == start.loop_type:
                best_end = c
                break

        # Second: generic end (LoopEndDynamic matches any start)
        if best_end is None:
            for c in candidates:
                if c.loop_type == LoopType.GENERIC_LOOP.value:
                    best_end = c
                    break

        if best_end is None:
            continue  # Orphan start, skip

        used_ends.add(best_end.ref)

        # Resolve the actual loop type (start takes precedence)
        resolved_type = start_lt

        # Get body nodes by tracing connections in the scope
        scope = start.parent_metanode
        scope_conns = [
            c for c in all_connections
            if c["scope"] == scope
        ]

        # Get all nodes in this scope's sub_workflow
        scope_nodes = _resolve_scope_nodes(scope, workflow)

        body_nodes, body_connections = _find_body_between(
            start.node_id, best_end.node_id, scope_conns, scope_nodes,
        )

        # Detect nested loops in body
        body_ids = {bn["id"] for bn in body_nodes}
        nested = [
            n.ref for n in loop_nodes
            if n.node_id in body_ids and n.ref != start.ref and n.ref != best_end.ref
        ]

        config_summary = _make_config_summary(
            resolved_type,
            start.config,
            best_end.config,
        )

        pair = LoopPair(
            pair_id=_build_pair_id(start.ref, best_end.ref, resolved_type.value),
            loop_type=resolved_type.value,
            python_hint=PYTHON_HINTS.get(resolved_type, ""),
            parent_metanode=scope,
            level=start.level,
            start_node=asdict(start),
            end_node=asdict(best_end),
            body_nodes=body_nodes,
            body_connections=body_connections,
            nested_loops=nested,
            config_summary=config_summary,
        )
        pairs.append(pair)

    return pairs, standalones


def _resolve_scope_nodes(scope_ref: str, workflow: dict) -> dict:
    """Navigate to the sub_workflow.nodes dict for a given scope ref."""
    if not scope_ref:
        return workflow.get("nodes", {})

    parts = scope_ref.split(":")
    current = workflow.get("nodes", {})

    for part in parts:
        node = current.get(part, {})
        sub = node.get("sub_workflow", {})
        if isinstance(sub, dict):
            current = sub.get("nodes", {})
        else:
            return {}

    return current


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def scan_workflow(json_path: str) -> LoopMap:
    """Scan a KNIME workflow JSON and produce a LoopMap."""
    path = Path(json_path)
    with open(path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    wf_name = path.stem
    total_nodes = _count_nodes(workflow.get("nodes", {}))

    # Add root connections
    root_conns = []
    for conn in workflow.get("connections", []):
        root_conns.append({
            "scope": "",
            "source_id": str(conn.get("source_id", "")),
            "source_port": conn.get("source_port", 0),
            "dest_id": str(conn.get("dest_id", "")),
            "dest_port": conn.get("dest_port", 0),
            "scope_prefix": "",
        })

    # Scan all nodes recursively
    loop_nodes, all_connections = _scan_nodes_recursive(
        workflow.get("nodes", {}),
        workflow.get("connections", []),
    )
    all_connections = root_conns + all_connections

    # Pair loops
    pairs, standalones = _pair_loops(loop_nodes, all_connections, workflow)

    # Build type counts
    by_type: dict[str, int] = {}
    for p in pairs:
        by_type[p.loop_type] = by_type.get(p.loop_type, 0) + 1
    for s in standalones:
        by_type[s.loop_type] = by_type.get(s.loop_type, 0) + 1

    loop_map = LoopMap(
        workflow_name=wf_name,
        scan_timestamp=datetime.now().isoformat(),
        total_nodes=total_nodes,
        total_loop_nodes=len(loop_nodes),
        loop_pairs=[asdict(p) for p in pairs],
        standalone_loops=[asdict(s) for s in standalones],
        by_type=by_type,
        summary={
            "total_loop_structures": len(pairs) + len(standalones),
            "paired_loops": len(pairs),
            "standalone_loops": len(standalones),
            "unique_types": list(by_type.keys()),
            "deepest_nesting": max((p.level for p in pairs), default=0),
            "has_param_optimization": any(
                p.loop_type == LoopType.PARAM_OPTIMIZATION.value for p in pairs
            ),
            "has_database_looping": any(
                s.loop_type == LoopType.DATABASE_LOOPING.value for s in standalones
            ),
        },
    )

    return loop_map


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

    output_path = "loop_map.json"
    result.to_json(output_path)

    print(result.to_summary())
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
