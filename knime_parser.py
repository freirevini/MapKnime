"""
KNIME Workflow Parser & Analyzer
Parses .knwf (KNIME Workflow) extracted files, builds execution DAG,
and generates comprehensive analysis in MD, JSON, and HTML formats.
"""

import xml.etree.ElementTree as ET
import os
import json
import html as html_module
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional
import sys

# KNIME XML namespace
NS = "{http://www.knime.org/2008/09/XMLConfig}"


def _longpath(p: str) -> str:
    """Prepend \\\\?\\ on Windows to bypass 260-char MAX_PATH limit."""
    if sys.platform == "win32" and not p.startswith("\\\\?\\"):
        return "\\\\?\\" + os.path.abspath(p)
    return p


def decode_knime_text(text: str) -> str:
    """Decode KNIME-encoded text (%%00010 = newline, %%00009 = tab, %%00013 = CR)."""
    if not text:
        return ""
    text = text.replace("%%00010", "\n").replace("%%00009", "\t").replace("%%00013", "")
    text = html_module.unescape(text)
    return text.strip()


def find_entry(parent, key: str) -> Optional[str]:
    """Find an entry element by key and return its value."""
    for entry in parent.findall(f"{NS}entry"):
        if entry.get("key") == key:
            val = entry.get("value", "")
            isnull = entry.get("isnull", "false")
            return None if isnull == "true" and val == "" else val
    return None


def find_config(parent, key: str):
    """Find a config element by key."""
    for cfg in parent.findall(f"{NS}config"):
        if cfg.get("key") == key:
            return cfg
    return None


def extract_array(config_elem) -> list[str]:
    """Extract array values from a config element with array-size pattern."""
    if config_elem is None:
        return []
    size_str = find_entry(config_elem, "array-size")
    if not size_str:
        return []
    size = int(size_str)
    items = []
    for i in range(size):
        val = find_entry(config_elem, str(i))
        if val is not None:
            items.append(val)
    return items


def parse_model_recursive(elem, depth=0) -> dict:
    """Recursively parse a model config into a nested dict, extracting all values."""
    result = {}
    if depth > 10:
        return result

    for entry in elem.findall(f"{NS}entry"):
        key = entry.get("key", "")
        val = entry.get("value", "")
        entry_type = entry.get("type", "")
        # Skip internal metadata keys
        if key in ("SettingsModelID", "EnabledStatus", "array-size"):
            continue
        # Try numeric index keys -> collect as array
        if key.isdigit():
            continue
        if entry_type == "xboolean":
            result[key] = val == "true"
        elif entry_type in ("xint", "xdouble", "xlong", "xshort"):
            try:
                result[key] = float(val) if "." in val else int(val)
            except ValueError:
                result[key] = val
        elif entry_type == "xpassword":
            result[key] = "***ENCRYPTED***"
        else:
            result[key] = decode_knime_text(val) if val else val

    for cfg in elem.findall(f"{NS}config"):
        key = cfg.get("key", "")
        if key.endswith("_Internals"):
            continue
        # Check if it's an array
        arr_size = find_entry(cfg, "array-size")
        if arr_size is not None:
            result[key] = extract_array(cfg)
        else:
            sub = parse_model_recursive(cfg, depth + 1)
            if sub:
                result[key] = sub

    return result


def parse_settings_xml(settings_path: str) -> dict:
    """Parse a node's settings.xml and extract all configuration details."""
    info = {
        "factory": None,
        "node_name": None,
        "state": None,
        "annotation": None,
        "model": {},
        "ports": [],
    }
    try:
        tree = ET.parse(_longpath(settings_path))
    except (ET.ParseError, FileNotFoundError):
        return info

    root = tree.getroot()
    info["factory"] = find_entry(root, "factory")
    info["node_name"] = find_entry(root, "node-name") or find_entry(root, "name")
    info["state"] = find_entry(root, "state")

    # Node annotation
    ann_cfg = find_config(root, "nodeAnnotation")
    if ann_cfg is not None:
        ann_text = find_entry(ann_cfg, "text")
        if ann_text:
            info["annotation"] = decode_knime_text(ann_text)

    # Model (all configs)
    model_cfg = find_config(root, "model")
    if model_cfg is not None:
        info["model"] = parse_model_recursive(model_cfg)

    # Ports
    ports_cfg = find_config(root, "ports")
    if ports_cfg is not None:
        for pcfg in ports_cfg.findall(f"{NS}config"):
            port_info = {}
            port_info["index"] = find_entry(pcfg, "index")
            port_info["summary"] = find_entry(pcfg, "port_object_summary")
            info["ports"].append(port_info)

    return info


def parse_workflow_knime(workflow_path: str, base_dir: str, level: int = 0) -> dict:
    """Parse a workflow.knime file and recursively resolve MetaNodes."""
    result = {
        "metadata": {},
        "annotations": [],
        "nodes": {},
        "connections": [],
        "execution_order": [],
    }

    try:
        tree = ET.parse(_longpath(workflow_path))
    except (ET.ParseError, FileNotFoundError) as e:
        result["metadata"]["error"] = str(e)
        return result

    root = tree.getroot()

    # Metadata
    result["metadata"]["created_by"] = find_entry(root, "created_by") or ""
    result["metadata"]["version"] = find_entry(root, "version") or ""
    result["metadata"]["state"] = find_entry(root, "state") or ""

    author_cfg = find_config(root, "authorInformation")
    if author_cfg:
        result["metadata"]["authored_by"] = find_entry(author_cfg, "authored-by") or ""
        result["metadata"]["authored_when"] = find_entry(author_cfg, "authored-when") or ""
        result["metadata"]["last_edited_by"] = find_entry(author_cfg, "lastEdited-by") or ""
        result["metadata"]["last_edited_when"] = find_entry(author_cfg, "lastEdited-when") or ""

    # Annotations
    ann_cfg = find_config(root, "annotations")
    if ann_cfg:
        for acfg in ann_cfg.findall(f"{NS}config"):
            text = find_entry(acfg, "text")
            if text:
                x = find_entry(acfg, "x-coordinate") or "0"
                y = find_entry(acfg, "y-coordinate") or "0"
                result["annotations"].append({
                    "text": decode_knime_text(text),
                    "x": int(x),
                    "y": int(y),
                })

    # Sort annotations by x-coordinate for logical reading order
    result["annotations"].sort(key=lambda a: (a["x"], a["y"]))

    # Nodes
    nodes_cfg = find_config(root, "nodes")
    if nodes_cfg:
        for ncfg in nodes_cfg.findall(f"{NS}config"):
            node_id = find_entry(ncfg, "id")
            if not node_id:
                continue
            node_id = int(node_id)
            settings_file = find_entry(ncfg, "node_settings_file") or ""
            is_meta = find_entry(ncfg, "node_is_meta") == "true"
            node_type = find_entry(ncfg, "node_type") or ""

            # Extract UI bounds for x-position (used in execution order)
            bounds = {"x": 0, "y": 0}
            ui_cfg = find_config(ncfg, "ui_settings")
            if ui_cfg:
                bounds_cfg = find_config(ui_cfg, "extrainfo.node.bounds")
                if bounds_cfg:
                    bx = find_entry(bounds_cfg, "0")
                    by = find_entry(bounds_cfg, "1")
                    if bx:
                        bounds["x"] = int(bx)
                    if by:
                        bounds["y"] = int(by)

            # Derive folder name from settings_file path
            folder_name = settings_file.split("/")[0] if "/" in settings_file else ""

            node_info = {
                "id": node_id,
                "folder": folder_name,
                "is_meta": is_meta,
                "node_type": node_type,
                "bounds": bounds,
                "settings": {},
                "sub_workflow": None,
            }

            # Parse settings or sub-workflow
            node_full_path = os.path.join(base_dir, settings_file)
            if is_meta and settings_file.endswith("workflow.knime"):
                meta_dir = os.path.join(base_dir, folder_name)
                node_info["sub_workflow"] = parse_workflow_knime(
                    node_full_path, meta_dir, level + 1
                )
                node_info["settings"]["node_name"] = folder_name.split(" (#")[0] if " (#" in folder_name else folder_name
            elif settings_file.endswith("settings.xml"):
                node_info["settings"] = parse_settings_xml(node_full_path)

            result["nodes"][node_id] = node_info

    # Connections
    conn_cfg = find_config(root, "connections")
    if conn_cfg:
        for ccfg in conn_cfg.findall(f"{NS}config"):
            src = find_entry(ccfg, "sourceID")
            dst = find_entry(ccfg, "destID")
            sp = find_entry(ccfg, "sourcePort")
            dp = find_entry(ccfg, "destPort")
            if src and dst:
                result["connections"].append({
                    "source_id": int(src),
                    "dest_id": int(dst),
                    "source_port": int(sp) if sp else 0,
                    "dest_port": int(dp) if dp else 0,
                })

    # Topological sort
    result["execution_order"] = topological_sort(result["nodes"], result["connections"])

    return result


def topological_sort(nodes: dict, connections: list) -> list[int]:
    """Kahn's algorithm for topological sorting based on connections."""
    valid_ids = set(nodes.keys())
    in_degree = defaultdict(int)
    adjacency = defaultdict(set)

    for nid in valid_ids:
        in_degree[nid] = 0

    for conn in connections:
        src, dst = conn["source_id"], conn["dest_id"]
        if src in valid_ids and dst in valid_ids and dst not in adjacency[src]:
            adjacency[src].add(dst)
            in_degree[dst] += 1

    queue = deque()
    for nid in valid_ids:
        if in_degree[nid] == 0:
            queue.append(nid)

    # Sort initial queue by x-coordinate for deterministic order
    queue = deque(sorted(queue, key=lambda n: nodes[n]["bounds"]["x"]))

    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in sorted(adjacency[node], key=lambda n: nodes[n]["bounds"]["x"] if n in nodes else 0):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Any remaining nodes (cycle or orphan) - add by x-coordinate
    remaining = valid_ids - set(order)
    if remaining:
        order.extend(sorted(remaining, key=lambda n: nodes[n]["bounds"]["x"]))

    return order


def get_node_display_name(node: dict) -> str:
    """Get a human-readable display name for a node."""
    settings = node.get("settings", {})
    name = settings.get("node_name") or node.get("folder", f"Node #{node['id']}")
    return f"{name} (#{node['id']})"


def get_node_config_summary(node: dict) -> dict:
    """Extract key configuration details for display."""
    settings = node.get("settings", {})
    model = settings.get("model", {})
    summary = {}

    # SQL statements
    if "statement" in model:
        summary["SQL"] = model["statement"]
    # Math expressions
    if "expression" in model:
        summary["Expression"] = model["expression"]
        if "replaced_column" in model:
            summary["Target Column"] = model["replaced_column"]
        if "append_column" in model:
            summary["Append"] = model["append_column"]
    # Rule Engine rules
    if "rules" in model and isinstance(model["rules"], list):
        rules = [r for r in model["rules"] if not r.startswith("//")]
        if rules:
            summary["Rules"] = rules
    if "new-column-name" in model:
        summary["New Column"] = model["new-column-name"]
    # Column Filter
    col_filter = model.get("column-filter", {})
    if isinstance(col_filter, dict):
        inc = col_filter.get("included_names", [])
        exc = col_filter.get("excluded_names", [])
        if inc:
            summary["Included Columns"] = inc
        if exc:
            summary["Excluded Columns"] = exc
    # Row Filter / Row Splitter
    row_filter = model.get("rowFilter", {})
    if isinstance(row_filter, dict) and row_filter:
        summary["Filter Type"] = row_filter.get("RowFilter_TypeID", "")
        summary["Filter Column"] = row_filter.get("ColumnName", "")
        summary["Filter Pattern"] = row_filter.get("Pattern", "")
        summary["Include"] = row_filter.get("include", "")
    # Joiner
    left_pred = model.get("leftTableJoinPredicate", [])
    right_pred = model.get("rightTableJoinPredicate", [])
    if left_pred or right_pred:
        summary["Join Left Key"] = left_pred
        summary["Join Right Key"] = right_pred
        summary["Join Mode"] = model.get("compositionMode", "")
        summary["Duplicate Suffix"] = model.get("suffix", "")
        # Column selections
        left_cols = model.get("leftColumnSelectionConfig", {})
        right_cols = model.get("rightColumnSelectionConfig", {})
        if isinstance(left_cols, dict):
            summary["Left Included Columns"] = left_cols.get("included_names", [])
        if isinstance(right_cols, dict):
            summary["Right Included Columns"] = right_cols.get("included_names", [])
    # Column Rename
    if "column_rename" in model or any(k.startswith("change_") for k in model):
        renames = {}
        for k, v in model.items():
            if isinstance(v, dict) and "new_name" in v:
                old = v.get("old_name", k)
                renames[old] = v["new_name"]
        if renames:
            summary["Column Renames"] = renames
    # GroupBy
    group_cols = model.get("grp_col_config", {})
    if isinstance(group_cols, dict):
        summary["Group Columns"] = group_cols.get("included_names", [])
    # Sorter
    if "includedColumns" in model or "sortColumns" in model:
        summary["Sort Config"] = {k: v for k, v in model.items() if "sort" in k.lower() or "column" in k.lower()}
    # Generic: include entire model if nothing specific matched
    if not summary and model:
        summary["Configuration"] = model

    return summary


# ──────────────────── MARKDOWN GENERATOR ────────────────────

def generate_markdown(workflow: dict, output_path: str):
    """Generate comprehensive Markdown report."""
    lines = []

    def w(text=""):
        lines.append(text)

    def render_workflow(wf: dict, level: int = 0, prefix: str = ""):
        """Render a workflow (root or sub-workflow) into markdown."""
        heading = "#" * min(level + 2, 6)

        if level == 0:
            w("# Análise Completa do Workflow KNIME — `Indicador_Calculo_CET_Rodas`")
            w()
            w("---")
            w()
            # Metadata
            w("## 1. Metadados do Projeto")
            w()
            meta = wf["metadata"]
            w(f"| Atributo | Valor |")
            w(f"|----------|-------|")
            w(f"| Versão KNIME | {meta.get('created_by', 'N/A')} |")
            w(f"| Formato | {meta.get('version', 'N/A')} |")
            w(f"| Estado | {meta.get('state', 'N/A')} |")
            w(f"| Autor | {meta.get('authored_by', 'N/A')} |")
            w(f"| Data Criação | {meta.get('authored_when', 'N/A')} |")
            w(f"| Último Editor | {meta.get('last_edited_by', 'N/A')} |")
            w(f"| Última Edição | {meta.get('last_edited_when', 'N/A')} |")
            w()

        # Annotations
        if wf["annotations"]:
            w(f"{heading} {'2. ' if level == 0 else ''}Anotações (Blocos Lógicos)")
            w()
            for i, ann in enumerate(wf["annotations"]):
                w(f"| {i+1} | {ann['text']} | (x={ann['x']}, y={ann['y']}) |")
            w()

        # Node Inventory
        nodes = wf["nodes"]
        order = wf["execution_order"]

        native = {nid: n for nid, n in nodes.items() if not n["is_meta"]}
        metas = {nid: n for nid, n in nodes.items() if n["is_meta"]}

        w(f"{heading} {'3. ' if level == 0 else ''}Inventário de Nodes ({len(nodes)} total: {len(native)} NativeNodes + {len(metas)} MetaNodes)")
        w()
        w("| # | ID | Nome | Tipo | Anotação |")
        w("|---|-----|------|------|----------|")
        for step, nid in enumerate(order, 1):
            node = nodes.get(nid, {})
            name = get_node_display_name(node)
            ntype = "MetaNode" if node.get("is_meta") else "NativeNode"
            ann = node.get("settings", {}).get("annotation", "")
            if not ann and node.get("sub_workflow"):
                ann = ""
            w(f"| {step} | {nid} | {name} | {ntype} | {ann or '-'} |")
        w()

        # Connections Map
        w(f"{heading} {'4. ' if level == 0 else ''}Mapa de Conexões ({len(wf['connections'])} conexões)")
        w()
        w("| Origem (ID:Porta) | Destino (ID:Porta) | Origem (Nome) | Destino (Nome) |")
        w("|--------------------|--------------------|---------------|-----------------|")
        for conn in wf["connections"]:
            src_id = conn["source_id"]
            dst_id = conn["dest_id"]
            src_name = get_node_display_name(nodes[src_id]) if src_id in nodes else f"Port #{src_id}"
            dst_name = get_node_display_name(nodes[dst_id]) if dst_id in nodes else f"Port #{dst_id}"
            w(f"| {src_id}:{conn['source_port']} | {dst_id}:{conn['dest_port']} | {src_name} | {dst_name} |")
        w()

        # Execution Sequence
        w(f"{heading} {'5. ' if level == 0 else ''}Sequência Cronológica de Execução")
        w()
        for step, nid in enumerate(order, 1):
            node = nodes.get(nid, {})
            name = get_node_display_name(node)
            ntype = "MetaNode" if node.get("is_meta") else "NativeNode"

            # Find incoming and outgoing connections
            incoming = [c for c in wf["connections"] if c["dest_id"] == nid]
            outgoing = [c for c in wf["connections"] if c["source_id"] == nid]

            w(f"**Passo {step}: {name}** (`{ntype}`)")
            w()

            ann = node.get("settings", {}).get("annotation")
            if ann:
                w(f"> 📝 {ann}")
                w()

            if incoming:
                source_parts = []
                for c in incoming:
                    sid = c['source_id']
                    sname = get_node_display_name(nodes[sid]) if sid in nodes else f"Port #{sid}"
                    source_parts.append(f"{sname}(porta {c['source_port']})")
                w(f"- **Entrada**: {', '.join(source_parts)}")
            else:
                w("- **Entrada**: Nenhuma (nó inicial)")

            if outgoing:
                dest_parts = []
                for c in outgoing:
                    did = c['dest_id']
                    dname = get_node_display_name(nodes[did]) if did in nodes else f"Port #{did}"
                    dest_parts.append(f"{dname}(porta {c['dest_port']})")
                w(f"- **Saída**: {', '.join(dest_parts)}")
            else:
                w("- **Saída**: Nenhuma (nó terminal)")

            # Config summary
            config = get_node_config_summary(node)
            if config:
                w("- **Configurações**:")
                for ck, cv in config.items():
                    if isinstance(cv, list):
                        if len(cv) > 3:
                            w(f"  - {ck}:")
                            for item in cv:
                                w(f"    - `{item}`")
                        else:
                            w(f"  - {ck}: {', '.join(f'`{x}`' for x in cv)}")
                    elif isinstance(cv, str) and "\n" in cv:
                        w(f"  - {ck}:")
                        w("    ```sql")
                        for sql_line in cv.split("\n"):
                            w(f"    {sql_line}")
                        w("    ```")
                    elif isinstance(cv, dict):
                        w(f"  - {ck}: `{json.dumps(cv, ensure_ascii=False, default=str)}`")
                    else:
                        w(f"  - {ck}: `{cv}`")

            # Port summaries
            ports = node.get("settings", {}).get("ports", [])
            if ports:
                port_strs = [f"Port {p['index']}: {p['summary']}" for p in ports if p.get("summary")]
                if port_strs:
                    w(f"- **Portas**: {'; '.join(port_strs)}")

            w()
            w("---")
            w()

        # MetaNode details
        if metas:
            w(f"{heading} {'6. ' if level == 0 else ''}Detalhamento dos MetaNodes")
            w()
            for nid in order:
                node = nodes.get(nid, {})
                if not node.get("is_meta") or not node.get("sub_workflow"):
                    continue
                sub_wf = node["sub_workflow"]
                name = get_node_display_name(node)
                sub_heading = "#" * min(level + 3, 6)
                w(f"{sub_heading} MetaNode: {name}")
                w()
                sub_nodes = sub_wf.get("nodes", {})
                sub_native = sum(1 for n in sub_nodes.values() if not n["is_meta"])
                sub_meta = sum(1 for n in sub_nodes.values() if n["is_meta"])
                w(f"- **Nodes internos**: {len(sub_nodes)} ({sub_native} NativeNodes + {sub_meta} MetaNodes)")
                w(f"- **Conexões internas**: {len(sub_wf.get('connections', []))}")
                w()
                render_workflow(sub_wf, level + 1, prefix=f"{prefix}{name} → ")

        # Mermaid diagram
        if level == 0:
            w("## 7. Diagrama de Fluxo (Mermaid)")
            w()
            w("```mermaid")
            w("graph LR")
            for nid, node in nodes.items():
                name = node.get("settings", {}).get("node_name") or node.get("folder", str(nid))
                clean_name = name.replace('"', "'").replace("(", "[").replace(")", "]")
                if node["is_meta"]:
                    w(f'    N{nid}["{clean_name} #{nid}"]:::meta')
                else:
                    w(f'    N{nid}("{clean_name} #{nid}")')
            for conn in wf["connections"]:
                label = f"|p{conn['source_port']}→p{conn['dest_port']}|"
                w(f"    N{conn['source_id']} -->{label} N{conn['dest_id']}")
            w("    classDef meta fill:#f9d71c,stroke:#333,stroke-width:2px")
            w("```")
            w()

    render_workflow(workflow)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Markdown: {output_path}")


# ──────────────────── JSON GENERATOR ────────────────────

def generate_json(workflow: dict, output_path: str):
    """Generate structured JSON output."""

    def clean_for_json(obj):
        """Ensure all values are JSON-serializable."""
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [clean_for_json(i) for i in obj]
        elif isinstance(obj, set):
            return list(obj)
        return obj

    output = clean_for_json(workflow)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"[OK] JSON: {output_path}")


# ──────────────────── HTML GENERATOR ────────────────────

def generate_html(workflow: dict, output_path: str):
    """Generate interactive HTML report."""

    def node_card(node: dict, step: int, connections: list, nodes_map: dict) -> str:
        """Generate HTML card for a single node."""
        name = get_node_display_name(node)
        ntype = "MetaNode" if node.get("is_meta") else "NativeNode"
        badge_class = "meta" if node.get("is_meta") else "native"
        ann = node.get("settings", {}).get("annotation", "")
        config = get_node_config_summary(node)

        incoming = [c for c in connections if c["dest_id"] == node["id"]]
        outgoing = [c for c in connections if c["source_id"] == node["id"]]

        incoming_parts = []
        for c in incoming:
            sid = c["source_id"]
            sname = get_node_display_name(nodes_map[sid]) if sid in nodes_map else f"Port #{sid}"
            incoming_parts.append(f'<span class="conn-ref">{html_module.escape(sname)}</span>')
        incoming_html = ", ".join(incoming_parts) or '<span class="no-conn">Nenhuma (nó inicial)</span>'

        outgoing_parts = []
        for c in outgoing:
            did = c["dest_id"]
            dname = get_node_display_name(nodes_map[did]) if did in nodes_map else f"Port #{did}"
            outgoing_parts.append(f'<span class="conn-ref">{html_module.escape(dname)}</span>')
        outgoing_html = ", ".join(outgoing_parts) or '<span class="no-conn">Nenhuma (nó terminal)</span>'

        config_html = ""
        if config:
            config_items = []
            for ck, cv in config.items():
                if isinstance(cv, list):
                    val = "<br>".join(f"<code>{html_module.escape(str(x))}</code>" for x in cv)
                elif isinstance(cv, str) and "\n" in cv:
                    val = f"<pre>{html_module.escape(cv)}</pre>"
                elif isinstance(cv, dict):
                    val = f"<pre>{html_module.escape(json.dumps(cv, ensure_ascii=False, indent=2, default=str))}</pre>"
                else:
                    val = f"<code>{html_module.escape(str(cv))}</code>"
                config_items.append(f"<dt>{html_module.escape(ck)}</dt><dd>{val}</dd>")
            config_html = f'<details class="config-details"><summary>⚙️ Configurações</summary><dl>{"".join(config_items)}</dl></details>'

        ports_html = ""
        ports = node.get("settings", {}).get("ports", [])
        if ports:
            port_items = [f"<li>Port {p['index']}: {html_module.escape(p.get('summary', 'N/A'))}</li>" for p in ports if p.get("summary")]
            if port_items:
                ports_html = f'<div class="ports">📊 {"".join(port_items)}</div>'

        return f"""
        <div class="node-card" id="node-{node['id']}">
            <div class="node-header">
                <span class="step-badge">#{step}</span>
                <span class="node-name">{html_module.escape(name)}</span>
                <span class="type-badge {badge_class}">{ntype}</span>
            </div>
            {f'<div class="annotation">📝 {html_module.escape(ann)}</div>' if ann else ''}
            <div class="connections">
                <div class="conn-in">⬅️ <strong>Entrada:</strong> {incoming_html}</div>
                <div class="conn-out">➡️ <strong>Saída:</strong> {outgoing_html}</div>
            </div>
            {config_html}
            {ports_html}
        </div>"""

    def render_workflow_html(wf: dict, level: int = 0) -> str:
        """Render workflow nodes as HTML cards."""
        cards = []
        order = wf.get("execution_order", [])
        nodes_map = wf.get("nodes", {})
        connections = wf.get("connections", [])

        for step, nid in enumerate(order, 1):
            node = nodes_map.get(nid)
            if not node:
                continue
            cards.append(node_card(node, step, connections, nodes_map))

            # If metanode, render sub-workflow
            if node.get("is_meta") and node.get("sub_workflow"):
                sub = node["sub_workflow"]
                sub_name = get_node_display_name(node)
                sub_html = render_workflow_html(sub, level + 1)
                cards.append(f"""
                <details class="metanode-expand">
                    <summary>🔍 Expandir MetaNode: {html_module.escape(sub_name)} ({len(sub.get('nodes', {}))} nodes internos)</summary>
                    <div class="sub-workflow" style="margin-left: {20*(level+1)}px;">
                        {sub_html}
                    </div>
                </details>""")

        return "\n".join(cards)

    nodes_html = render_workflow_html(workflow)
    meta = workflow.get("metadata", {})
    total_nodes = len(workflow.get("nodes", {}))
    total_conn = len(workflow.get("connections", []))

    meta_count = sum(1 for n in workflow.get('nodes', {}).values() if n.get('is_meta'))
    exec_steps = len(workflow.get('execution_order', []))

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KNIME Workflow Analysis — Indicador_Calculo_CET_Rodas</title>
    <style>
        :root {{
            --bg: #0f172a; --card-bg: #1e293b; --text: #e2e8f0;
            --accent: #38bdf8; --meta-bg: #fbbf24; --meta-text: #1e293b;
            --native-bg: #22d3ee; --native-text: #1e293b;
            --border: #334155; --code-bg: #0f172a;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg); color: var(--text);
            line-height: 1.6; padding: 2rem;
        }}
        h1 {{ color: var(--accent); margin-bottom: 0.5rem; font-size: 1.8rem; }}
        .meta-table {{ width: 100%; border-collapse: collapse; margin: 1rem 0 2rem; }}
        .meta-table td {{ padding: 0.4rem 1rem; border-bottom: 1px solid var(--border); }}
        .meta-table td:first-child {{ font-weight: 600; color: var(--accent); width: 200px; }}
        .stats {{ display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }}
        .stat-card {{
            background: var(--card-bg); border-radius: 12px; padding: 1rem 1.5rem;
            border: 1px solid var(--border); flex: 1; min-width: 140px; text-align: center;
        }}
        .stat-card .num {{ font-size: 2rem; font-weight: 700; color: var(--accent); }}
        .stat-card .label {{ font-size: 0.85rem; opacity: 0.7; }}
        .node-card {{
            background: var(--card-bg); border-radius: 12px; padding: 1.2rem;
            margin-bottom: 1rem; border: 1px solid var(--border);
            transition: transform 0.15s, box-shadow 0.15s;
        }}
        .node-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 20px rgba(56,189,248,0.15); }}
        .node-header {{ display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem; flex-wrap: wrap; }}
        .step-badge {{
            background: var(--accent); color: var(--bg); border-radius: 6px;
            padding: 0.15rem 0.6rem; font-weight: 700; font-size: 0.85rem;
        }}
        .node-name {{ font-weight: 600; font-size: 1.05rem; }}
        .type-badge {{
            border-radius: 20px; padding: 0.1rem 0.7rem; font-size: 0.75rem; font-weight: 600;
        }}
        .type-badge.meta {{ background: var(--meta-bg); color: var(--meta-text); }}
        .type-badge.native {{ background: var(--native-bg); color: var(--native-text); }}
        .annotation {{ font-style: italic; opacity: 0.8; margin-bottom: 0.5rem; padding-left: 0.5rem; border-left: 3px solid var(--accent); }}
        .connections {{ font-size: 0.9rem; margin: 0.5rem 0; }}
        .conn-in, .conn-out {{ margin: 0.2rem 0; }}
        .conn-ref {{ background: var(--code-bg); padding: 0.1rem 0.4rem; border-radius: 4px; font-size: 0.82rem; }}
        .no-conn {{ opacity: 0.5; }}
        .config-details {{ margin: 0.5rem 0; }}
        .config-details summary {{
            cursor: pointer; font-weight: 600; padding: 0.3rem 0;
            color: var(--accent); user-select: none;
        }}
        .config-details dl {{ padding: 0.5rem 0 0 1rem; }}
        .config-details dt {{ font-weight: 600; margin-top: 0.5rem; color: #94a3b8; }}
        .config-details dd {{ margin-left: 1rem; }}
        .config-details pre {{
            background: var(--code-bg); padding: 0.8rem; border-radius: 8px;
            overflow-x: auto; font-size: 0.85rem; border: 1px solid var(--border);
            white-space: pre-wrap; word-wrap: break-word;
        }}
        code {{
            background: var(--code-bg); padding: 0.1rem 0.4rem; border-radius: 4px;
            font-size: 0.85rem; word-break: break-all;
        }}
        .ports {{ font-size: 0.85rem; opacity: 0.7; margin-top: 0.3rem; }}
        .ports li {{ list-style: none; }}
        .metanode-expand {{ margin: 0.5rem 0 1rem 1rem; }}
        .metanode-expand summary {{
            cursor: pointer; font-weight: 600; color: var(--meta-bg);
            padding: 0.5rem 0; user-select: none; font-size: 1.05rem;
        }}
        .sub-workflow {{
            border-left: 3px solid var(--meta-bg); padding-left: 1rem;
        }}
        .filter-bar {{
            background: var(--card-bg); border-radius: 12px; padding: 1rem;
            margin-bottom: 1.5rem; border: 1px solid var(--border);
        }}
        .filter-bar input {{
            background: var(--bg); color: var(--text); border: 1px solid var(--border);
            border-radius: 8px; padding: 0.5rem 1rem; width: 100%; font-size: 1rem;
        }}
        .filter-bar input:focus {{ outline: none; border-color: var(--accent); }}
    </style>
</head>
<body>
    <h1>🔬 KNIME Workflow Analysis</h1>
    <h2 style="color: #94a3b8; margin-bottom: 1.5rem;">Indicador_Calculo_CET_Rodas</h2>

    <table class="meta-table">
        <tr><td>Versão KNIME</td><td>{meta.get('created_by', 'N/A')}</td></tr>
        <tr><td>Autor</td><td>{meta.get('authored_by', 'N/A')} ({meta.get('authored_when', '')})</td></tr>
        <tr><td>Último Editor</td><td>{meta.get('last_edited_by', 'N/A')} ({meta.get('last_edited_when', '')})</td></tr>
        <tr><td>Estado</td><td>{meta.get('state', 'N/A')}</td></tr>
    </table>

    <div class="stats">
        <div class="stat-card"><div class="num">{total_nodes}</div><div class="label">Nodes (Raiz)</div></div>
        <div class="stat-card"><div class="num">{total_conn}</div><div class="label">Conexões</div></div>
        <div class="stat-card"><div class="num">{meta_count}</div><div class="label">MetaNodes</div></div>
        <div class="stat-card"><div class="num">{exec_steps}</div><div class="label">Passos</div></div>
    </div>

    <div class="filter-bar">
        <input type="text" id="searchInput" placeholder="🔍 Buscar node por nome, ID ou configuração..."
               oninput="filterNodes(this.value)">
    </div>

    <div id="nodesContainer">
        {nodes_html}
    </div>

    <script>
        function filterNodes(query) {{
            const cards = document.querySelectorAll('.node-card');
            const q = query.toLowerCase();
            cards.forEach(card => {{
                const text = card.textContent.toLowerCase();
                card.style.display = text.includes(q) ? '' : 'none';
            }});
        }}
    </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[OK] HTML: {output_path}")


# ──────────────────── MAIN ────────────────────

def main():
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".temp_knime_extract", "fluxo_knime_exemplo", "document", "Indicador_Calculo_CET_Rodas"
    )
    workflow_path = os.path.join(base_dir, "workflow.knime")

    if not os.path.exists(workflow_path):
        print(f"[ERROR] Arquivo não encontrado: {workflow_path}")
        return

    print("[INFO] Parsing workflow.knime (recursivo)...")
    workflow = parse_workflow_knime(workflow_path, base_dir)

    # Count stats
    def count_all_nodes(wf):
        total = len(wf.get("nodes", {}))
        for n in wf.get("nodes", {}).values():
            if n.get("sub_workflow"):
                total += count_all_nodes(n["sub_workflow"])
        return total

    total = count_all_nodes(workflow)
    root_nodes = len(workflow["nodes"])
    root_conn = len(workflow["connections"])

    print(f"[INFO] Total nodes (todos os níveis): {total}")
    print(f"[INFO] Nodes raiz: {root_nodes}, Conexões raiz: {root_conn}")

    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Generate all formats
    generate_markdown(workflow, os.path.join(output_dir, "KNIME_WORKFLOW_ANALYSIS.md"))
    generate_json(workflow, os.path.join(output_dir, "KNIME_WORKFLOW_ANALYSIS.json"))
    generate_html(workflow, os.path.join(output_dir, "KNIME_WORKFLOW_ANALYSIS.html"))

    print(f"\n[DONE] Análise completa gerada com sucesso!")


if __name__ == "__main__":
    main()
