"""
MapKnime — KNIME Workflow Pattern Mapping & Transpilation Toolkit.

Three mappers that scan KNIME workflow JSON and produce structured metadata
for AI-driven transpilation to Python:

  - temporal_mapper: Date/time nodes, variable chains
  - loop_mapper:     Loop structures (Group, ParamOpt, DBLooping)
  - logic_mapper:    Rule Engine, JEP/String expressions, Java/Python snippets
  - avaliacao_IA:    AI transpiler (Vertex AI Gemini 2.5 Pro)

Usage:
    from MapKnime.temporal_mapper import scan_workflow as scan_temporal
    from MapKnime.loop_mapper import scan_workflow as scan_loops
    from MapKnime.logic_mapper import scan_workflow as scan_logic
    from MapKnime.avaliacao_IA import transpile
"""

from .temporal_mapper import scan_workflow as scan_temporal
from .loop_mapper import scan_workflow as scan_loops
from .logic_mapper import scan_workflow as scan_logic

__all__ = ["scan_temporal", "scan_loops", "scan_logic", "transpile"]
