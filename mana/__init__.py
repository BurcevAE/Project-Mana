"""
mana — modular package for the MANA cognitive agent.

Layout (bottom-up dependency order):
    optional_deps  -> capability flags for optional third-party libraries
    config         -> Config, RandomManager
    knowledge      -> legacy KnowledgeBase (pickle-backed)
    web            -> WebSearcher
    llm            -> LLMClient (multi-provider)
    pipeline       -> PipelineSpec / PipelineFactory / BenchmarkSuite
    experience     -> ExperienceDB
    verifier       -> LocalVerifier (sandboxed arithmetic/code checks)
    memory         -> MemoryManager (persistent SQLite memory)
    agent          -> ManaAgent (orchestration + self-improvement)
    voice          -> VoiceInterface
    cli            -> argparse entry point

This split replaces the original single ~4700-line MANA_5_4.py file.
Behavior is unchanged; only module boundaries were introduced so that
memory, tools, LLM routing and the evolutionary loop can be developed,
tested and extended independently.
"""
from .config import Config, RandomManager
from .agent import ManaAgent

__all__ = ["Config", "RandomManager", "ManaAgent", "__version__"]
from .version import PRODUCT_VERSION as __version__  # single source of truth
