"""
FinGPT AutoGen Orchestrator Module

LLM-powered multi-agent orchestration system based on Microsoft AutoGen framework.
Coordinates agent interactions, manages workflows, and aggregates results for trading decisions.
"""

from fingpt.autogen_orchestrator.autogen_model import AutoGenOrchestrator, create_orchestrator

__all__ = ['AutoGenOrchestrator', 'create_orchestrator']
