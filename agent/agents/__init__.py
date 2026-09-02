"""M3 domain-agent ports and immutable manifest contracts."""
from .manifest import AgentManifest, build_m3_manifests, validate_manifests
from .ports import AgentPort, AfterSalesAgent, LogisticsAgent, OrderAgent, PolicyAgent, ProductAgent, TypedAgentResult

__all__ = ["AgentManifest", "AgentPort", "AfterSalesAgent", "LogisticsAgent", "OrderAgent", "PolicyAgent", "ProductAgent", "TypedAgentResult", "build_m3_manifests", "validate_manifests"]
