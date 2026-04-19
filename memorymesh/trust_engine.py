"""BayesianTrustEngine — per-agent Beta-Binomial trust model.

Each agent maintains a Beta(α, β) prior that is updated upon external
confirmation or refutation of memories it has written.

    BTS(agent) = α / (α + β)

The trust score influences conflict resolution: when two agents
disagree, the agent with the higher BTS has its belief weighted
more heavily in the posterior probability computation.

PRD Requirements covered:
    TL-001  Beta-Binomial trust model with (α, β) parameters
    TL-002  BTS computation in < 1ms
    TL-007  trust_update(agent_id, memory_id, outcome) API
    TL-008  trust_audit(agent_id) returning full history
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, NamedTuple, Optional


# ---------------------------------------------------------------------------
# Audit trail entry — append-only forensic record
# ---------------------------------------------------------------------------

class AuditEntry(NamedTuple):
    """Single trust update record for forensic inspection (TL-008).

    Every field is immutable and serializable — the full audit trail
    can be exported as JSON for compliance artifacts.
    """

    timestamp: str          # ISO 8601 UTC
    memory_id: str          # Which memory triggered the update
    was_correct: bool       # External confirmation outcome
    alpha_before: float     # α before update
    beta_before: float      # β before update
    alpha_after: float      # α after update
    beta_after: float       # β after update


# ---------------------------------------------------------------------------
# Per-agent trust state
# ---------------------------------------------------------------------------

@dataclass
class AgentTrustState:
    """Mutable trust state for a single agent.

    Attributes
    ----------
    alpha : float
        Success count + prior.  Default prior α=2 (weakly informative).
    beta : float
        Failure count + prior.  Default prior β=2.
    audit_log : list[AuditEntry]
        Append-only history of all trust updates.
    """

    alpha: float = 2.0
    beta: float = 2.0
    audit_log: List[AuditEntry] = field(default_factory=list)

    @property
    def bts(self) -> float:
        """Bayesian Trust Score: α / (α + β).  O(1), < 1μs."""
        return self.alpha / (self.alpha + self.beta)


# ---------------------------------------------------------------------------
# Trust Engine
# ---------------------------------------------------------------------------

class BayesianTrustEngine:
    """Per-agent Beta-Binomial trust manager.

    Thread-safety note: this class is NOT thread-safe.  For concurrent
    agent pipelines, wrap calls in a lock or use one engine per thread
    and merge via CRDT-style max(α) / max(β) reconciliation.

    Usage
    -----
    >>> engine = BayesianTrustEngine()
    >>> engine.register_agent("agent-researcher")
    >>> engine.get_bts("agent-researcher")
    0.5
    >>> engine.trust_update("agent-researcher", "mem-001", outcome=True)
    >>> engine.get_bts("agent-researcher")
    0.6
    """

    def __init__(self, default_alpha: float = 2.0, default_beta: float = 2.0):
        """Initialize the trust engine with configurable default priors.

        Parameters
        ----------
        default_alpha : float
            Default α prior for newly registered agents.
        default_beta : float
            Default β prior for newly registered agents.
        """
        self._default_alpha = default_alpha
        self._default_beta = default_beta
        self._agents: Dict[str, AgentTrustState] = {}

    # -----------------------------------------------------------------
    # Agent registration
    # -----------------------------------------------------------------

    def register_agent(
        self,
        agent_id: str,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> None:
        """Register an agent with an initial Beta(α, β) prior (TL-001).

        If the agent is already registered, this is a no-op — existing
        trust state is preserved.

        Parameters
        ----------
        agent_id : str
            Unique agent identifier.
        alpha : float, optional
            Initial α.  Defaults to engine-level default.
        beta : float, optional
            Initial β.  Defaults to engine-level default.
        """
        if agent_id in self._agents:
            return  # Idempotent — don't reset existing state

        self._agents[agent_id] = AgentTrustState(
            alpha=alpha if alpha is not None else self._default_alpha,
            beta=beta if beta is not None else self._default_beta,
        )

    # -----------------------------------------------------------------
    # BTS computation (TL-002: < 1ms, guaranteed by O(1) arithmetic)
    # -----------------------------------------------------------------

    def get_bts(self, agent_id: str) -> float:
        """Compute Bayesian Trust Score for an agent.

        Returns α / (α + β).  Sub-microsecond, no external calls.

        Raises
        ------
        KeyError
            If agent_id has not been registered.
        """
        state = self._agents.get(agent_id)
        if state is None:
            raise KeyError(f"Agent not registered: {agent_id}")
        return state.bts

    # -----------------------------------------------------------------
    # Trust update (TL-007)
    # -----------------------------------------------------------------

    def trust_update(
        self,
        agent_id: str,
        memory_id: str,
        outcome: bool,
    ) -> float:
        """Update trust parameters upon external confirmation/refutation.

        Parameters
        ----------
        agent_id : str
            Agent whose trust is being updated.
        memory_id : str
            Memory node that was confirmed or refuted.
        outcome : bool
            True = confirmed correct, False = refuted.

        Returns
        -------
        float
            Updated BTS after the parameter change.

        Raises
        ------
        KeyError
            If agent_id has not been registered.
        """
        state = self._agents.get(agent_id)
        if state is None:
            raise KeyError(f"Agent not registered: {agent_id}")

        # Snapshot before
        alpha_before = state.alpha
        beta_before = state.beta

        # Beta-Binomial conjugate update
        if outcome:
            state.alpha += 1.0
        else:
            state.beta += 1.0

        # Record audit entry
        now_utc = datetime.now(timezone.utc).isoformat()
        entry = AuditEntry(
            timestamp=now_utc,
            memory_id=memory_id,
            was_correct=outcome,
            alpha_before=alpha_before,
            beta_before=beta_before,
            alpha_after=state.alpha,
            beta_after=state.beta,
        )
        state.audit_log.append(entry)

        return state.bts

    # -----------------------------------------------------------------
    # Audit trail (TL-008)
    # -----------------------------------------------------------------

    def trust_audit(self, agent_id: str) -> List[AuditEntry]:
        """Return the complete trust update history for an agent.

        Returns
        -------
        list[AuditEntry]
            Append-only history: (timestamp, memory_id, was_correct,
            α_before, β_before, α_after, β_after).

        Raises
        ------
        KeyError
            If agent_id has not been registered.
        """
        state = self._agents.get(agent_id)
        if state is None:
            raise KeyError(f"Agent not registered: {agent_id}")
        return list(state.audit_log)  # Return a copy for safety

    # -----------------------------------------------------------------
    # Introspection helpers
    # -----------------------------------------------------------------

    def get_parameters(self, agent_id: str) -> tuple[float, float]:
        """Return (α, β) for an agent.  Useful for testing."""
        state = self._agents.get(agent_id)
        if state is None:
            raise KeyError(f"Agent not registered: {agent_id}")
        return (state.alpha, state.beta)

    def registered_agents(self) -> list[str]:
        """Return list of all registered agent IDs."""
        return list(self._agents.keys())

    def __repr__(self) -> str:
        agents_summary = ", ".join(
            f"{aid}(BTS={s.bts:.3f})"
            for aid, s in self._agents.items()
        )
        return f"BayesianTrustEngine([{agents_summary}])"
