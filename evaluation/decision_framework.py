"""
evaluation/decision_framework.py
----------------------------------
The DecisionEngine: rule-based strategy selector.

This is the article's primary deliverable — a structured decision framework
that maps observable production signals to strategy recommendations.

The decision logic encodes the heuristics that experienced ML engineers apply
informally.  Making them explicit and testable serves two purposes:
  1. Consistency: the same signals produce the same decision across teams.
  2. Auditability: every recommendation is traceable to a specific rule.

The engine does not replace judgment — it structures it.  When multiple
signals conflict, the engine returns all valid strategies ranked by preference
with explicit rationale for each.

SIGNAL TAXONOMY
---------------
DataSignals     — properties of the available data
ModelSignals    — properties of the current champion model
DriftSignals    — evidence of distribution or concept change
ConstraintSignals — operational constraints (privacy, latency, data retention)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------

class Strategy(Enum):
    TRAIN_FROM_SCRATCH       = "train_from_scratch"
    FINE_TUNE_HEAD_ONLY      = "fine_tune_head_only"
    FINE_TUNE_FULL_NETWORK   = "fine_tune_full_network"
    RETRAIN_WARM_NEW_ONLY    = "retrain_warm_new_only"
    RETRAIN_WARM_COMBINED    = "retrain_warm_combined"
    RETRAIN_COLD_COMBINED    = "retrain_cold_combined"


# ---------------------------------------------------------------------------
# Signal dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataSignals:
    """
    Signals derived from the data available for the update.

    n_new_samples       : Number of labelled examples in the new dataset
    n_historical_samples: Historical labelled examples available for retraining
                          0 = no historical data (GDPR, data deletion policy, etc.)
    new_classes_added   : True if new output classes are being introduced
    n_new_classes       : Number of new classes (0 if new_classes_added=False)
    label_noise         : Estimated noise rate in new labels (0.0–1.0)
    class_imbalance     : True if minority class is < 10% of the new dataset
    """
    n_new_samples        : int   = 0
    n_historical_samples : int   = 0
    new_classes_added    : bool  = False
    n_new_classes        : int   = 0
    label_noise          : float = 0.0
    class_imbalance      : bool  = False


@dataclass
class ModelSignals:
    """
    Signals derived from the current champion model.

    current_accuracy    : Champion accuracy on the current evaluation set
    accuracy_on_new_data: Champion accuracy evaluated against new test data
                          (measures how much the distribution has changed)
    architecture_fits   : True if the architecture is compatible with the new
                          task — False if, e.g., output dim must change
    is_undertrained     : True if the model never fully converged during prior
                          training (e.g. early stopped too aggressively)
    transfer_quality    : Estimated quality of the prior trunk representations
                          for the new task.  Measured via linear probe accuracy:
                          freeze trunk, train head on new data, measure accuracy.
                          0.0 = random, 1.0 = perfect linear separability.
    """
    current_accuracy     : float = 0.0
    accuracy_on_new_data : float = 0.0
    architecture_fits    : bool  = True
    is_undertrained      : bool  = False
    transfer_quality     : float = 0.5


@dataclass
class DriftSignals:
    """
    Signals from the monitoring system indicating distributional change.

    drift_detected      : True if a drift detector (ADWIN, DDM, etc.) fired
    drift_severity      : 'none', 'mild', 'moderate', 'severe'
                          mild     = performance drop < 5%
                          moderate = performance drop 5–15%
                          severe   = performance drop > 15%
    new_task            : True if a genuinely new task arrived (new domain,
                          new concept, not just a shifted distribution)
    task_id_known       : True if task identity is available at inference —
                          affects architecture choices (Article 07)
    """
    drift_detected  : bool  = False
    drift_severity  : str   = "none"   # 'none' | 'mild' | 'moderate' | 'severe'
    new_task        : bool  = False
    task_id_known   : bool  = True


@dataclass
class ConstraintSignals:
    """
    Operational constraints that override performance-based decisions.

    can_store_historical  : False = GDPR / HIPAA / data retention policy
                            prohibits historical data storage — eliminates all
                            combined-data strategies
    max_training_budget_s : Maximum acceptable training time in seconds
                            (0 = no constraint)
    latency_critical      : True if inference latency is a hard constraint —
                            eliminates model expansion strategies that increase
                            parameter count
    catastrophic_forgetting_forbidden: True if any drop in old-task performance
                            is unacceptable — routes to architecture methods
                            (PNN, PackNet from Articles 05 and 07)
    """
    can_store_historical           : bool  = True
    max_training_budget_s          : float = 0.0
    latency_critical               : bool  = False
    catastrophic_forgetting_forbidden: bool = False


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    """
    A single strategy recommendation with rationale and confidence.
    """
    strategy  : Strategy
    confidence: str        # 'high' | 'medium' | 'low'
    rationale : str
    warnings  : List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"  Strategy   : {self.strategy.value}",
            f"  Confidence : {self.confidence}",
            f"  Rationale  : {self.rationale}",
        ]
        if self.warnings:
            lines.append("  Warnings   :")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


@dataclass
class DecisionResult:
    """
    Full output of the DecisionEngine.
    """
    primary    : Decision
    alternatives: List[Decision] = field(default_factory=list)
    ruled_out  : List[Tuple[Strategy, str]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "  DECISION ENGINE OUTPUT",
            "=" * 70,
            "",
            "PRIMARY RECOMMENDATION",
            "-" * 70,
            str(self.primary),
        ]
        if self.alternatives:
            lines += ["", "ALTERNATIVES (ranked by preference)", "-" * 70]
            for i, alt in enumerate(self.alternatives, 1):
                lines.append(f"  [{i}]")
                lines.append(str(alt))
        if self.ruled_out:
            lines += ["", "RULED OUT", "-" * 70]
            for strat, reason in self.ruled_out:
                lines.append(f"  {strat.value:<35} — {reason}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Decision Engine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """
    Maps observable production signals to strategy recommendations.

    Usage
    -----
    engine = DecisionEngine()
    result = engine.decide(
        data       = DataSignals(n_new_samples=300, ...),
        model      = ModelSignals(accuracy_on_new_data=0.72, ...),
        drift      = DriftSignals(drift_detected=True, drift_severity='severe'),
        constraints= ConstraintSignals(can_store_historical=True),
    )
    print(result.summary())
    """

    # Thresholds — adjust to match your organisation's risk tolerance
    FEW_SHOT_THRESHOLD        = 500    # n_new_samples below this = few-shot regime
    ABUNDANT_DATA_THRESHOLD   = 5_000  # n_new_samples above this = data-rich regime
    GOOD_TRANSFER_THRESHOLD   = 0.70   # linear probe accuracy above this = good transfer
    POOR_TRANSFER_THRESHOLD   = 0.55   # below this = poor transfer, scratch is safer
    SEVERE_DRIFT_ACC_DROP     = 0.15   # accuracy drop that qualifies as severe drift
    MILD_DRIFT_ACC_DROP       = 0.05   # accuracy drop that qualifies as mild drift

    def decide(
        self,
        data        : DataSignals,
        model       : ModelSignals,
        drift       : DriftSignals,
        constraints : ConstraintSignals,
    ) -> DecisionResult:

        ruled_out   : List[Tuple[Strategy, str]] = []
        candidates  : List[Decision]             = []

        # ------------------------------------------------------------------
        # Step 1: Apply hard constraints
        # ------------------------------------------------------------------
        eliminated = self._apply_constraints(data, model, drift, constraints, ruled_out)

        # ------------------------------------------------------------------
        # Step 2: Score each non-eliminated strategy
        # ------------------------------------------------------------------
        all_strategies = list(Strategy)
        for strategy in all_strategies:
            if strategy in eliminated:
                continue
            decision = self._score_strategy(
                strategy, data, model, drift, constraints
            )
            if decision is not None:
                candidates.append(decision)

        # ------------------------------------------------------------------
        # Step 3: Rank candidates
        # ------------------------------------------------------------------
        if not candidates:
            # Fallback: if everything is ruled out, recommend scratch with warning
            primary = Decision(
                strategy   = Strategy.TRAIN_FROM_SCRATCH,
                confidence = "low",
                rationale  = "All other strategies were eliminated by constraints.",
                warnings   = ["Review constraints — this fallback may indicate a misconfiguration."],
            )
            return DecisionResult(primary=primary, ruled_out=ruled_out)

        # Sort by confidence tier: high > medium > low
        tier_order = {"high": 0, "medium": 1, "low": 2}
        candidates.sort(key=lambda d: tier_order.get(d.confidence, 3))

        primary      = candidates[0]
        alternatives = candidates[1:3]  # top 2 alternatives

        return DecisionResult(
            primary      = primary,
            alternatives = alternatives,
            ruled_out    = ruled_out,
        )

    # ------------------------------------------------------------------
    # Constraint elimination
    # ------------------------------------------------------------------

    def _apply_constraints(
        self,
        data        : DataSignals,
        model       : ModelSignals,
        drift       : DriftSignals,
        constraints : ConstraintSignals,
        ruled_out   : list,
    ) -> set:
        eliminated = set()

        # No historical data — all combined strategies are impossible
        if not constraints.can_store_historical or data.n_historical_samples == 0:
            for s in [Strategy.RETRAIN_WARM_COMBINED, Strategy.RETRAIN_COLD_COMBINED]:
                eliminated.add(s)
                ruled_out.append((s, "Historical data unavailable (retention policy or no stored data)"))

        # Architecture does not fit — cannot fine-tune without expansion
        if not model.architecture_fits:
            eliminated.add(Strategy.FINE_TUNE_HEAD_ONLY)
            eliminated.add(Strategy.FINE_TUNE_FULL_NETWORK)
            ruled_out.append((Strategy.FINE_TUNE_HEAD_ONLY,    "Architecture incompatible with new task"))
            ruled_out.append((Strategy.FINE_TUNE_FULL_NETWORK, "Architecture incompatible with new task"))

        # Zero forgetting required — fine-tuning and warm-new-only are too risky
        if constraints.catastrophic_forgetting_forbidden:
            for s in [Strategy.FINE_TUNE_FULL_NETWORK, Strategy.RETRAIN_WARM_NEW_ONLY]:
                if s not in eliminated:
                    eliminated.add(s)
                    ruled_out.append((s, "Catastrophic forgetting unacceptable — use PackNet/PNN (Article 07)"))

        return eliminated

    # ------------------------------------------------------------------
    # Strategy scoring
    # ------------------------------------------------------------------

    def _score_strategy(
        self,
        strategy    : Strategy,
        data        : DataSignals,
        model       : ModelSignals,
        drift       : DriftSignals,
        constraints : ConstraintSignals,
    ) -> Optional[Decision]:
        """
        Return a Decision for the given strategy, or None if clearly wrong.
        """

        few_shot    = data.n_new_samples < self.FEW_SHOT_THRESHOLD
        data_rich   = data.n_new_samples >= self.ABUNDANT_DATA_THRESHOLD
        good_xfer   = model.transfer_quality >= self.GOOD_TRANSFER_THRESHOLD
        poor_xfer   = model.transfer_quality < self.POOR_TRANSFER_THRESHOLD
        severe_drift= drift.drift_severity == "severe"
        no_drift    = drift.drift_severity in ("none", "mild") and not drift.drift_detected

        # ---- TRAIN FROM SCRATCH ------------------------------------------
        if strategy == Strategy.TRAIN_FROM_SCRATCH:
            if data_rich and (poor_xfer or drift.new_task or model.is_undertrained):
                return Decision(
                    strategy   = strategy,
                    confidence = "high",
                    rationale  = (
                        "Sufficient new data available and prior representations "
                        "are unlikely to transfer well (poor transfer quality, "
                        "new task, or undertrained prior model)."
                    ),
                )
            if data_rich and severe_drift:
                return Decision(
                    strategy   = strategy,
                    confidence = "medium",
                    rationale  = (
                        "Severe drift detected with abundant new data. "
                        "Warm start may carry over harmful prior representations."
                    ),
                    warnings   = [
                        "Verify that retrain_cold_combined is not better — "
                        "it preserves historical data coverage."
                    ],
                )
            if data_rich:
                return Decision(
                    strategy   = strategy,
                    confidence = "low",
                    rationale  = "Data-rich scenario — scratch is feasible but warm-start options are likely faster.",
                )
            return None  # Not enough data to train from scratch

        # ---- FINE TUNE HEAD ONLY -----------------------------------------
        if strategy == Strategy.FINE_TUNE_HEAD_ONLY:
            if few_shot and good_xfer and not severe_drift:
                return Decision(
                    strategy   = strategy,
                    confidence = "high",
                    rationale  = (
                        "Few-shot regime with good trunk transfer quality and "
                        "no severe drift. Head-only fine-tuning is the lowest-cost "
                        "strategy with strong forgetting protection."
                    ),
                    warnings   = [
                        "If head-only convergence is slow, switch to full fine-tuning."
                    ],
                )
            if few_shot and good_xfer and drift.drift_severity == "mild":
                return Decision(
                    strategy   = strategy,
                    confidence = "medium",
                    rationale  = (
                        "Few-shot with mild drift. Trunk representations likely "
                        "still useful. Monitor old-class accuracy after update."
                    ),
                )
            if data.new_classes_added and good_xfer and few_shot:
                return Decision(
                    strategy   = strategy,
                    confidence = "medium",
                    rationale  = (
                        "New classes added with good trunk transfer. Expand head, "
                        "freeze trunk, fine-tune new class head weights only."
                    ),
                )
            return None

        # ---- FINE TUNE FULL NETWORK --------------------------------------
        if strategy == Strategy.FINE_TUNE_FULL_NETWORK:
            if few_shot and not good_xfer and not severe_drift:
                return Decision(
                    strategy   = strategy,
                    confidence = "high",
                    rationale  = (
                        "Few-shot with moderate transfer quality. Full fine-tuning "
                        "at a lower LR lets the trunk adapt to the new distribution "
                        "without destroying prior representations."
                    ),
                    warnings   = [
                        "Use a 10x lower LR than initial training.",
                        "Monitor backward transfer — old-class accuracy may drop.",
                    ],
                )
            if not few_shot and not data_rich and not poor_xfer:
                return Decision(
                    strategy   = strategy,
                    confidence = "medium",
                    rationale  = (
                        "Moderate data with usable prior representations. "
                        "Full fine-tuning should converge faster than scratch."
                    ),
                )
            return None

        # ---- RETRAIN WARM NEW ONLY ---------------------------------------
        if strategy == Strategy.RETRAIN_WARM_NEW_ONLY:
            if no_drift and not few_shot and not data.new_classes_added:
                return Decision(
                    strategy   = strategy,
                    confidence = "medium",
                    rationale  = (
                        "Minimal drift and moderate new data. Warm start on new "
                        "data takes advantage of prior convergence. Low compute cost."
                    ),
                    warnings   = [
                        "Risk of forgetting prior distribution if new data "
                        "is not representative of historical patterns.",
                        "Consider EWC (Article 05) if forgetting is a concern.",
                    ],
                )
            return None

        # ---- RETRAIN WARM COMBINED ---------------------------------------
        if strategy == Strategy.RETRAIN_WARM_COMBINED:
            if not few_shot and drift.drift_detected and not drift.new_task:
                confidence = "high" if drift.drift_severity in ("moderate", "severe") else "medium"
                return Decision(
                    strategy   = strategy,
                    confidence = confidence,
                    rationale  = (
                        "Drift detected with historical data available. "
                        "Warm-start combined training is the standard production "
                        "response — adapts to new data while retaining old distribution."
                    ),
                )
            if not few_shot and not drift.drift_detected:
                return Decision(
                    strategy   = strategy,
                    confidence = "medium",
                    rationale  = (
                        "No detected drift but new labelled data arrived. "
                        "Warm combined retraining adds the new examples without "
                        "discarding prior knowledge."
                    ),
                )
            return None

        # ---- RETRAIN COLD COMBINED ---------------------------------------
        if strategy == Strategy.RETRAIN_COLD_COMBINED:
            if severe_drift and data_rich and data.n_historical_samples > 0:
                return Decision(
                    strategy   = strategy,
                    confidence = "medium",
                    rationale  = (
                        "Severe drift detected. Prior weights may be harmful as "
                        "initialisation. Cold start on combined data gives a clean "
                        "start while preserving historical distribution coverage."
                    ),
                    warnings   = [
                        "More expensive than warm combined retraining.",
                        "Only justified if warm start shows negative transfer.",
                    ],
                )
            if model.is_undertrained and data.n_historical_samples > 0:
                return Decision(
                    strategy   = strategy,
                    confidence = "medium",
                    rationale  = (
                        "Prior model was undertrained. Cold start avoids "
                        "inheriting a poor initialisation."
                    ),
                )
            return None

        return None
