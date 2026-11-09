"""
Lever taxonomy and trigger rules for the pricing-review questioning task.

PROVENANCE (state this in the write-up):
  The ladder and trigger conditions were drafted with model assistance from a
  single real pricing-review transcript, then reviewed and corrected by the
  author, who works in commercial pricing for medical diagnostics.
  Record below how many rules were modified on review -- it is the honesty
  signal that distinguishes expert-endorsed rules from model-invented ones.

      RULES_MODIFIED_ON_EXPERT_REVIEW = None   # <-- set this after review

WHY RULES AND NOT A MODEL CHOOSE THE LEVERS:
  If a general-purpose model decided which question comes next, the ground
  truth would encode that model's guess about an industry it does not work in,
  and the whole "institutional knowledge the base model lacks" argument
  collapses. The rules decide WHICH lever fires; a large model is used only to
  PHRASE it. That keeps the domain knowledge in the labels.

NAMING:
  All metric names here are deliberately generic (deal realised price, margin
  floor). No employer-internal metric vocabulary or threshold values are used.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Deal state: what the rules may look at
# ---------------------------------------------------------------------------


@dataclass
class DealState:
    # commercial shape
    drp_vs_threshold: float          # deal realised price as fraction of threshold; <1.0 is below
    worst_line_margin: float         # lowest line-item margin in the deal
    margin_floor: float              # policy floor for a line item
    hardware_outlay_k: float
    annual_reagent_k: float
    lease_months: int
    breakeven_months: int

    # account / competitive shape
    competitor_present: bool
    competitor_gtm: Optional[str] = None        # "direct" | "dealer" | None if unknown
    competitor_bundling: Optional[bool] = None  # None if unknown
    installed_base: bool = False
    alt_config_available: bool = False
    higher_margin_assay_available: bool = False
    customer_demands_reduction: bool = False

    # conversation state
    exhausted: set = field(default_factory=set)   # lever ids already refused
    answered: dict = field(default_factory=dict)  # lever id -> rep's answer
    consecutive_refusals: int = 0

    # derived
    @property
    def breakeven_outside_term(self) -> bool:
        return self.breakeven_months > self.lease_months

    @property
    def drp_negative(self) -> bool:
        return self.drp_vs_threshold < 0

    @property
    def drp_below_threshold(self) -> bool:
        return self.drp_vs_threshold < 1.0

    @property
    def closed(self) -> set:
        """
        Levers the manager is finished with -- either refused (exhausted) or
        properly answered. Triggers must test THIS, not `exhausted`.

        BUG THIS FIXES: an earlier version tested `exhausted` only, so a lever
        the rep actually ANSWERED never unlocked the next rung and the
        conversation died after about two turns. Mean review length went from
        2.1 turns to a realistic length once triggers moved to `closed`.
        """
        return self.exhausted | set(self.answered)

    @property
    def margin_levers_exhausted(self) -> bool:
        return {1, 2, 3, 4}.issubset(self.closed)


# ---------------------------------------------------------------------------
# Lever definitions
# ---------------------------------------------------------------------------


@dataclass
class Lever:
    id: int
    name: str
    intent: str                      # what the manager is trying to establish
    trigger: Callable[[DealState], bool]
    is_followup: bool = False        # drills into a bare answer rather than advancing


LEVERS = [
    Lever(
        1, "price_increase_headroom",
        "Establish whether there is any room to raise price at all.",
        # Always opens the review. An earlier version gated this on the price
        # being below threshold, which left 178 of 1000 reviews with ZERO
        # questions -- unusable as training data, and wrong anyway: a pricing
        # manager opens with this question on every deal, including healthy
        # ones. The answer on a healthy deal is simply short.
        lambda s: True,
    ),
    Lever(
        2, "targeted_increase_low_margin_lines",
        "Test a narrower increase on the weakest line items.",
        lambda s: 1 in s.closed and s.worst_line_margin < s.margin_floor,
    ),
    Lever(
        3, "rebate_instead_of_reduction",
        "Preserve list price by moving the concession into a rebate.",
        lambda s: 2 in s.closed and s.customer_demands_reduction,
    ),
    Lever(
        4, "config_or_assay_mix",
        "Improve margin through what is sold rather than the price of it.",
        lambda s: 3 in s.closed
        and (s.alt_config_available or s.higher_margin_assay_available),
    ),
    Lever(
        5, "competitor_gtm_model",
        "Understand how the competitor is able to compete on price.",
        lambda s: s.margin_levers_exhausted and s.competitor_present,
    ),
    Lever(
        6, "pressure_and_sensitivity_justified",
        "Challenge whether the claimed pressure is actually evidenced.",
        lambda s: 5 in s.closed,
    ),
    Lever(
        7, "sensitivity_drivers",
        "Drill into WHY the customer is price sensitive.",
        lambda s: 6 in s.answered and _is_bare(s.answered.get(6)),
        is_followup=True,
    ),
    Lever(
        8, "our_price_competitiveness",
        "Get an explicit read on whether our price is competitive in the tender.",
        lambda s: 6 in s.closed and (7 in s.closed or 6 in s.exhausted),
    ),
    Lever(
        9, "drp_improvement",
        "Establish whether the realised price can be improved, and why not if negative.",
        lambda s: 8 in s.closed and (s.drp_negative or s.drp_below_threshold),
    ),
    Lever(
        10, "negative_drp_line_drivers",
        "Localise the damage to specific lines and test repricing room there.",
        lambda s: 9 in s.closed,
    ),
    Lever(
        11, "loss_risk_and_installed_base",
        "Quantify the downside of walking away, including pull-through at risk.",
        lambda s: 10 in s.closed and s.installed_base,
    ),
    Lever(
        12, "competitor_bundling",
        "Establish whether the competitor is bundling or placing instruments free.",
        lambda s: s.competitor_present and s.competitor_bundling is None,
    ),
    Lever(
        13, "post_win_matching_and_renewal",
        "Test what protects the business after the win and at renewal.",
        lambda s: s.competitor_bundling is True,
    ),
    Lever(
        14, "term_vs_breakeven_mismatch",
        "Challenge a lease term that ends before the deal pays back.",
        lambda s: s.breakeven_outside_term and 4 in s.closed,
    ),
]

LEVERS_BY_ID = {l.id: l for l in LEVERS}

# Terse answers that assert a constraint without explaining it. A real manager
# drills into these rather than advancing the ladder -- this reproduces the
# follow-up turns in the source transcript.
BARE_ANSWERS = {
    "no", "none", "nope", "no chance", "move on", "move", "not possible",
    "cant", "can't", "n/a", "no room", "no flexibility",
}


MAX_UNEXPLAINED_WORDS = 4


def _is_bare(answer: Optional[str]) -> bool:
    """
    An answer is 'bare' when it asserts something without explaining it, so the
    manager drills in before advancing.

    CORRECTED AGAINST THE SOURCE TRANSCRIPT: an earlier draft only treated flat
    refusals ("no", "move on") as bare. But in the real transcript the manager
    also drills into short unexplained ASSERTIONS -- "pricing pressure",
    "low base price" -- which name a cause without giving a reason. Both kinds
    trigger a follow-up. This is the second drafted rule the real data corrected.
    """
    if answer is None:
        return False
    a = answer.strip().lower()
    return a in BARE_ANSWERS or len(a.split()) <= MAX_UNEXPLAINED_WORDS


# ---------------------------------------------------------------------------
# Cross-cutting rules -- these apply at every turn
# ---------------------------------------------------------------------------

# NOTE (corrected against the source transcript): an earlier draft of these
# rules terminated the review after 3 consecutive refusals. The real transcript
# shows the manager pushing through ~10 straight refusals -- a refusal moves the
# conversation DOWN the ladder, it does not end it. The review ends only when no
# lever still has anything to establish. Keep this comment: it is a concrete
# example of a drafted rule corrected by real data, which the write-up cites.

TERMINAL_VERDICTS = ["approve_with_conditions", "rework_pricing", "escalate", "walk_away"]


def next_lever(state: DealState) -> Optional[Lever]:
    """
    Decide which question the manager asks next.

    Rule 1 -- never re-ask an exhausted lever (this is what 'repeat rate' measures).
    Rule 2 -- a bare answer is drilled into before the ladder advances, so
              follow-up levers are considered ahead of ordinary ones.
    Rule 3 -- otherwise take the lowest-numbered lever whose trigger fires,
              which encodes the ladder order: fix the price, then understand the
              competitor, then weigh the downside of losing.
    Rule 4 -- a refusal does not end the review; it advances the ladder. The
              review ends only when no lever still triggers.
    """
    candidates = [
        l for l in LEVERS
        if l.id not in state.closed
        and l.trigger(state)
    ]
    if not candidates:
        return None

    followups = [l for l in candidates if l.is_followup]
    if followups:
        return followups[0]
    return min(candidates, key=lambda l: l.id)


def terminal_verdict(state: DealState) -> str:
    """
    Verdict when the ladder is exhausted. Deliberately simple and stated, so a
    reviewer can see the policy rather than infer it.
    """
    if state.drp_negative and state.margin_levers_exhausted and not state.installed_base:
        return "walk_away"
    if state.drp_negative:
        return "escalate"
    if state.breakeven_outside_term:
        return "rework_pricing"
    return "approve_with_conditions"
