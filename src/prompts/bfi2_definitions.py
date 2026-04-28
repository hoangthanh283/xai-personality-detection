"""BFI-2 trait + facet definitions used by EPR-S / EPR-T prompts.

Source: Soto & John (2017) — "The Next Big Five Inventory (BFI-2): Developing
and assessing a hierarchical model with 15 facets to enhance bandwidth, fidelity,
and predictive power." Journal of Personality and Social Psychology 113(1).

Item phrasings are paraphrased to avoid copyright issues; semantic content
preserved.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Facet:
    name: str
    description: str
    high_items: str  # comma-separated paraphrased BFI-2 high-pole items
    low_items: str   # comma-separated paraphrased BFI-2 low-pole items


@dataclass(frozen=True)
class BigFiveDim:
    code: str          # O, C, E, A, N
    name: str          # full name
    definition: str    # 1-2 sentence definition
    facets: tuple[Facet, Facet, Facet]


# ── BFI-2 dimensions ─────────────────────────────────────────────────────────
OPENNESS = BigFiveDim(
    code="O",
    name="Openness",
    definition=(
        "Openness measures the breadth, depth, and complexity of an individual's "
        "mental and experiential life. It captures intellectual curiosity, "
        "aesthetic sensitivity, and creative imagination."
    ),
    facets=(
        Facet(
            name="Intellectual Curiosity",
            description=(
                "Tendency to be curious about many different things, complex, "
                "deep thinker."
            ),
            high_items=(
                "is curious about many different things, is complex / a deep "
                "thinker, has intellectual interests"
            ),
            low_items=(
                "avoids intellectual or philosophical discussions, has little "
                "interest in abstract ideas"
            ),
        ),
        Facet(
            name="Aesthetic Sensitivity",
            description=(
                "Tendency to be fascinated by art, music, or literature; "
                "values art and beauty."
            ),
            high_items=(
                "is fascinated by art / music / literature, values art and beauty"
            ),
            low_items=(
                "has few artistic interests, thinks poetry and plays are boring"
            ),
        ),
        Facet(
            name="Creative Imagination",
            description=(
                "Tendency to be inventive, find clever ways to do things; is "
                "original, comes up with new ideas."
            ),
            high_items=(
                "is inventive, finds clever ways to do things, is original / "
                "comes up with new ideas"
            ),
            low_items=(
                "has little creativity, has difficulty imagining things"
            ),
        ),
    ),
)


CONSCIENTIOUSNESS = BigFiveDim(
    code="C",
    name="Conscientiousness",
    definition=(
        "Conscientiousness measures the tendency to be organized, dependable, "
        "disciplined, and goal-oriented; driven by accomplishment and duty."
    ),
    facets=(
        Facet(
            name="Organization",
            description=(
                "Tendency to be systematic, like to keep things in order, neat "
                "and tidy."
            ),
            high_items=(
                "is systematic / likes to keep things in order, keeps things "
                "neat and tidy"
            ),
            low_items=(
                "tends to be disorganized, leaves a mess / doesn't clean up"
            ),
        ),
        Facet(
            name="Productiveness",
            description=(
                "Tendency to be efficient, get things done, persistent until "
                "task is finished."
            ),
            high_items=(
                "is efficient / gets things done, is persistent / works until "
                "task is finished"
            ),
            low_items=(
                "tends to be lazy, has difficulty getting started on tasks"
            ),
        ),
        Facet(
            name="Responsibility",
            description=(
                "Tendency to be dependable, steady, reliable; can always be "
                "counted on."
            ),
            high_items=(
                "is dependable / steady, is reliable / can always be counted on"
            ),
            low_items=(
                "can be somewhat careless, sometimes behaves irresponsibly"
            ),
        ),
    ),
)


EXTRAVERSION = BigFiveDim(
    code="E",
    name="Extraversion",
    definition=(
        "Extraversion measures the tendency to be sociable, energetic, "
        "outgoing, and assertive; seeks stimulation in the company of others."
    ),
    facets=(
        Facet(
            name="Sociability",
            description="Tendency to be outgoing, sociable, talkative.",
            high_items="is outgoing / sociable, is talkative",
            low_items="tends to be quiet, is sometimes shy / introverted",
        ),
        Facet(
            name="Assertiveness",
            description=(
                "Tendency to have an assertive personality, dominant, acts as a "
                "leader."
            ),
            high_items=(
                "has an assertive personality, is dominant / acts as a leader"
            ),
            low_items=(
                "finds it hard to influence people, prefers to have others take "
                "charge"
            ),
        ),
        Facet(
            name="Energy Level",
            description=(
                "Tendency to be full of energy, enthusiastic; excited / eager."
            ),
            high_items="is full of energy, shows a lot of enthusiasm",
            low_items=(
                "rarely feels excited / eager, is less active than other people"
            ),
        ),
    ),
)


AGREEABLENESS = BigFiveDim(
    code="A",
    name="Agreeableness",
    definition=(
        "Agreeableness measures the tendency to be compassionate, cooperative, "
        "trusting, and helpful towards others; values interpersonal harmony."
    ),
    facets=(
        Facet(
            name="Compassion",
            description=(
                "Tendency to be compassionate, soft-hearted, helpful and "
                "unselfish with others."
            ),
            high_items=(
                "is compassionate / has a soft heart, is helpful and unselfish "
                "with others"
            ),
            low_items=(
                "feels little sympathy for others, can be cold and uncaring"
            ),
        ),
        Facet(
            name="Respectfulness",
            description=(
                "Tendency to be respectful, polite, courteous to others."
            ),
            high_items=(
                "is respectful / treats others with respect, is polite / "
                "courteous to others"
            ),
            low_items=(
                "starts arguments with others, is sometimes rude to others"
            ),
        ),
        Facet(
            name="Trust",
            description=(
                "Tendency to have a forgiving nature, assume the best about "
                "people."
            ),
            high_items=(
                "has a forgiving nature, assumes the best about people"
            ),
            low_items=(
                "tends to find fault with others, is suspicious of others' "
                "intentions"
            ),
        ),
    ),
)


NEUROTICISM = BigFiveDim(
    code="N",
    name="Neuroticism",
    definition=(
        "Neuroticism measures the tendency to experience negative emotions "
        "(anxiety, depression, emotional volatility); the opposite pole is "
        "emotional stability."
    ),
    facets=(
        Facet(
            name="Anxiety",
            description="Tendency to be tense, worried, anxious.",
            high_items="can be tense, worries a lot",
            low_items=(
                "is relaxed / handles stress well, rarely feels anxious or afraid"
            ),
        ),
        Facet(
            name="Depression",
            description=(
                "Tendency to often feel sad, depressed, blue."
            ),
            high_items="often feels sad, tends to feel depressed / blue",
            low_items=(
                "stays optimistic after experiencing a setback, feels secure / "
                "comfortable with self"
            ),
        ),
        Facet(
            name="Emotional Volatility",
            description=(
                "Tendency to be moody, have up and down mood swings, get "
                "emotional easily."
            ),
            high_items=(
                "is moody / has up and down mood swings, is temperamental / "
                "gets emotional easily"
            ),
            low_items=(
                "is emotionally stable / not easily upset, keeps emotions under "
                "control"
            ),
        ),
    ),
)


# ── Lookup ───────────────────────────────────────────────────────────────────
DIMS: dict[str, BigFiveDim] = {
    "O": OPENNESS,
    "C": CONSCIENTIOUSNESS,
    "E": EXTRAVERSION,
    "A": AGREEABLENESS,
    "N": NEUROTICISM,
}


def get_dim(code: str) -> BigFiveDim:
    """Get BigFiveDim by code (O / C / E / A / N) or by full name (case-insensitive)."""
    code_upper = code.upper()
    if code_upper in DIMS:
        return DIMS[code_upper]
    # Try by full name
    for dim in DIMS.values():
        if dim.name.lower() == code.lower():
            return dim
    raise KeyError(f"Unknown Big Five dimension: {code!r}")


def to_template_dict(code: str) -> dict:
    """Render a BigFiveDim into the dict shape consumed by the Jinja2 templates."""
    dim = get_dim(code)
    return {
        "target_dim": dim.name,
        "dim_definition": dim.definition,
        "facets": [
            {
                "name": f.name,
                "description": f.description,
                "high_items": f.high_items,
                "low_items": f.low_items,
            }
            for f in dim.facets
        ],
    }
