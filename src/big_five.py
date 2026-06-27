import json
import os
import random
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import config
from openrouter import get_llm_response


@dataclass
class BigFiveProfile:
    """Score for each of the Big Five personality traits, each in [1, 5]."""
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BigFiveProfile":
        return cls(**d)


def random_bigfive_profile() -> BigFiveProfile:
    """Generate a random Big Five profile with uniformly distributed values in [1, 5]."""
    return BigFiveProfile(
        openness=round(random.uniform(1.0, 5.0), 2),
        conscientiousness=round(random.uniform(1.0, 5.0), 2),
        extraversion=round(random.uniform(1.0, 5.0), 2),
        agreeableness=round(random.uniform(1.0, 5.0), 2),
        neuroticism=round(random.uniform(1.0, 5.0), 2),
    )


# Type alias – the shape of a single assessment record.
BigFiveAssessment = dict  # {observer, speaker, round, phase, timestamp, scores: BigFiveProfile}

_NEUTRAL_PROFILE = BigFiveProfile(3.0, 3.0, 3.0, 3.0, 3.0)


def _estimate_bigfive(message: str, speaker: str, model_name: str) -> BigFiveProfile:
    """Internal: ask an LLM to estimate the speaker's Big Five traits based on one message.

    Falls back to neutral scores (3.0 each) on error.
    """
    prompt = (
        f"Analyze the following message from player {speaker} in a Mafia game.\n"
        f"Message: \"{message}\"\n\n"
        "Estimate the speaker's Big Five personality traits on a scale from 1 to 5:\n"
        "- openness  (Openness to experience)\n"
        "- conscientiousness\n"
        "- extraversion\n"
        "- agreeableness\n"
        "- neuroticism\n\n"
        "Respond ONLY with a JSON object that has exactly those five keys.\n"
        "Example: {\"openness\":3.5,\"conscientiousness\":4.0,\"extraversion\":2.5,\"agreeableness\":3.0,\"neuroticism\":2.0}\n"
        "Do not include any other text."
    )

    try:
        response = get_llm_response(model_name, prompt)
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return BigFiveProfile(
                openness=float(data.get("openness", 3.0)),
                conscientiousness=float(data.get("conscientiousness", 3.0)),
                extraversion=float(data.get("extraversion", 3.0)),
                agreeableness=float(data.get("agreeableness", 3.0)),
                neuroticism=float(data.get("neuroticism", 3.0)),
            )
    except Exception:
        pass  # fall through to fallback

    # Fallback neutral profile
    return _NEUTRAL_PROFILE


def estimate_bigfive(message: str, speaker: str, model_name: str) -> BigFiveProfile:
    """Public wrapper around the internal Big Five estimation."""
    return _estimate_bigfive(message, speaker, model_name)


# ----------------------------------------------------------------------
#  Cumulative, iteratively refined estimates (like TrustGraph)
# ----------------------------------------------------------------------

# observer -> { speaker -> BigFiveProfile }
_cumulative_profiles: Dict[str, Dict[str, BigFiveProfile]] = {}


def _compute_blended_profile(old: BigFiveProfile, new: BigFiveProfile, factor: float) -> BigFiveProfile:
    """Blend two profiles using exponential moving average with the given factor."""
    return BigFiveProfile(
        openness=round(old.openness * (1 - factor) + new.openness * factor, 2),
        conscientiousness=round(old.conscientiousness * (1 - factor) + new.conscientiousness * factor, 2),
        extraversion=round(old.extraversion * (1 - factor) + new.extraversion * factor, 2),
        agreeableness=round(old.agreeableness * (1 - factor) + new.agreeableness * factor, 2),
        neuroticism=round(old.neuroticism * (1 - factor) + new.neuroticism * factor, 2),
    )


def update_bigfive_for_speaker(
    observer: str,
    speaker: str,
    message: str,
    model_name: str,
    blend_factor: float = 0.3
) -> BigFiveProfile:
    """
    Refine the observer's cumulative estimate of the speaker's Big Five traits
    using the given message.  Returns the updated cumulative profile.
    """
    global _cumulative_profiles
    previous = _cumulative_profiles.get(observer, {}).get(speaker)
    if previous is None:
        previous = _NEUTRAL_PROFILE

    raw = _estimate_bigfive(message, speaker, model_name)
    new_cum = _compute_blended_profile(previous, raw, blend_factor)

    if observer not in _cumulative_profiles:
        _cumulative_profiles[observer] = {}
    _cumulative_profiles[observer][speaker] = new_cum

    return new_cum


def get_cumulative_profile(observer: str, speaker: str) -> Optional[BigFiveProfile]:
    """
    Return the observer's current cumulative Big Five estimate for the speaker,
    or None if no estimate has been made yet.
    """
    return _cumulative_profiles.get(observer, {}).get(speaker)


# ----------------------------------------------------------------------
#  Persistent registry (loaded once before first game, saved after each)
# ----------------------------------------------------------------------
_registry_profiles: Dict[str, BigFiveProfile] = {}
_registry_assessments: Dict[str, list] = {}


def load_registry(path: str = None) -> None:
    """Load persisted Big Five profiles, assessments, and cumulative estimates from a JSON file."""
    global _registry_profiles, _registry_assessments, _cumulative_profiles
    if path is None:
        path = config.BIGFIVE_REGISTRY_FILE
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _registry_profiles = {}
        for name, prof_dict in data.get("profiles", {}).items():
            _registry_profiles[name] = BigFiveProfile.from_dict(prof_dict)
        _registry_assessments = data.get("assessments", {})

        _cumulative_profiles = {}
        cum_data = data.get("cumulative_profiles", {})
        for observer, speakers_dict in cum_data.items():
            _cumulative_profiles[observer] = {}
            for speaker, prof_dict in speakers_dict.items():
                _cumulative_profiles[observer][speaker] = BigFiveProfile.from_dict(prof_dict)
    else:
        _registry_profiles = {}
        _registry_assessments = {}
        _cumulative_profiles = {}


def save_registry(path: str = None) -> None:
    """Persist current Big Five profiles, assessments, and cumulative estimates to a JSON file."""
    if path is None:
        path = config.BIGFIVE_REGISTRY_FILE
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    profiles_ser = {
        name: prof.to_dict() for name, prof in _registry_profiles.items()
    }
    cum_ser = {}
    for observer, speakers in _cumulative_profiles.items():
        cum_ser[observer] = {speaker: prof.to_dict() for speaker, prof in speakers.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "profiles": profiles_ser,
                "assessments": _registry_assessments,
                "cumulative_profiles": cum_ser,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def get_profile(player_name: str) -> BigFiveProfile:
    """Return the permanent Big Five profile for a player name, creating a random one if needed."""
    if player_name not in _registry_profiles:
        _registry_profiles[player_name] = random_bigfive_profile()
    return _registry_profiles[player_name]


def add_assessments(observer: str, assessments_list: list) -> None:
    """Append newly created assessment records under a specific observer."""
    if observer not in _registry_assessments:
        _registry_assessments[observer] = []
    _registry_assessments[observer].extend(assessments_list)
