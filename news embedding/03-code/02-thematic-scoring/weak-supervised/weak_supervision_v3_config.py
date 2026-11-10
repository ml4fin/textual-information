# weak_supervision_v3_config.py

"""
Frozen configuration for Weak Supervision V3.0.

Development set:
    485 manually reviewed article-theme observations.

This configuration was frozen after development-set tuning.
The rules and signal dictionaries must not be modified using the same
development observations.
"""


VERSION = "Weak Supervision V3.0 - Frozen Development Version"

PROXIMITY_WINDOW = 50


# =============================================================================
# SINGLE-THEME SIGNALS
# =============================================================================

# Strong signals provide direct theme-specific lexical evidence.
STRONG_SIGNALS = {
    "Climate / Environmental Risk": {
        "carbon emissions",
        "wildfire",
        "wildfires",
        "extreme weather",
        "global warming",
        "climate change",
        "flooding",
        "floods",
    },

    "Cybersecurity Risk": {
        "cyberattack",
        "cyberattacks",
        "hackers",
    },

    "Pandemic Risk": {
        "coronavirus",
        "covid-19",
        "covid",
    },

    "Supply Chain Risk": {
        "supply chain",
        "supply chains",
    },
}


# Ambiguous signals require additional contextual evidence in Weak V3.
AMBIGUOUS_SIGNALS = {
    "Climate / Environmental Risk": {
        "drought",
        "flood",
        "heatwave",
    },

    "Cybersecurity Risk": {
        "hacked",
        "hacking",
        "hacker",
        "cybersecurity",
    },

    "Pandemic Risk": {
        "pandemic",
        "epidemic",
    },

    "Supply Chain Risk": set(),
}


# =============================================================================
# NO-THEME-SIGNAL PROXIMITY RULES
# =============================================================================

CLIMATE_THEME_SIGNALS = [
    "emission",
    "emissions",
    "carbon",
    "pollution",
    "hurricane",
    "hurricanes",
    "storm",
    "storms",
    "weather",
    "climate",
]


CYBER_OBJECT_SIGNALS = [
    "computer",
    "computers",
    "device",
    "devices",
    "software",
    "internet",
    "phone",
    "phones",
    "iphone",
]


CYBER_CONTEXT_SIGNALS = [
    "security",
    "privacy",
    "vulnerability",
    "vulnerabilities",
    "fraud",
    "unauthorized access",
    "unauthorised access",
    "security incident",
    "security incidents",
]


SUPPLY_OBJECT_SIGNALS = [
    "port",
    "ports",
    "oil",
    "food",
    "refinery",
    "refineries",
    "energy",
    "freight",
    "shipping",
    "delivery",
    "deliveries",
]


SUPPLY_DISRUPTION_SIGNALS = [
    "disruption",
    "disruptions",
    "disrupted",
    "shutdown",
    "shut",
    "shortage",
    "shortages",
    "recall",
    "blackout",
    "blackouts",
]


# =============================================================================
# AI-RELATED RISK
# =============================================================================

# AI relevance requires an AI-theme mention in sufficiently close proximity
# to at least one adverse-context signal.
AI_STRICT_SIGNALS = [
    # Governance and regulation
    "regulator",
    "regulators",
    "regulation",
    "regulated",
    "rules",
    "law",
    "laws",
    "legal",

    # Intellectual property and disputes
    "copyright",
    "copyrighted",
    "lawsuit",
    "lawsuits",
    "sue",
    "sues",
    "sued",
    "trade secret",
    "trade secrets",
    "stealing",
    "stolen",

    # Accidents and safety
    "crash",
    "crashes",
    "collision",
    "collisions",
    "accident",
    "accidents",
    "safety",
    "unsafe",

    # Abuse and harmful content
    "extremist",
    "extremism",
    "hateful",
    "offensive",
    "abuse",
    "misuse",

    # Explicit risk and concern
    "concern",
    "concerns",
    "fear",
    "fears",
    "panic",
    "danger",
    "dangerous",
    "risk",
    "risks",
    "threat",
    "threats",

    # Autonomous weapons
    "weapon",
    "weapons",
    "killer",
    "lethal",

    # Intervention and shutdown
    "shut down",
    "shutdown",
    "suspended",
    "suspension",
    "investigate",
    "investigation",

    # Competition and governance
    "antitrust",
]