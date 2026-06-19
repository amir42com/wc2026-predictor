"""
Canonical team-name normalization — single source of truth.

Maps every known alias to ONE canonical name (the app/group + source-data
spelling). Three alias families are covered:

  * football-data.org API names (the live tracker feed),
  * source-data / accent / abbreviation variants,
  * app/group spellings (already canonical; included so canonical() is idempotent).

Route every team-name lookup through `canonical()`: features.py's confederation
lookup, the tracker's API-name mapping, and (validated in tests) simulate.GROUPS.
This is what stops a Curaçao-style accented fall-through where one spelling
silently resolves to confederation "Other" or to a missing model state.
"""

# alias -> canonical
ALIASES: dict[str, str] = {
    # football-data.org API names
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR":           "DR Congo",
    "Czechia":            "Czech Republic",
    "IR Iran":            "Iran",
    "Korea Republic":     "South Korea",
    "Republic of Korea":  "South Korea",
    "USA":                "United States",
    # source-data / accent / abbreviation variants
    "Curacao":            "Curaçao",
    "UAE":                "United Arab Emirates",
}


def canonical(name: str | None) -> str | None:
    """Return the canonical team name for any known alias (idempotent)."""
    if name is None:
        return None
    return ALIASES.get(name, name)
