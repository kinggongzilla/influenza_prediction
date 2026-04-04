"""
Geographic neighbors for each country.
Used to add neighboring countries' ILI as past covariates for forecasting.
"""

# Maps country name → list of neighbor country names (as they appear in extracted_data filenames)
COUNTRY_NEIGHBORS: dict[str, list[str]] = {
    "Argentina":   ["Chile", "Bolivia", "Uruguay", "Paraguay", "Brazil"],
    "Austria":     ["Germany", "Italy", "Czech Republic", "Switzerland", "Slovakia", "Hungary"],
    "Belgium":     ["France", "Germany", "Netherlands", "Luxembourg"],
    "Bulgaria":    ["Romania", "Serbia", "Greece", "Türkiye"],
    "Chile":       ["Argentina", "Peru", "Bolivia"],
    "Germany":     ["France", "Austria", "Belgium", "Netherlands", "Czech Republic", "Poland", "Denmark", "Switzerland", "Luxembourg"],
    "Italy":       ["France", "Austria", "Switzerland", "Slovenia"],
    # Broader set for other countries that may be evaluated
    "France":      ["Belgium", "Germany", "Italy", "Switzerland", "Spain", "Luxembourg"],
    "Spain":       ["France", "Portugal"],
    "Netherlands": ["Belgium", "Germany"],
    "Poland":      ["Germany", "Czech Republic", "Slovakia", "Ukraine"],
    "Switzerland": ["Germany", "France", "Italy", "Austria"],
    "Czech Republic": ["Germany", "Austria", "Slovakia", "Poland"],
    "Slovakia":    ["Czech Republic", "Austria", "Hungary", "Poland"],
    "Hungary":     ["Austria", "Slovakia", "Romania", "Serbia"],
    "Romania":     ["Hungary", "Bulgaria", "Moldova", "Ukraine"],
    "Greece":      ["Bulgaria", "Albania", "North Macedonia"],
    "Portugal":    ["Spain"],
    "Denmark":     ["Germany"],
    "Sweden":      ["Norway", "Finland", "Denmark"],
    "Norway":      ["Sweden", "Finland"],
    "Finland":     ["Sweden", "Norway"],
    "Serbia":      ["Hungary", "Romania", "Bulgaria"],
    "Brazil":      ["Argentina", "Uruguay", "Paraguay", "Bolivia"],
    "Bolivia":     ["Argentina", "Chile", "Peru", "Brazil", "Paraguay"],
    "Uruguay":     ["Argentina", "Brazil"],
    "Paraguay":    ["Argentina", "Brazil", "Bolivia"],
    "Peru":        ["Chile", "Bolivia", "Brazil"],
}


def get_neighbors(country_name: str) -> list[str]:
    """Return list of neighbor country names, or empty list if not in map."""
    return COUNTRY_NEIGHBORS.get(country_name, [])
