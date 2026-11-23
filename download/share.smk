"""Let modules that share a `<downloads>` folder download each file once.

Include this file after the last `use rule` statement of a workflow importing several modules
whose `download` wrapper rules write to the same paths.
Rules writing the same files are put in a rule order, following the order the modules were imported in:
the rule of the first module downloads the file for all of them.

The modules must resolve the shared dataset to the same registry entry, apart from its
`description` and default `version`. Otherwise, the workflow fails, listing the differences:
set `downloads.<dataset>` in the configuration of those modules so that they match.
"""

import yaml
from snakemake.rules import Rule

SHARED_DOWNLOAD_IGNORED_KEYS = {"description", "version"}


def _shared_download_rules() -> list[list[Rule]]:
    """Download rules writing the same files, in the order they were imported."""
    groups: dict[tuple[str, ...], list[Rule]] = {}
    for rule in workflow.rules:
        if "_download_" in rule.name and rule.output:
            groups.setdefault(tuple(map(str, rule.output)), []).append(rule)
    return [rules for rules in groups.values() if len(rules) > 1]


def _shared_download_entry(rule: Rule) -> dict:
    """The registry entry of a download rule's dataset, as resolved by its module."""
    entry = rule.module_globals["config"]["downloads"][rule.params.dataset]
    return {
        key: value
        for key, value in entry.items()
        if key not in SHARED_DOWNLOAD_IGNORED_KEYS
    }


def _check_shared_download_entries(rules: list[Rule]) -> None:
    """Fail if the modules of rules writing the same files resolve different registry entries."""
    entries = {rule.name: _shared_download_entry(rule) for rule in rules}
    keys = sorted({key for entry in entries.values() for key in entry})
    differing = [
        key
        for key in keys
        if len({yaml.safe_dump(entry.get(key)) for entry in entries.values()}) > 1
    ]
    if not differing:
        return
    dataset = rules[0].params.dataset
    details = {
        name: {key: entry.get(key) for key in differing}
        for name, entry in entries.items()
    }
    raise ValueError(
        f"Rules {list(entries)} download the same files of dataset {dataset!r}, "
        f"but their modules' registries differ in {differing}:\n"
        f"{yaml.safe_dump(details, sort_keys=False)}"
        f"Set `downloads.{dataset}` in the configuration of these modules so that they match."
    )


for _rules in _shared_download_rules():
    _check_shared_download_entries(_rules)
    workflow.ruleorder(*(rule.name for rule in _rules))
