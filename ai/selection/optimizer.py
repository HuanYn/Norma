from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OptimizationCandidate:
    index: int
    score: float
    group_key: str


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    indices: list[int]
    solver: str
    status: str


def optimize_collection(
    candidates: list[OptimizationCandidate],
    target_count: int,
    max_per_group: int,
) -> OptimizationResult:
    capacities = _capacity(candidates, max_per_group)
    if sum(capacities.values()) < target_count:
        return OptimizationResult([], _solver_name(), "infeasible")

    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return _greedy(candidates, target_count, max_per_group)

    model = cp_model.CpModel()
    variables = [
        model.new_bool_var(f"photo_{candidate.index}") for candidate in candidates
    ]
    model.add(sum(variables) == target_count)
    groups: dict[str, list[int]] = {}
    for position, candidate in enumerate(candidates):
        groups.setdefault(candidate.group_key, []).append(position)
    for positions in groups.values():
        model.add(sum(variables[position] for position in positions) <= max_per_group)

    objective = []
    count = len(candidates)
    for position, candidate in enumerate(candidates):
        primary = round(candidate.score * 1_000_000)
        deterministic_tie_break = count - position
        objective.append(
            (primary * (count + 1) + deterministic_tie_break) * variables[position]
        )
    model.maximize(sum(objective))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2.0
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        return OptimizationResult(
            [], "ortools-cp-sat", solver.status_name(status).lower()
        )
    selected = [
        candidate.index
        for position, candidate in enumerate(candidates)
        if solver.value(variables[position])
    ]
    return OptimizationResult(
        selected, "ortools-cp-sat", solver.status_name(status).lower()
    )


def _greedy(
    candidates: list[OptimizationCandidate], target_count: int, max_per_group: int
) -> OptimizationResult:
    selected: list[int] = []
    group_counts: dict[str, int] = {}
    ranked = sorted(candidates, key=lambda item: (-item.score, item.index))
    for candidate in ranked:
        if group_counts.get(candidate.group_key, 0) >= max_per_group:
            continue
        selected.append(candidate.index)
        group_counts[candidate.group_key] = group_counts.get(candidate.group_key, 0) + 1
        if len(selected) == target_count:
            break
    return OptimizationResult(selected, "deterministic-partition-greedy", "optimal")


def _capacity(
    candidates: list[OptimizationCandidate], max_per_group: int
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for candidate in candidates:
        sizes[candidate.group_key] = sizes.get(candidate.group_key, 0) + 1
    return {key: min(size, max_per_group) for key, size in sizes.items()}


def _solver_name() -> str:
    try:
        import ortools  # noqa: F401
    except ImportError:
        return "deterministic-partition-greedy"
    return "ortools-cp-sat"
