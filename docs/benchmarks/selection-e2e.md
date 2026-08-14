# Structured Selection E2E — 2026-08-14

The public 81-image evaluation album was indexed and embedded with the M1/M2
CPU providers. The following M3 prompt was then sent to `SelectionService`:

> 选 12 张夜景 night，质量至少 45，相似组最多 1 张

## Result

| Metric | Value |
| --- | ---: |
| Solver | `ortools-cp-sat` |
| Solver status | `optimal` |
| Eligible candidates | 77 |
| Requested / returned | 12 / 12 |
| Duration | 698 ms |
| Selected auto-rejects | 0 |
| Minimum selected quality | 48.635 |
| Distinct selected group keys | 12 |

The returned set contains source titles such as New York City at night HDR,
Wellington City Night, Night time in a city, Night Photography City, San
Francisco City Hall at nighttime, Gothenburg City Theatre at night, UB City at
night, and Church of the Virgin of the Burgh Rhodes at night. The lightweight
semantic provider also admits several general city images; this remains a
ranking-quality limitation, not a hard-constraint violation.

A second prompt, `Select 20 photos of architecture, maximum 1 per similarity
group`, returned 20 photos with four grouped photos and a maximum observed count
of exactly one for every selected similarity group. This supplies a positive
group-cap case in addition to the synthetic regression test.

## HTTP verification

A real Uvicorn process handled `POST /selections` against the same cached public
album. It returned HTTP success, `feasible=true`, 12 selected photos from 77
candidates, `ortools-cp-sat / optimal`, minimum quality 48.635, and a 700 ms
service duration. The selection and parsed constraint JSON were persisted in
SQLite.

## Hard-constraint gates

- Target count is an equality constraint, not a best-effort hint.
- Suggested rejects and photos below the quality floor are removed before
  optimization.
- Every similarity group has an explicit capacity constraint.
- If the available group capacities cannot reach the target, the result is
  infeasible and contains no silently truncated partial selection.
- A requested concept unsupported by the active semantic provider returns an
  error instead of silently switching to quality-only ranking.
- Pure cardinality prompts such as `选 2 张` may use quality-only ranking and
  include an explicit warning.

OR-Tools is optional. The fallback solver produces the same optimum for the
current partition-capacity constraint family; more general future constraints
will require CP-SAT.
