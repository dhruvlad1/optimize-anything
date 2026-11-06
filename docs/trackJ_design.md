# Track J: Test-suite minimization

## Problem and formulation

Large test suites increase continuous-integration time. This experiment selects a smaller set of test cases while preserving every line and branch element exercised by the full suite. For each test `t`, choose `x_t ∈ {0,1}` and minimize `Σ_t c_t x_t`, where `c_t` is the median measured test runtime. For each line or branch element `e` covered by the full suite, require `Σ_{t covers e} x_t ≥ 1`. This is weighted set cover, an NP-hard problem; the greedy algorithm has an `ln(n)+1` approximation guarantee, while the integer program yields an optimum when proven optimal.

The methods are greedy selection by newly covered elements per second; an OR-Tools SCIP integer program cross-checked against `scipy.optimize.milp`; and baselines consisting of the full suite, 20 random subsets with the ILP subset size, the first `K` tests in collection/file order, and fastest tests until coverage is complete. A second solve uses `c_t=1` to minimize the number of tests.

The correctness cost is measured with hand-generated single-change mutants: among mutants killed by the full suite, measure the fraction still killed by each subset. The baseline profile reports the slowest cases and runtime share by test file. Coverage preservation is not fault detection. Mutants are synthetic; equivalent or undetected mutants are excluded from the subset score. Timing uses three repetitions on one machine.

## Limitations

Individual test runtimes are quantised at pytest's 10 ms reporting resolution, so the floored per-test runtime objective is approximate; it also excludes fixed pytest startup and collection overhead.

## Run status and results

The target is pinned to upstream `grantjenks/python-sortedcontainers` commit `3ac3586`. No target checkout was available when this experiment was prepared. The setup and analysis scripts are present, but results and the measured gain/correctness tradeoff are **not measured** until the pinned target is available and the scripts complete. Do not interpret the design or solver tests as an experiment result.
