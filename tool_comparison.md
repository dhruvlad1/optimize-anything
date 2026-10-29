# Tool Comparison

## Task
Hand-written backpropagation for a 2-layer tanh network with softmax cross-entropy and L2 regularization (numpy only). Identical task text given to both tools. Both outputs were judged by the same 9-test file (trackA_continuous/compare/test_backprop_compare.py), which was written before either tool produced code and not edited afterwards: finite-difference gradient checks (with and without L2), loss vs an independent reference, shapes and keys, numerical stability at logits of magnitude 1000, input immutability, deterministic init, and a gradient-descent run on XOR.

## Tools compared
- Antigravity (Gemini 4.8 Flash), code saved as trackA_continuous/compare/backprop_antigravity.py
- Claude (SONET 5.5), run in a fresh chat with only the task text, code saved as trackA_continuous/compare/backprop_claude.py

## Correctness
Both passed 9/9 tests on the first attempt. No code was edited by hand.

## Hallucinations
NONE OBSERVED

## Speed
Antigravity: about 11 seconds. Claude: about 8 seconds. (Rough stopwatch times, single run.)

## Prompt iterations
Antigravity: 1 (no follow-ups). Claude: 1 (no follow-ups).

## Verdict
No measurable difference on this task. Both tools produced correct, numerically stable backpropagation on the first try. The task was small and fully specified, so it could not separate them; a harder task (for example Hessian-vector products) could. Limitations of this comparison: one task, one run per tool, and the tests were written by Claude, one of the two tools being compared, so the test design may not be neutral, although both tools were judged by the same file and its checks are independent mathematical ones.