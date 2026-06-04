# AGENTS.md

This directory is an independent learn-by-doing research artifact for activation checkpointing.

Scope:
- Do not inherit implementation or design choices from sibling research directories.
- Keep the Python implementation small enough to teach, but complete enough to run real autograd tests.
- Prefer source-grounded explanations from PyTorch docs/source when describing production behavior.
- The website should be a real interactive lesson, not a landing page. Use code-native SVG/HTML/CSS/JS for deterministic diagrams.

Verification expectations:
- Run the Python tests on CPU.
- If MPS is available, the same tests should exercise MPS paths.
- Open the static site locally and inspect desktop and mobile layouts before handoff.
