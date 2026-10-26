# Image-model arithmetic

Research page for **Can Generative Image Models Reason Through Each Other’s Outputs?** by Dhruv Saini.

The page contains the full preprint, experiment results, interactive model-pair matrices, saved digit images, and the original learning-curve and held-out-example figures.

## Local preview

Run `python3 -m http.server 8000` from this directory and open `http://localhost:8000/`. The matrix loads `data/results.json`, so serve the page over HTTP.

## Contents

- `paper/`: full preprint prepared October 3, 2026.
- `data/results.json`: values from the preprint’s modulo-23 table, binary model-pair figure, and multiplication table. Includes the paper’s SHA-256 hash.
- `data/figure-sources.json`: hashes and source identities for the original research images.
- `study/`: experiment source from the prepared supplementary package.
- `assets/`: local styles, JavaScript, and research figures.

The modulo-23 result of 210/215 is conditional on a correct first drawing and success with a clean digit in the second problem. The complete different-model two-step result is 210/276. The page keeps those denominators separate.

Template credits and licenses are listed in `THIRD_PARTY.md`.
