# Experiment source

Source accompanying **Can Generative Image Models Reason Through Each Other’s Outputs?** by Dhruv Saini, retained from the prepared supplementary package.

- `src/image_reasoning/symmetric_segment_mod23.py`: the scratch-trained modulo-23 image model and held-out addition experiment.
- `scripts/mac_sana_column/`: SANA column-addition preparation, training, evaluation and independent reading.
- `scripts/mac_sana_column_followup/`: binary arithmetic, model exchanges and repeated calculations.
- `scripts/mac_sana_iclr/`: separately trained study models and binary operations.
- `configs/`: experiment configurations.

This directory contains experiment code. Model checkpoints and complete generated-image archives are not included in the website repository. The paper describes architectures, training schedules, image coordinates and digit-reading rules; the website’s `data/` directory contains the displayed aggregate results.
