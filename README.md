# 19S-Thesis

## Data
The directory is `../data/`.

List of datasets:
  - [MultiWOZ2](http://dialogue.mi.eng.cam.ac.uk/index.php/corpus/)
  - [Frames](https://datasets.maluuba.com/Frames/dl)

## Notes
- Run the command `$export PYTHONPATH=<PATH_OF_ROOT_OF_THE_REPO>`.
- Call the function `generate_mixed_dataset()` in `utils/mix_multiwoz.py` to generate the synthetic dataset.
- The accuracy computed for FRAMES is not accurate, the true accuracy is usually about ten pertentage points more. To compute the correct accuracy, call the `test()` functions in `train_*.py` scripts to generate a json with the predictions `gen_<MODEL_FILENAME>.json`, then use the command `$frametracking-evaluate eval <PATH_TO_GROUND_TRUTH_FILE> <PATH_TO_PREDICTION_JSON> <FOLD>` to get the accuracy.  `frametracking-evaluate` is from the submodule `frames`.

