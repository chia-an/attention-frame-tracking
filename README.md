# Dialogue frame tracking with attention

## Data
1. Download the datasets [MultiWOZ2](http://dialogue.mi.eng.cam.ac.uk/index.php/corpus/) and [FRAMES](https://datasets.maluuba.com/Frames/dl) to the directory `../data/`.
2. Call `create_dictionaries()` in `utils/parse_frames.py` and `utils/parse_multiwoz.py`.
3. Call `split_dataset()` in `utils/parse_frames.py` to do the 10-fold split on FRAMES.
4. Call `generate_mix_datasets()` in `utils/mix_multiwoz.py`. Comment / uncomment different parts to change the setting of generation.
5. Call `split_dataset(DATASET_NAME)` in `utils/mix_multiwoz.py` to split the generated synthetic dataset.

## Training
Pre-compute the BERT feature using the script `utils/bert_feature.py`.
- `train.py`: trains a model both from scratch and with transfer learning.
- `train-transfer.py`: trains a model with different pre-training dataset.
- `hyper.py`: is the script for hyperparameter tuning.

## Results
The three training scripts generate a log file in `./runs` when they run. There is also another log file per model and training. Run `parse_log.py` to get a summary of the results.

## Notes
- Run the command `$export PYTHONPATH=<PATH_OF_ROOT_OF_THE_REPO>`.
- Create directories `./runs` and `./model/model`.
