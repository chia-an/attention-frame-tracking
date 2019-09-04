# Dialogue frame tracking with attention

This repository contains the code for the project **Embedding Memory into Advanced Dialogue Management**. Follow the instructions below to reproduce experimental results. Please look at the [report](https://github.com/chia-an/attention-frame-tracking/blob/master/report/report.pdf) for more details.

## Dependency
- PyTorch 1.2.0
- Other dependencies are in `requirements.txt`

## Data and preprocessing
1. Download the datasets [MultiWOZ2](http://dialogue.mi.eng.cam.ac.uk/index.php/corpus/) and [FRAMES](https://datasets.maluuba.com/Frames/dl) to the directory `../data/`.
2. In the directory `./utils`, call `create_dictionaries()` in `utils/parse_frames.py` and `utils/parse_multiwoz.py`.
```
$ python3 -c "import parse_frames as m; m.create_dictionaries()"
$ python3 -c "import parse_multiwoz as m; m.create_dictionaries()"
```
3. In the directory `./utils`, call `split_dataset()` in `utils/parse_frames.py` to do the 10-fold split on FRAMES.
```
$ python3 -c "import parse_frames as m; m.split_dataset()"
```
4. Move the synthetic datasets used in the project from the directory `./synthetic` to `../data/MULTIWOZ2 2`. If you want to generate your own dataset, in the directory `./utils`, call `generate_mixed_dataset()` in `utils/mix_multiwoz.py`. Comment / uncomment different parts to change the setting of generation.
```
$ python3 -c "import mix_multiwoz as m; m.generate_mixed_dataset()"
```
5. In the directory `./utils`, call `split_dataset(DATASET_NAME)` in `utils/mix_multiwoz.py` to split the synthetic dataset.
```
$ python3 -c "import mix_multiwoz as m; m.split_dataset(DATASET_NAME)"
```

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
