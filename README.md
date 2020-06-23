# Shadow Detection for Ultrasound Images Using Unlabeled Data and Synthetic Shadows
This is an official repository to reproduce the experimental results reported in [Shadow Detection for Ultrasound Images Using Unlabeled Data and Synthetic Shadows](https://openreview.net/forum?id=rJl0fYG2KN) (MIDL 2019).

## Requirements
- `docker`
- `docker-compose`
- Nvidia GPU
- Nvidia Docker 2
    - Due to limitation of `docker-compose`, need to enable the syntax `docker run --runtime=nvidia`, not `docker run --gpus`.

## Data preparation
### Training data
Place

- Images for training
- List of the training images named `image_list.txt`

at `dataset` directory.

### Testing data
Place

- Images for testing
- Label images corresponding the images above
- List of the testing images named `image_list.txt`
- List of the label images named `label_list.txt`

at `test_dataset` directory.

## Training
On directory which this repository cloned, run:

```
docker-compose up train
```

Results and trained weights will be stored in `result` directory.

## Testing
On directory which this repository cloned, run:

```
docker-compose up test
```

Results are output to stdout and `test_result` directory.
