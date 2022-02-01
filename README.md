# NeRF

NeRF Tensorflow2, Keras implementation

## Quickstart

### 1. Download dataset

```
./download_example_data.bash
```

### 2. Install dependencies

`virtual environment`
```
python3 -m venv ~/dev/tf2
source ~/dev/tf2/bin/activate
```


`requirements`
```
pip install -r requirements.txt
```

### 3. Run unit tests

`tests`
```
python3 simple_dataloader_test.py
python3 train_test.py
python3 -m nerf_core.ray_test
```

- If there aren't any error, go to the next step.
- When you found a bug, please raise an github issue.

### 4. Run demo

```
python3 train.py
```

## Results

*20 epoch, 32 samples per ray*
![40epoch_32samples](https://github.com/ProtossDragoon/NeRF-TF2-Keras/blob/master/docs/training.gif)

*80 epoch, 64 samples per ray*
![80epoch_64samples](https://github.com/ProtossDragoon/NeRF-TF2-Keras/blob/master/docs/training_2.gif)

## Note

### 1. File Structure

- `FILLME`

### 2. Implementataion

- `FILLME`

### 3. Test Environments

- 19' MacBook Pro 16inch, 16GB

## References

- [Keras example (TensorFlow2+keras, COLAB notebook)](https://keras.io/examples/vision/nerf/)
- [Official implementation (TensorFlow1)](https://github.com/bmild/nerf)
