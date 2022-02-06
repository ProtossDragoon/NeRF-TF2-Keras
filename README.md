# NeRF

NeRF Tensorflow2, Keras implementation

## Quickstart

### 1. Download dataset

```bash
./download_example_data.bash
```

### 2. Install dependencies

`virtual environment`
```bash
python3 -m venv ~/dev/tf2
source ~/dev/tf2/bin/activate
```


`requirements`
```bash
pip install -r requirements.txt
```

### 3. Run unit tests

`tests`
```bash
python3 sampledata_loader_test.py
python3 train_test.py
python3 -m nerf_core.ray_test
python3 -m utils.videovis_test
```

- If there aren't any error, go to the next step.
- When you found a bug, please raise an github issue.

### 4. Run demo

```bash
python3 train.py
```

## Results

<table>
<thead align="center">
  <tr>
    <th></th>
    <th>Trainig</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>20 epochs<br>32 samples per ray</td>
    <td colspan="2" align="center"><img src=https://github.com/ProtossDragoon/NeRF-TF2-Keras/blob/master/docs/training.gif alt="20epoch_32samples"></td>
  </tr>
  <tr>
    <td>80 epochs<br>64 samples per ray</td>
    <td colspan="2" align="center"><img src=https://github.com/ProtossDragoon/NeRF-TF2-Keras/blob/master/docs/training_2.gif alt="80epoch_64samples"></td>
  </tr>
</tbody>
</table>

<table>
<thead align="center">
  <tr>
    <th></th>
    <th>Visualization</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>40 epochs<br>32 samples per ray</td>
    <td colspan="2" align="center"><img src=https://github.com/ProtossDragoon/NeRF-TF2-Keras/blob/master/docs/result.gif alt="40epoch_32samples" width="300" height="300"></td>
  </tr>
</tbody>
</table>

## Note

### 1. File Structure

- `FILLME`

### 2. Implementataion

- `FILLME`

### 3. Test Environments

- 19' MacBook Pro 16inch, 16GB
- Google COLAB GPU Runtime

## References

- [Keras example (TensorFlow2+keras, COLAB notebook)](https://keras.io/examples/vision/nerf/)
- [Official implementation (TensorFlow1)](https://github.com/bmild/nerf)
