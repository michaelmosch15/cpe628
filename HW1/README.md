# CPE 628 - HW1: Testing TensorFlow and PyTorch in Colab

For this homework I ran the given TensorFlow code and the equivalent PyTorch code on two tasks: linear regression on California Housing and digit classification on MNIST. After getting the given code working I changed some hyperparameters to see what actually makes a difference. Screenshots are in the `screenshots/` folder.

## Files

| File | What it is |
|---|---|
| `HW1_TensorFlow.ipynb` | Given TF/Keras code for both tasks + my tuning experiments |
| `HW1_PyTorch.ipynb` | Given PyTorch code for both tasks + my tuning experiments |
| `screenshots/` | Screenshots of the outputs and plots |


## How the code works

### Task 1: Linear regression (California Housing)

- Data: 20,640 California districts from the 1990 census, 8 features (median income, house age, avg rooms, avg bedrooms, population, avg occupancy, latitude, longitude). The target is median house value in units of $100k.
- Split: 80% train / 20% test, and then 20% of the train set is used as validation during training.
- Scaling: `StandardScaler` makes every feature mean 0 / std 1. It's fit on the training data only so nothing from the test set leaks in. This matters here because the features are on really different scales (population is in the thousands, avg bedrooms is around 1).
- Model: one `Dense(1)` layer with no activation, so it's just `y = w1*x1 + ... + w8*x8 + b` (9 parameters). So it's plain linear regression, just trained with gradient descent instead of solved directly.
- Training: Adam, lr = 0.01, MSE loss, batch size 64, 60 epochs. MAE is also tracked because it's easier to read (MAE of 0.53 = off by about $53k on average).

### Task 2: MNIST classification

- Data: 70k grayscale 28x28 images of handwritten digits (60k train / 10k test). Pixels are divided by 255 so they're between 0 and 1.
- Model: MLP: Flatten (784) -> Dense 512 -> Dense 256 -> Dense 128 -> Dense 10, with ReLU and 20% dropout after each hidden layer. That's 567,434 parameters.
- Logits instead of softmax: the last layer outputs raw scores and the loss uses `from_logits=True` (Keras) / `nn.CrossEntropyLoss` (PyTorch, which always expects logits). The loss does the softmax internally in a more numerically stable way.
- Dropout: randomly zeros 20% of neurons during training so the network can't rely on specific neurons. It's turned off at test time (`model.eval()` in PyTorch, Keras does it automatically).
- Training: Adam, lr = 1e-3, batch 128, 15 epochs.

### TensorFlow vs PyTorch differences I noticed

| | TensorFlow / Keras | PyTorch |
|---|---|---|
| Training loop | `model.fit()` does it all | Have to write it: `zero_grad()` -> forward -> `loss.backward()` -> `optimizer.step()` |
| Train/eval mode | Automatic | Have to call `model.train()` / `model.eval()` yourself or dropout stays on during testing |
| Validation split | `validation_split=0.2` takes the last 20% of the data | `random_split` takes a random 20% |
| Data | NumPy arrays go straight in | Convert to tensors, wrap in `TensorDataset` + `DataLoader` |
| GPU | Automatic | Have to `.to(device)` the model and every batch |

Keras is a lot less code. PyTorch is more work but you can see exactly what happens each step, which helped me understand what `fit()` is doing.

## Results with the given code

| Task | Metric | TensorFlow | PyTorch |
|---|---|---|---|
| Housing (linear) | Test MSE | 0.5485 | 0.5629 |
| Housing (linear) | Test MAE | 0.5348 | 0.5466 |
| MNIST (MLP) | Test accuracy | 98.24% | 98.16% |

Pretty much the same in both, which makes sense since it's the same model and optimizer. The small differences come from different random weight init (Keras and PyTorch use different defaults) and the validation split being different (see table above).

To check the linear model I also compared it to sklearn's `LinearRegression`, which solves it exactly. sklearn got a test MSE of 0.5559, and the PyTorch weights were really close to the exact ones (e.g. MedInc 0.857 vs 0.854, Latitude -0.938 vs -0.897). So gradient descent basically found the right answer, it just bounces around it a little because the learning rate never goes down.

## Tuning experience

### Task 1: learning rate, batch size, optimizer, and a non-linear model

Learning rate (TF, Adam, 60 epochs, batch 64):

| Learning rate | Test MSE | Test MAE |
|---|---|---|
| 0.001 | 0.5616 | 0.5338 |
| 0.01 (given) | 0.5485 | 0.5348 |
| 0.1 | 0.7247 | 0.5756 |

0.001 was slower at the start but still got there within 60 epochs. 0.1 was clearly too big: the val loss curve jumped around a lot and it ended up worse (final val MSE 0.81).

Learning rate + optimizer (PyTorch):

| Optimizer | lr | Test MSE | Test MAE |
|---|---|---|---|
| Adam | 0.001 | 0.5603 | 0.5339 |
| Adam | 0.01 | 0.6196 | 0.5383 |
| Adam | 0.1 | 0.5617 | 0.5546 |
| SGD | 0.01 | 0.5782 | 0.5330 |

This one confused me at first because lr = 0.01 did worse in PyTorch but best in TF, and lr = 0.1 was fine in PyTorch but bad in TF. I think what's happening is that with a big learning rate the weights keep bouncing around the best solution instead of settling, so the test MSE depends on where it happens to be on the last epoch. The MAE barely changes between runs, so the actual model is almost the same. Basically one run with a high lr isn't very reliable. A smaller lr (0.001) gave the most consistent results in both frameworks.

Batch size (TF, lr = 0.01):

| Batch size | Test MSE | Test MAE |
|---|---|---|
| 16 | 0.5715 | 0.5343 |
| 64 | 0.5485 | 0.5348 |
| 256 | 0.5554 | 0.5349 |

MAE is basically identical for all three. Batch 16 is a bit noisier (more updates per epoch, each one on less data), batch 256 runs faster per epoch. For a model with 9 parameters it really doesn't matter much.

Linear vs a small MLP: this was the biggest result for Task 1. Same data and training setup, but two hidden ReLU layers (64 -> 32):

| Model | Test MSE | Test MAE |
|---|---|---|
| Linear (given) | 0.5485 | 0.5348 |
| MLP 64-32 | 0.2897 | 0.3744 |

The error almost got cut in half. None of the learning rate or batch size changes got the linear model below about 0.55, so the model itself was the problem, not the hyperparameters. House prices aren't linear in latitude/longitude (coast vs inland, Bay Area vs LA, etc.) and a straight line can't capture that. Train and val loss for the linear model were almost the same the whole time, which means it's underfitting, not overfitting.

Looking at the learned weights (TF), MedInc (+0.82) was the biggest positive one, and Latitude (-0.87) and Longitude (-0.86) were the biggest negative ones, which is the model's way of encoding location with a straight line.

### Task 2: MNIST

One thing I noticed is that the given code passes the test set as `validation_data`. That's fine for plotting, but if you pick hyperparameters based on it you're kind of tuning on the test set. So for my tuning runs I held out 10% of the training set as validation and only checked the test set at the end. Each run was 10 epochs and I changed one thing at a time from the baseline.

TensorFlow:

| Config | Train acc | Val acc | Test acc | Epoch with lowest val loss |
|---|---|---|---|---|
| Baseline (lr 1e-3, dropout 0.2, batch 128) | 98.90% | 98.10% | 98.01% | 7 |
| lr = 1e-2 | 94.55% | 97.32% | 96.77% | 10 |
| lr = 1e-4 | 97.46% | 97.88% | 97.60% | 10 |
| No dropout | 99.39% | 97.98% | 97.88% | 3 |
| Dropout 0.5 | 97.20% | 98.25% | 97.84% | 10 |
| Batch 32 | 98.68% | 98.42% | 97.97% | 5 |
| Smaller net (one 128 layer, 102k params) | 98.11% | 97.95% | 97.65% | 10 |

PyTorch (same idea, but I tried SGD with momentum instead of batch size / smaller net):

| Config | Val acc | Test acc | Epoch with lowest val loss |
|---|---|---|---|
| Baseline | 97.95% | 97.92% | 7 |
| No dropout | 97.68% | 97.93% | 7 |
| Dropout 0.5 | 97.88% | 98.08% | 9 |
| Adam lr = 1e-2 | 96.35% | 96.59% | 9 |
| Adam lr = 1e-4 | 96.42% | 97.01% | 10 |
| SGD + momentum lr = 0.05 | 97.83% | 98.11% | 9 |

What I got from this:

- Learning rate mattered the most again. lr = 1e-2 was the worst in both frameworks (~96.6-96.8% test). lr = 1e-4 was stable but slow, still improving at epoch 10.
- No dropout = overfitting. In TF the train accuracy got up to 99.4% but val loss was lowest at epoch 3 and went up after that. The "dropout effect" plot shows this pretty clearly: the no-dropout train loss keeps going down while its val loss goes up.
- Dropout 0.5: train accuracy (97.2%) was actually *lower* than val accuracy (98.25%). That confused me until I realized dropout is only on during training, so the training numbers are measured on a "handicapped" network.
- Smaller network: one 128-unit layer (102k params, about 5.5x fewer) only lost about 0.4% test accuracy.
- SGD with momentum worked just as well as Adam in PyTorch (98.11% test), it just needed a higher lr (0.05).
- Most of the configs ended up within about 0.5% of each other, so the given settings were already pretty good.

Early stopping (TF): in the original 15-epoch run the val loss was lowest around epoch 7 (0.0631) and then crept back up to 0.0724 by epoch 15, while train loss kept going down. Val accuracy stayed about the same though (~98.1-98.3%). So I added `EarlyStopping(patience=3, restore_best_weights=True)`. It stopped after 10 epochs, went back to the epoch 7 weights, and got 97.97% test accuracy while training 1/3 fewer epochs. (It's a bit lower than 98.24% because it only trains on 90% of the training data.)

Where it messes up: 176 wrong out of 10,000 in TF. The biggest ones in the confusion matrix were 5 -> 3 (10 times), 9 -> 7 (9), 2 -> 3 (7), 7 -> 3 (7), 7 -> 2 (6), and 4 -> 9 (6). Looking at the misclassified images, a lot of them are just really messy handwriting. A couple of the 7s I would have guessed wrong too.

## Observations

1. TF and PyTorch gave basically the same results (98.24% vs 98.16% on MNIST, 0.55 vs 0.56 MSE on housing). The difference between them is how much code you write, not what the model learns.
2. Learning rate was the most sensitive hyperparameter in both tasks. Too big = noisy and worse, too small = slow.
3. For housing, the model mattered more than tuning. No hyperparameter change got the linear model under ~0.55 MSE, but adding hidden layers dropped it to 0.29.
4. Dropout controls overfitting more than it raises peak accuracy. Without it the gap between train and val got bigger and val loss started rising after only a few epochs.
5. Loss and accuracy don't always move together. On MNIST the val loss went up in later epochs but val accuracy stayed flat, so it's worth plotting both.
6. Results from one run can be misleading, like the lr = 0.01 vs 0.1 thing on housing where TF and PyTorch disagreed. Running a few seeds would be better.
7. Using the test set as validation (like the given code) is something to watch out for when tuning.

## Conclusion

Both frameworks worked in Colab and gave almost the same results: about 0.55 test MSE for linear regression on California Housing and about 98.2% test accuracy on MNIST. For housing, the linear model is just too simple for the data, and a small neural net did way better (0.29 MSE). For MNIST, the given settings (Adam 1e-3, dropout 0.2) were already close to the best I found. The main improvement was stopping around epoch 7-10 instead of 15 since the extra epochs just start overfitting. To go much higher than ~98.3% you'd probably need a CNN, because the MLP flattens the image and loses the 2D structure.

If I had to pick one, Keras is easier for getting something working fast, and PyTorch is better when you want control over the training loop or want to understand what's going on.

## Screenshots

1. `tf_01_setup_linear.png` - TF version output and the linear model training log + test MSE
2. `tf_02_housing_loss.png` - housing loss curve
3. `tf_03_weights.png` - learned weights bar chart + predicted vs actual plot
4. `tf_04_lr_batch.png` - learning rate sweep plot/table and batch size table
5. `tf_05_linear_vs_mlp.png` - linear vs MLP comparison
6. `tf_06_mnist_train.png` - MNIST sample digits and the 15-epoch training log
7. `tf_07_mnist_curves.png` - accuracy/loss curves + test accuracy
8. `tf_08_confusion.png` - confusion matrix and misclassified digits
9. `tf_09_mnist_tuning.png` - tuning table + val accuracy/loss plots + dropout plot
10. `tf_10_early_stopping.png` - early stopping output
11. `pt_01_linear.png` - PyTorch device check, training log + test MSE, loss curve
12. `pt_02_ols_and_lr.png` - sklearn comparison table + optimizer/lr sweep
13. `pt_03_mnist.png` - PyTorch MNIST epoch log + curves
14. `pt_04_mnist_tuning.png` - PyTorch tuning table + plots
