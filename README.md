# Pytorch-lightning-practice
MNIST Dataset classification using pytorch lightning
* 20 epoch
* CE loss
* Custom Network
* Adam optimizer with lr=5e-4
* Batch Size: 128

## Dev Environment
* Chip: M1 Pro (8-core CPU &14-core GPU)
* RAM: 16GB
* Accelerator: CPU, not GPU (pytorch lightning not yet supported MPS accelerator )

## Results
### Accuracy and Loss
|    Metric     | Result |
|:-------------:|:------:|
| Test Accuracy | 0.9767 |
|   Test Loss   | 0.0830 |

### Time
|  **Task** | **Time (sec)** |
|:---------:|:--------------:|
| Data Load |      2.95      |
|   Train   |      13.99     |
|    Test   |      0.57      |
