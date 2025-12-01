# Data Preparation Instructions

## Double MNIST Dataset
### 1. Download the dataset
kaggle datasets download -d amankumar234/mnist-2-digit-classification-dataset

### 2. Unzip into the data directory
unzip mnist-2-digit-classification-dataset.zip -d data/double_mnist


## Fashion MNIST Dataset
## 📂 Dataset Setup

### 1. FashionMNIST (Task 2)
The FashionMNIST dataset is integrated directly into the `torchvision` library. You do not need to manually download it. The code will automatically check for the data in the `./data` folder and download it if missing.

**To verify or download manually:**
```python
from torchvision import datasets

# This will download the dataset to the 'data/' directory
train_data = datasets.FashionMNIST(root='data', train=True, download=True)
test_data = datasets.FashionMNIST(root='data', train=False, download=True)

```


