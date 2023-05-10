# Cats_vs_Dogs
Using Pytorch for Cats and Dogs recognition.

You can get dataset from [Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/data).

![Image text](https://github.com/brianlin314/Cats_vs_Dogs/blob/main/assets/cat.jpg)
![Image text](https://github.com/brianlin314/Cats_vs_Dogs/blob/main/assets/dog.jpg)
## Data Structure
```
├── cat_dog_dataset
    └── training_dataset
        └── Cat
            └── 0.jpg
            └── 1.jpg
            ...
        └── Dog
    └── training_dataset
        └── Cat
        └── Dog
```
## Custom Dataset Introduction
### Pytorch Custom Dataset
contributed by < `brianlin314` >

```python
from torch.utils.data.dataset import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, ...):
        # stuff
        
    def __getitem__(self, index):
        # stuff
        return (img, label)

    def __len__(self):
        return count # of how many examples(images?) you have
```
`Dataset`必須包含以下函數，供 `Dataloader` 使用。

- `__init__()` 初始邏輯發生的地方，比如: 讀取 csv、資料 transforms、過濾 data 等。
- `__getitem__()` 通常傳回 `data` 和 `label`，但不代表只能傳回這些，可以由你自己客製化定義。需要注意的一點是 `__getitem__()` 返回 Data 的特定類型（如 tensor、numpy 等），否則，在 Dataloader 中會噴錯。
```python
img, label = MyCustomDataset.__getitem__(99)  # For 99th item
```

- `__len__()` 返回樣本數。

### Using Torchvision Transforms
在大多數範例中，會在 `__init__()` 中看到 `transforms = None`，這用於將 `torchvision` 轉換應用於數據/圖像。可參考[這裡](https://pytorch.org/docs/0.2.0/torchvision/transforms.html)轉換最常見的用法是這樣的：
```python
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class MyCustomDataset(Dataset):
    def __init__(self, ..., transforms=None):
        # stuff
        ...
        self.transforms = transforms
        
    def __getitem__(self, index):
        # stuff
        ...
        data = # Some data read from a file or image
        if self.transforms is not None:
            data = self.transforms(data)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return (img, label)

    def __len__(self):
        return count # of how many data(images?) you have
        
if __name__ == '__main__':
    # Define transforms (1)
    transformations = transforms.Compose([transforms.CenterCrop(100), transforms.ToTensor()])
    # Call the dataset
    custom_dataset = MyCustomDataset(..., transformations)
    
```

### Using Data Loader
Pytorch `DataLoaders` 只需調用 `__getitem__()` 並將它們包裝成一個 `batch`。 其實可以不使用 `Dataloader` 並一次調用一個 `__getitem__()` 並將數據餵給模型。假設有一個 `CustomDatasetFromCSV` 的自定義 Dataset，那麼可以像這樣調用 `Dataloader`：

```python
if __name__ == "__main__":
    # Define transforms
    transformations = transforms.Compose([transforms.ToTensor()])
    # Define custom dataset
    custom_mnist_from_csv = \
        CustomDatasetFromCSV('../data/mnist_in_csv.csv',
                             28, 28,
                             transformations)
    # Define data loader
    mn_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_csv,
                                                    batch_size=10,
                                                    shuffle=False)
    
    for images, labels in mn_dataset_loader:
        # Feed the data to the model
```

`Dataloader` 的第一個參數是 `Dataset`，它調用該 `Dataset` 的 `__getitem__()` 。 `batch_size` 設定一個 `batch` 將包裝多少個 `Data`。 如果我們假設單張圖像的 `tensor` 大小為：1x28x28（D:1、H:28、W:28），那麼使用此`Dataloader`返回的張量將為 10x1x28x28（Batch-Depth-Height-Width）。
