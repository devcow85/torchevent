# **Torchevent: Spiking Neural Network Framework**

`Torchevent` is a PyTorch-based framework for **Spiking Neural Networks (SNNs)**. It supports training and inference for event-based datasets like NMNIST, offering custom models, loss functions, and transformations optimized for SNN workflows.

---

## References
This project draws inspiration from the following works:

1. **Paper**:  
   **TSSL-BP: Temporal-Spike-Sequence Learning via Backpropagation for Spiking Neural Networks**  
   Proceedings of the 34th Conference on Neural Information Processing Systems (NeurIPS), 2020.  
   [Link to Paper](https://proceedings.neurips.cc/paper/2020/hash/8bdb5058376143fa358981954e7626b8-Abstract.html)

2. **GitHub Repository**:  
   **TSSL-BP: Temporal Spike-Sequence Learning Framework**  
   [GitHub Repository](https://github.com/stonezwr/TSSL-BP)

We thank the authors of these works for providing valuable insights into spiking neural network research and implementation.

---

## **Features**

### **1. SNN Models (TSSL-BP)**
- Models like `NMNISTNet` and `NCARSNet` are specifically designed for event-based datasets.
- Easily configurable for various spiking network architectures and time-step dynamics.

### **2. Event Data Transformations**
- Transformations tailored for event-based data processing:
  - `RandomTemporalCrop`: Randomly crops events based on a given time window.
  - `TemporalCrop`: Sequentially crops events within a fixed time window.
  - `ToFrameAuto`: Converts events into frames with dynamic configurations.

### **3. Loss Functions**
- Loss functions designed for SNN-specific requirements:
  - `SpikeKernelLoss`: Computes the loss using Post-Synaptic Potentials (PSP).
  - `SpikeCountLoss`: Optimizes models to match desired spike counts.
  - `SpikeSoftmaxLoss`: Combines spike data with softmax and cross-entropy for classification tasks.

---

## **Installation**
To install `torchevent` manually:
```bash
git clone https://github.com/devcow85/torchevent.git
cd torchevent
pip install .
```

---

## **Usage**
The following script demonstrates training the `NMNISTNet` model using the `NMNIST` dataset from `tonic` api:
```python
import tonic
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader

from torchevent.utils import set_seed, spike2data
from torchevent.transforms import RandomTemporalCrop, TemporalCrop
from torchevent import models, loss

# Set seed for reproducibility
set_seed(7)

# Prepare the dataset
transform = transforms.Compose([
    RandomTemporalCrop(time_window=99000),
    transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, n_time_bins=5),
])

train_ds = tonic.datasets.NMNIST(save_to="data", train=True, transform=transform)
val_ds = tonic.datasets.NMNIST(save_to="data", train=False, transform=transform)

# Create data loaders
train_loader = DataLoader(train_ds, shuffle=True, batch_size=32, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, shuffle=False, batch_size=32, num_workers=8, pin_memory=True)

# Initialize model, optimizer, and loss function
model = models.NMNISTNet(5, 1, n_steps=5).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
criterion = loss.SpikeCountLoss(desired_count=4, undesired_count=1)

# Training loop
for epoch in range(3):
    model.train()
    for data, targets in train_loader:
        data, targets = data.to("cuda", non_blocking=True), targets.to("cuda", non_blocking=True)
        optimizer.zero_grad()
        spikes = model(data.to(torch.float32))
        spike_loss = criterion(spikes, targets)
        spike_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        print(f"Epoch [{epoch+1}], Loss: {spike_loss.item():.4f}")
```

### **Expected Result**

```bash
Epoch [1/3], Step [10/1875], Loss: 40.6000, Elapsed Time: 0.13s
...
Epoch [1/3] completed. Average Loss: 22.7644, Elapsed Time: 163.37s
...
Epoch [2/3], Step [1870/1875], Loss: 20.1000, Elapsed Time: 0.06s
Epoch [2/3] completed. Average Loss: 18.2996, Elapsed Time: 107.86s
...
Epoch [3/3], Step [1870/1875], Loss: 10.9000, Elapsed Time: 0.06s
Epoch [3/3] completed. Average Loss: 15.9984, Elapsed Time: 108.05s
Validation Loss: 15.2796, Accuracy: 91.01%, Elapsed Time: 5.16s
```

## **Contact**
For quenstions, suggestions, or support, please contact Seokhun Jeon (seokhun.jeon@keti.re.kr)