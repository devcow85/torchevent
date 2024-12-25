import time
import tonic
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader

from torchevent.utils import set_seed, spike2data
from torchevent.transforms import RandomTemporalCrop, TemporalCrop
from torchevent import models, loss

torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(7)
    
    # load dataset
    transform = transforms.Compose([
        RandomTemporalCrop(time_window = 99000),
        transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size,
                           n_time_bins=5)
    ])
    
    train_ds = tonic.datasets.NMNIST(save_to = 'data', 
                                         train=True, 
                                         transform = transform)
    
    transform = transforms.Compose([
        TemporalCrop(time_window = 99000),
        transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size,
                           n_time_bins=5)
    ])
    
    val_ds = tonic.datasets.NMNIST(save_to = 'data', 
                                         train=False, 
                                         transform = transform)
    
    batch_size = 32
    
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=8)
    
    model = models.NMNISTNet(5, 1, n_steps = 5)
    model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    criterion = loss.SpikeCountLoss(4, 1)
    
    model.train()
    
    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()  # Start time for epoch
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            batch_start_time = time.time()  # Start time for batch
            
            data, targets = data.to("cuda", non_blocking=True), targets.to("cuda", non_blocking=True)
            spikes = model(data.to(torch.float32))
            
            optimizer.zero_grad()
            spike_loss = criterion(spikes, targets)
            spike_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            epoch_loss += spike_loss.item()
            
            # Calculate elapsed time
            batch_elapsed_time = time.time() - batch_start_time
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {spike_loss.item():.4f}, Elapsed Time: {batch_elapsed_time:.2f}s")
        
        # Calculate epoch elapsed time
        epoch_elapsed_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {epoch_loss / len(train_loader):.4f}, "
              f"Elapsed Time: {epoch_elapsed_time:.2f}s")
    
    # Validation loop
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct = 0
        total = 0
        validation_start_time = time.time()  # Start time for validation
        
        for data, targets in val_loader:
            data, targets = data.to("cuda"), targets.to("cuda")
            spikes = model(data.to(torch.float32))
            
            # Calculate loss
            spike_loss = criterion(spikes, targets)
            val_loss += spike_loss.item()
            
            # Calculate predictions
            predictions = spike2data(spikes, return_pred=True)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
        
        validation_elapsed_time = time.time() - validation_start_time  # Validation elapsed time
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, "
              f"Accuracy: {100 * correct / total:.2f}%, Elapsed Time: {validation_elapsed_time:.2f}s")