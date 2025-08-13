from dvc.monitor import DVCMonitor
import pandas as pd

if __name__ == "__main__":
    
    mon = DVCMonitor("NMNIST")
    
    stats = mon.get_event_dataset_info(save_as="test.csv")
    
    df = pd.DataFrame(stats)