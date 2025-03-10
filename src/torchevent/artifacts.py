
import json
import pickle
import os
import math

import pandas as pd
from datetime import datetime
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ArtifactManager:
    def __init__(self, experiment_name, cache_dir=None):
        self.experiment_name = experiment_name
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/torchevent")
        self.artifacts = {}
        os.makedirs(self.cache_dir, exist_ok=True)

    def __setitem__(self, key, value):
        self.artifacts[key] = value

    def __getitem__(self, key):
        return self.artifacts.get(key, None)

    def __delitem__(self, key):
        if key in self.artifacts:
            del self.artifacts[key]

    def __contains__(self, key):
        return key in self.artifacts

    def save(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment_name}_{timestamp}.pkl"
        save_path = os.path.join(self.cache_dir, filename)

        try:
            with open(save_path, "wb") as f:
                pickle.dump(self.artifacts, f)
            print(f"Artifacts saved successfully to {save_path}")
        except Exception as e:
            print(f"Error while saving artifacts: {e}")

    def load(self, filename):
        load_path = os.path.join(self.cache_dir, filename)
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Artifacts file not found at {load_path}")

        try:
            with open(load_path, "rb") as f:
                self.artifacts = pickle.load(f)
            print(f"Artifacts loaded successfully from {load_path}")
            return self.artifacts
        except Exception as e:
            print(f"Error while loading artifacts: {e}")
            return None

    def export(self, output_dir=None):
        os.makedirs(output_dir, exist_ok=True)

        for key, value in self.artifacts.items():
            file_path = os.path.join(output_dir, f"{key}")
            try:
                if key.lower() == "trace_data":
                    # Save as pickle
                    with open(f"{file_path}.pkl", "wb") as f:
                        pickle.dump(value, f)
                    print(f"Artifact '{key}' saved as pickle to {file_path}.pkl")
                elif key.lower() == "best_model":
                    torch.save(value,f"{file_path}.pt")
                    print(f"Artifact '{key}' saved as pickle to {file_path}.pt")
                elif key.lower() == "summary" and hasattr(value, "to_csv"):
                    # Save as CSV
                    csv_path = f"{file_path}.csv"
                    value.to_csv(csv_path, index=False)
                    print(f"Artifact '{key}' saved as CSV to {csv_path}")
                    
                elif key.lower() == "learning_curves":
                    with open(f"{file_path}.json", "w", encoding="utf-8") as f:
                        json.dump(value, f, indent=4)
                        
                    self._plot_learning_curves(learning_curve_json = value).savefig(f"{file_path}.png", dpi=300)
                    
                elif isinstance(value, (dict, list)):
                    with open(f"{file_path}.json", "w", encoding="utf-8") as f:
                        json.dump(value, f, indent=4)
                    print(f"Artifact '{key}' saved as JSON to {file_path}.json")
                else:
                    with open(f"{file_path}.txt", "w", encoding="utf-8") as f:
                        f.write(str(value))
                    print(f"Artifact '{key}' saved as plain text to {file_path}.txt")
            except Exception as e:
                print(f"Error while saving artifact '{key}': {e}")

    def _plot_learning_curves(self, learning_curve_json, metrics = None):
        df = pd.DataFrame(learning_curve_json)
        
        if metrics is None:
            metrics = [col for col in df.columns if col not in ['epoch', 'phase'] and pd.api.types.is_numeric_dtype(df[col])]

        num_metrics = len(metrics)
        grid_size = math.ceil(math.sqrt(num_metrics))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 5, grid_size * 5))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            for phase in df['phase'].unique():
                phase_data = df[df['phase'] == phase]
                ax.plot(phase_data['epoch'], phase_data[metric], label=phase)

            ax.set_title(f"{metric.capitalize()} Over Epochs")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        return plt