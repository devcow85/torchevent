from torchevent import models


if __name__ == "__main__":
    model = models.NMNISTNet(5, 1, n_steps = 5)
    trace_data = model.trace((5,2,34,34))
    
    model.summary((5,2,34,34))
