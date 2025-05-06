
class EarlyStopping:
    def __init__(self, patience, delta):
        """
        Early stopping class to prevent overfitting.
        
        :param patience: Number of epochs with no improvement after which training will be stopped.
        :param delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = float('inf')
        self.stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        """
        Call method to check if training should be stopped.
        
        :param val_loss: Current validation loss.
        :param model: Current model.
        
        :return: True if training should be stopped, False otherwise.
        """
        if val_loss < self.best_score - self.delta:
            self.counter = 0
            self.best_model = model
            self.best_score = val_loss
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop
    
    def get_best_model(self):
        return self.best_model   