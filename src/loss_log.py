import pickle
from matplotlib import pyplot as plt

class LossLog(object):
    def __init__(self):
        self.losses = {}

    def __str__(self):
        return self.losses.__str__()
    
    def add_loss(self, new_losses):
        losses_keys = self.losses.keys()
        for key in new_losses.keys():
            if key not in losses_keys:
                self.losses[key] = []
                for loss in new_losses[key]:
                    self.losses[key].append([loss])
            else:
                for i in range(len(self.losses[key])):
                    self.losses[key][i].append(new_losses[key][i])
        
    def save(self):
        with open('losses_log_dict.pickle', 'wb') as f:
            pickle.dump(self.losses, f)

    def load(self):
        with open('train_info.pickle', 'rb') as f:
            self.losses = pickle.load(f)

    def graph(self):
        pass

    
