import torch
from torch import nn

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ProteinInteraction(nn.Module):
    def __init__(self, dim_size=768):
        super(ProteinInteraction, self).__init__()

        self.input_layer = nn.Linear(2*dim_size, 768)
        self.h1_layer = nn.Linear(768, 265)
        self.h2_layer = nn.Linear(265, 2)


    def forward(self, protein_pair_tensor, interaction_tensor=None):

        x = self.input_layer(protein_pair_tensor)
        x = nn.functional.relu(x)
        x = self.h1_layer(x)
        x = nn.functional.relu(x)
        x = self.h2_layer(x)
        output = nn.functional.softmax(x, dim=1)
        prediction = torch.argmax(output, dim=1).detach()
        

        if interaction_tensor is not None:
            loss = nn.functional.binary_cross_entropy(output.float(), interaction_tensor.float())
            target = torch.argmax(interaction_tensor, dim=1)
            return output, loss, prediction, target

        else:
            return output.detach(), prediction


def evaluate(target, prediction):
    target = np.argmax(target, axis=1)

    accuracy = accuracy_score(target, prediction)
    f1 = f1_score(target, prediction)
    precision = precision_score(target, prediction)
    recall = recall_score(target, prediction)

    return accuracy, f1, precision, recall


