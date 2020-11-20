import torch
from torch import nn

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ProteinInteraction(nn.Module):
    def __init__(self, dim_size=768):
        super(ProteinInteraction, self).__init__()

        self.input_layer = nn.Linear(2*dim_size, 768)
        self.h1_layer = nn.Linear(768, 265)
        self.h2_layer = nn.Linear(265, 2)

    def evaluate(self, prediction, target):
        target = torch.argmax(target, dim=1)

        accuracy = accuracy_score(target, prediction)
        f1 = f1_score(target, prediction)
        precision = precision_score(target, prediction)
        recall = recall_score(target, prediction)

        return accuracy, f1, precision, recall


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
            accuracy, f1, precision, recall = self.evaluate(prediction, interaction_tensor)
            return output, loss, accuracy, f1, precision, recall, prediction

        else:
            return output.detach(), prediction



