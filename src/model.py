import torch
from torch import nn

from sklearn.metrics import accuracy_score


class ProteinInteraction(nn.Module):
    def __init__(self, dim_size=768):
        super(ProteinInteraction, self).__init__()

        self.input_layer = nn.Linear(2*dim_size, 768)
        self.h1_layer = nn.Linear(768, 265)
        self.h2_layer = nn.Linear(265, 2)

    def calculate_accuracy(self, output, target):
        prediction = torch.argmax(output, dim=1).detach()
        target = torch.argmax(target, dim=1)

        accuracy = accuracy_score(prediction, target)
        return accuracy, prediction


    def forward(self, protein_pair_tensor, interaction_tensor):

        x = self.input_layer(protein_pair_tensor)
        x = nn.functional.relu(x)
        x = self.h1_layer(x)
        x = nn.functional.relu(x)
        x = self.h2_layer(x)
        output = nn.functional.softmax(x, dim=1)

        loss = nn.functional.binary_cross_entropy(output.float(), interaction_tensor.float())

        accuracy, prediction = self.calculate_accuracy(output, interaction_tensor)

        return output, loss, accuracy, prediction



