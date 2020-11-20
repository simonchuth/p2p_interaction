import argparse
import time
import random

from os.path import join

import torch
from torch import optim

from src.utils import pickle_load, json_load, chunk_list, str2bool
from src.preprocess import preprocess_batch
from src.model import ProteinInteraction

def main(datapath, random_seed=1, test_fraction=0.1, batch_size=100, max_len=2046, num_epoch=10, subset_data=1, use_cuda=False):
    dataset_nip = pickle_load(join(datapath, 'dataset_nip.pkl'))
    dataset_ip = pickle_load(join(datapath, 'dataset_ip.pkl'))
    seq_dict = json_load(join(datapath, 'seq_dict.json'))

    random.shuffle(dataset_nip)
    random.shuffle(dataset_ip)

    test_size = int(test_fraction * len(dataset_nip))
    train_size = len(dataset_nip) - test_size
    train_nip = dataset_nip[:train_size]
    train_ip = dataset_ip[:train_size]
    test_nip = dataset_nip[-test_size:]
    test_ip = dataset_ip[-test_size:]

    train = train_nip + train_ip
    test = test_nip + test_ip

    random.shuffle(train)
    random.shuffle(test)

    if subset_data < 1:
        train_size = int(subset_data * len(train))
        test_size = int(subset_data * len(test))
        train = train[:train_size]
        test = test[:test_size]

    train_chunk = chunk_list(train, batch_size)
    test_chunk = chunk_list(test, batch_size)

    train_num_batch = len(train_chunk)
    test_num_batch = len(test_chunk)
    print(f'Number of train batch: {train_num_batch}')
    print(f'Number of test batch: {test_num_batch}')

    model = ProteinInteraction()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    for epoch in range(num_epoch):
        print(f'Epoch {epoch + 1}')
        train_start_time = time.time()
        train_loss = 0
        train_acc = 0

        for batch in train_chunk:
            # Batch preprocessing
            protein_pair_tensor, interaction_tensor = preprocess_batch(batch, seq_dict, max_len=max_len, use_cuda=use_cuda)

            # Train model
            optimizer.zero_grad()
            output, loss, accuracy, prediction = model(protein_pair_tensor, interaction_tensor)
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().numpy().item()
            train_acc += accuracy.item()

        train_loss = train_loss/train_num_batch
        train_acc = train_acc/train_num_batch

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        print(f'Epoch {epoch + 1} - Training: {int(time.time() - train_start_time)} sec')
        print(f'Train Loss: {train_loss} ... Train Accuracy: {train_acc}')

        test_start_time = time.time()
        test_loss = 0
        test_acc = 0

        for batch in test_chunk:
            # Batch preprocessing
            protein_pair_tensor, interaction_tensor = preprocess_batch(batch, seq_dict, max_len=max_len, use_cuda=use_cuda)

            # Train model
            with torch.no_grad():
                output, loss, accuracy, prediction = model(protein_pair_tensor, interaction_tensor)

            test_loss += loss.detach().numpy().item()
            test_acc += accuracy.item()

        test_loss = test_loss/test_num_batch
        test_acc = test_acc/test_num_batch

        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        print(f'Epoch {epoch + 1} - Testing: {int(time.time() - test_start_time)} sec')
        print(f'Test Loss: {test_loss} ... Test Accuracy: {test_acc}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('datapath', type=str, help='path to dataset')
    parser.add_argument('--random_seed', type=int, default=1, help='Random Seed')
    parser.add_argument('--test_fraction', type=float, default=0.1, help='Fraction of dataset for evaluation')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--max_len', type=int, default=2046, help='Max length of protein')
    parser.add_argument('--num_epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--subset_data', type=float, default=1.0, help='Use subset of data')
    parser.add_argument('--use_cuda', type=str2bool, default=False, help='Use subset of data')

    args = parser.parse_args()

    main(args.datapath,
         random_seed=args.random_seed,
         test_fraction=args.test_fraction,
         batch_size=args.batch_size,
         max_len=args.max_len,
         num_epoch=args.num_epoch,
         subset_data=args.subset_data,
         use_cuda=args.use_cuda)



