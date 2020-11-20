import argparse
import time
import random

from os.path import join

import torch
from torch import optim

from src.utils import pickle_load, json_load, chunk_list, str2bool, pickle_save
from src.preprocess import preprocess_batch
from src.model import ProteinInteraction, evaluate

def main(datapath,
         random_seed=1,
         test_fraction=0.1,
         batch_size=100,
         max_len=2046,
         num_epoch=10,
         subset_data=1,
         use_cuda=False,
         es_patience=3):

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
    train_f1_list = []
    train_precision_list = []
    train_recall_list = []
    test_loss_list = []
    test_acc_list = []
    test_f1_list = []
    test_precision_list = []
    test_recall_list = []

    best_loss = 1000
    es_counter = 0

    for epoch in range(num_epoch):
        print(f'Epoch {epoch + 1}')
        train_start_time = time.time()
        train_loss = 0
        train_prediction = []
        train_target = []

        for batch in train_chunk:
            # Batch preprocessing
            protein_pair_tensor, interaction_tensor = preprocess_batch(batch, seq_dict, max_len=max_len, use_cuda=use_cuda)

            # Train model
            optimizer.zero_grad()
            output, loss, prediction, target = model(protein_pair_tensor, interaction_tensor)
            train_prediction.append(prediction)
            train_target.append(target)

            loss.backward()
            optimizer.step()

            train_loss += loss.detach().numpy().item()

        train_loss = train_loss/train_num_batch

        prediction = torch.cat(train_prediction, 0).numpy()
        target = torch.cat(train_target, 0).numpy()
        accuracy, f1, precision, recall = evaluate(target, prediction)

        train_loss_list.append(train_loss)
        train_acc_list.append(accuracy)
        train_f1_list.append(f1)
        train_precision_list.append(precision)
        train_recall_list.append(recall)

        print(f'Epoch {epoch + 1} - Training: {int(time.time() - train_start_time)} sec')
        print(f'Train Loss: {train_loss} ... Train Accuracy: {accuracy}')
        print(f'Train F1: {f1} ... Train Precision: {precision}')
        print(f'Train Recall: {recall}')

        test_start_time = time.time()
        test_loss = 0
        test_prediction = []
        test_target = []

        for batch in test_chunk:
            # Batch preprocessing
            protein_pair_tensor, interaction_tensor = preprocess_batch(batch, seq_dict, max_len=max_len, use_cuda=use_cuda)

            # Train model
            with torch.no_grad():
                output, loss, prediction, target = model(protein_pair_tensor, interaction_tensor)
            test_prediction.append(prediction)
            test_target.append(target)

            test_loss += loss.detach().numpy().item()

        test_loss = test_loss/test_num_batch

        prediction = torch.cat(test_prediction, 0).numpy()
        target = torch.cat(test_target, 0).numpy()
        accuracy, f1, precision, recall = evaluate(target, prediction)

        test_loss_list.append(test_loss)
        test_acc_list.append(accuracy)
        test_f1_list.append(f1)
        test_precision_list.append(precision)
        test_recall_list.append(recall)

        print(f'Epoch {epoch + 1} - Testing: {int(time.time() - test_start_time)} sec')
        print(f'Test Loss: {test_loss} ... Test Accuracy: {accuracy}')
        print(f'Test F1: {f1} ... Test Precision: {precision}')
        print(f'Test Recall: {recall}')

        if test_loss < best_loss:
            torch.save(model, 'best_model.pt')
            best_loss = test_loss
            es_counter = 0
        else:
            print('Loss not decreasing')
            es_counter += 1

        pickle_save(train_loss_list, 'train_loss.pkl')
        pickle_save(train_acc_list, 'train_acc.pkl')
        pickle_save(train_f1_list, 'train_f1.pkl')
        pickle_save(train_precision_list, 'train_precision.pkl')
        pickle_save(train_recall_list, 'train_recall.pkl')
        pickle_save(test_loss_list, 'test_loss.pkl')
        pickle_save(test_acc_list, 'test_acc.pkl')
        pickle_save(test_f1_list, 'test_f1.pkl')
        pickle_save(test_precision_list, 'test_precision.pkl')
        pickle_save(test_recall_list, 'test_recall.pkl')

        if es_counter > es_patience:
            break

    torch.save(model, 'final_model.pt')



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



