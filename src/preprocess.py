import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder, alphabet = torch.hub.load("facebookresearch/esm", "esm1_t6_43M_UR50S")
dimension = 768
batch_converter = alphabet.get_batch_converter()

def preprocess_batch(batch, seq_dict, max_len=2046, use_cuda=False):
    protein_dict = {}
    protein_a_list = []
    protein_b_list = []
    interaction_list = []
    for pair_data in batch:
        protein_a, protein_b, interaction = pair_data
        protein_a_seq = seq_dict[protein_a]
        protein_b_seq = seq_dict[protein_b]
        if (len(protein_a_seq) > max_len) or (len(protein_b_seq) > max_len):
            continue
        protein_dict[protein_a] = protein_a_seq
        protein_dict[protein_b] = protein_b_seq
        protein_a_list.append(protein_a)
        protein_b_list.append(protein_b)
        interaction_list.append(interaction)

    batch_size = len(interaction_list)

    protein_encode = protein_dict.items()
    batch_labels, batch_strs, batch_tokens = batch_converter(protein_encode)

    if use_cuda:
        if torch.cuda.is_available:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    encoder.to(device)

    with torch.no_grad():
        results = encoder(batch_tokens.to(device), repr_layers=[6])
    token_embeddings = results["representations"][6]

    for i, (protein_name, seq) in enumerate(protein_encode):
        protein_dict[protein_name] = token_embeddings[i, 1:len(seq) + 1].mean(0)

    protein_a_tensor = torch.zeros(batch_size, dimension)
    protein_b_tensor = torch.zeros(batch_size, dimension)

    for idx in range(batch_size):
        protein_a_tensor[idx, :] = protein_dict[protein_a_list[idx]]
        protein_b_tensor[idx, :] = protein_dict[protein_b_list[idx]]
        protein_pair_tensor = torch.cat((protein_a_tensor, protein_b_tensor), 1)
    interaction_tensor = torch.tensor(interaction_list)
    dummy_interaction_tensor = (interaction_tensor - 1) * -1
    output_interaction_tensor = torch.stack((dummy_interaction_tensor, interaction_tensor), dim=1)

    return protein_pair_tensor.detach(), output_interaction_tensor
