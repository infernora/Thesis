
import torch
import torch.nn as nn
import spikingjelly
import torchvision
import torch.utils.data as data
from tqdm import tqdm
from spikingjelly.activation_based import neuron, layer, learning, surrogate, encoding, functional
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import nltk
import re
import string
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import binary_embedding as embed
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import random
from utils import binary_embedding as embed

# External Imports
import os
import random
# import h5py


import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
import nltk

import re
import string
import tensorflow_hub as hub

import lava.lib.dl.slayer as slayer
device = "cpu"

DATASET = "emotion"  # or "emotion"
DATA_SPLITS = {
    "train": .6,
    "valid": .1,
    "test": .3
}
NETWORK_TYPE = "SNN"
RATE_CODING = True
SPIKE_RAND = False
MAX_T = 10


def rate_code(embed_vector, duration, spike_rand=SPIKE_RAND):
    neurons = []
    times = []
    # Constrain the vector to (0, 1)
    embed_vector = (embed_vector + 1) / 2
    if spike_rand:
        for t in range(duration):
            active_neurons = [i for i in range(len(embed_vector)) if embed_vector[i] <= random.random()]
            neurons.extend(active_neurons)
            times.extend(([t] * len(active_neurons)))
    else:
        threshold = 1  # This controls the relative rates of each spiking neuron
        resonators = np.zeros(len(embed_vector))
        for t in range(duration):
            resonators = resonators + embed_vector
            active_neurons = [i for i in range(len(resonators)) if resonators[i] >= threshold]
            for i in active_neurons:
                resonators[i] = resonators[i] - threshold
            neurons.extend(active_neurons)
            times.extend(([t] * len(active_neurons)))
    return slayer.io.Event(neurons, None, np.zeros(len(neurons)), times)


def load_real_embedding(embed_loc):
    word_to_index = {}
    index_to_word = []
    embeddings = []

    with open(embed_loc, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)

            word_to_index[word] = len(index_to_word)
            index_to_word.append(word)
            embeddings.append(vector)

    embeddings = np.stack(embeddings)

    return embeddings, word_to_index, index_to_word


def average_embedding(text, embed_func):
    # Tokenize the text
    tokens = nltk.tokenize.word_tokenize(text)
    # For every token, embed it (if possible) & add it to the running total
    active_tokens = []
    token_sum = []
    counter = 0
    for token in tokens:
        embed_token = embed_func(token)
        if embed_token.any():
            active_tokens.append(token)
            counter += 1
            if not len(token_sum):
                token_sum = embed_token
            else:
                token_sum = token_sum + embed_token
    # Return all values
    return (token_sum * (1 / counter)), active_tokens


class RealWordDataset(Dataset):
    def __init__(self, pairs, embed_loc="embed/real_valued/glove.6B.200d.txt", embed_dim=200, log_tokens=False):
        super(RealWordDataset, self).__init__()
        self.num_samples = len(pairs[pairs.columns[0]])
        self.max_timestep = MAX_T
        # Load the embedding
        self.embedding, self.word2index, self.index2word = load_real_embedding(embed_loc)
        self.embed_dim = embed_dim
        self.max_in_neuron_id = embed_dim
        # Initialize the token log

        # Process everything
        column = "text"
        if (True):
            arr = []
            for item in tqdm(pairs[column], desc=f"RealWordDataset: Formatting \"{column}\" column..."):
                vector, tokens = average_embedding(
                    item,
                    lambda x: (
                        self.embedding[self.word2index[x]] if x in self.word2index.keys() else np.zeros(embed_dim))
                )

                arr.append(vector)
            self.vectors = arr
        # If the network is an SNN, convert it to events
        if NETWORK_TYPE == "SNN":
            # If rate-coding is enabled, do that
            if RATE_CODING:
                arr = []
                for item in tqdm(self.vectors, desc=f"RealWordDataset: Rate-coding \"{column}\" column..."):
                    arr.append(rate_code(item, self.max_timestep))
                self.events = arr
        # Set the labels
        self.labels = torch.from_numpy(pairs["label"].values)
        # If we're profiling, get the average input spike count

    def __getitem__(self, index):
        if NETWORK_TYPE == "SNN":
            return (
                self.events[index].fill_tensor(torch.zeros(1, 1, self.max_in_neuron_id, self.max_timestep)).squeeze(),
                self.labels[index]
            )
        else:  # NETWORK_TYPE == "ANN":
            return (
                torch.tensor(self.vectors[index]),
                self.labels[index]
            )

    def __len__(self):
        return self.num_samples


def load_data():
    # Get the dataset
    data = {"train": {}, "test": {}}
    if DATASET == "imdb":
        data = load_dataset("imdb")  # IMDB Sentiment analysis dataset
        keys = ["train", "test"]
        cols = ["text", "label"]
        df = pd.DataFrame(columns=cols)

        for col in cols:
            series = pd.Series(name=col)
            for key in keys:
                series = pd.concat([series, pd.Series(data[key][col])], ignore_index=True)
            # df = pd.concat([df, series], axis=1, ignore_index=True)
            df[col] = series
        df["text"] = df["text"].apply(clean_text)
        data = df.sample(frac=1, random_state=2500)
        data = data.head(800)
        # data = torch.utils.data.ConcatDataset([data["train"], data["test"]])
    elif DATASET == "emotion":
        data = load_dataset("emotion")  # Emotion classification dataset
        keys = ["train", "validation", "test"]
        cols = ["text", "label"]
        df = pd.DataFrame(columns=cols)
        for col in cols:
            series = pd.Series(name=col)
            for key in keys:
                series = pd.concat([series, pd.Series(data[key][col])], ignore_index=True)
            # df = pd.concat([df, series], axis=1, ignore_index=True)
            df[col] = series
        df["text"] = df["text"].apply(clean_text)
        data = df.sample(frac=1, random_state=2500)
        data = data.head(800)
        # data = torch.utils.data.ConcatDataset([data["train"], data["validation"], data["test"]])
    elif DATASET == "topic":
        data = load_dataset("")  # Topic <classification> dataset
    # print(data)
    # At this point we have a shuffled dataframe with the data
    # Split the data into train, test, and valid sets
    train_data, temp_data = train_test_split(data, test_size=1 - DATA_SPLITS["train"], random_state=2500)
    test_data, valid_data = train_test_split(temp_data, test_size=(DATA_SPLITS["valid"] / (1 - DATA_SPLITS["train"])),
                                             random_state=2500)

    # Return the splits
    return train_data, valid_data, test_data


def data_0():
    global MODEL_SIZE
    device = torch.device("cpu")

    # Load the data
    train_data, valid_data, test_data = load_data()
    # Determine which dataset/network encoding methods to use

    train_data = RealWordDataset(train_data)
    test_data = RealWordDataset(test_data)

    train_loader = DataLoader(dataset=train_data, batch_size=15)
    test_loader = DataLoader(dataset=test_data, batch_size=15)

    return train_loader, test_loader, train_data.embed_dim, train_data.max_timestep





def clean_text(text):
    text = text.lower()
    text = text.replace("_", " ")
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = " ".join(text.split())
    return text


# --- Spike train generator using binary_embedding.py ---
def make_spikes_from_text(text, embedding, embed_dim, num_repeats, spike_freq, max_tokens, spike_rand=0,
                          return_neuron_timings=False):
    spike_rand = 0.05
    # Tokenize the text
    tokens = nltk.tokenize.word_tokenize(text)[:max_tokens]
    # For every token, embed it (if possible) & add it to the 'active_neurons' array
    active_tokens = []
    active_neurons_per_token = []
    counter = 0
    for token in tokens:
        embed_token = embed.embed_item(token, embedding)[0]
        if (embed_token):
            active_tokens.append(token)
            binary = embed_token
            active_neurons = []
            counter = 0
            while (binary):
                if (binary & 1):
                    active_neurons.append(counter)
                binary = binary >> 1
                counter += 1
            active_neurons_per_token.append(active_neurons)
    # Add an EOS token
    active_neurons_per_token.append(np.ones(embed_dim).tolist())
    # Now, build an event object using the active neurons
    neurons = []
    times = []
    time_counter = 0
    for active_neurons in active_neurons_per_token:
        for i in range(num_repeats):
            if not spike_rand:
                neurons.extend(active_neurons)
                times.extend([time_counter for x in range(len(active_neurons))])
            else:
                temp = random.sample(active_neurons, int(np.floor(len(active_neurons) * spike_rand)))
                neurons.extend(temp)
                times.extend([time_counter for x in range(len(temp))])
            time_counter += spike_freq
    # Return the event object
    if not return_neuron_timings:
        return slayer.io.Event(neurons, None, np.zeros(len(neurons)), times), time_counter, active_tokens
    else:
        return neurons, times




class LayerOfLain():
    """
    This class is used to instantiate the layer object in the SFF algorithm,
    provide a training function that is uniformly called by the network during training,
    and perform local training independently.

    Member variables:
    threshold_pos (float): used to determine whether goodness_pos is large enough and directly participate in training.
    threshold_neg (float): used to determine whether goodness_neg is small enough and directly participate in training.
    min_weight (float): Used to provide a hard bound when calling the STDP module.
    max_weight (float): Used to provide a hard bound when calling the STDP module.
    encoder (encoder): Poisson encoder, which converts traditional data into pulse shape data that conforms to Poisson distribution.
    time_step (int): The time step length of the simulation for each data sample.
    learning_rate (float): learning rate.
    pre_time_au (float): Spiking neural network hyperparameters, time constants related to membrane potential decay and STDP.
    post_time_au (float): spiking neural network hyperparameters, time constants related to membrane potential decay and STDP.
    learner (MSTDPLearner): reward-modulated STDP learner, called when the STDP module of SFF is started.
    """

    def __init__(self, N_input, N_output, pre_time_au=2.,
                 post_time_au=100., time_step=10,
                 batch_size=15, learning_rate=0.009, threshold_both=0.005):
        self.single_net = nn.Sequential(
            layer.Linear(N_input, N_output, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        ).to(device)
        self.threshold_pos = threshold_both
        self.threshold_neg = threshold_both
        self.min_weight = -1.
        self.max_weight = 1.
        self.encoder = encoding.PoissonEncoder()
        self.time_step = time_step
        self.learning_rate = learning_rate
        self.pre_time_au = pre_time_au
        self.post_time_au = post_time_au
        self.batch_size = batch_size
        self.N_output = N_output
        self.encoder = encoding.PoissonEncoder()
        self.learner = learning.MSTDPLearner(step_mode='s', batch_size=self.batch_size,
                                             synapse=self.single_net[0], sn=self.single_net[1],
                                             tau_pre=self.pre_time_au, tau_post=self.post_time_au,
                                             )
        self.learner.disable()

    def goodness_cal(self, output):
        goodness = output.pow(2).mean(1)
        # print(goodness)
        return goodness

    def reward_from_goodness(self, output, pos_flag):
        alpha_pos = 1.
        alpha_neg = 1.
        goodness = output.pow(2).mean(1)
        if (pos_flag == True):
            return alpha_pos * (goodness - self.threshold_pos)
        else:
            return alpha_neg * (self.threshold_neg - goodness)

    def forward_with_training(self, input_pos, input_neg, insight_pos, insight_neg, stdpflag=True):

        weight_opter_stdp = torch.optim.SGD(self.single_net.parameters(), lr=0.01, momentum=0.)
        weight_opter_surrogate = torch.optim.Adam(self.single_net.parameters(), lr=self.learning_rate)
        if (stdpflag == True):
            with torch.no_grad():
                self.learner.enable()
                reward_pos = 0.
                for t in range(self.time_step):
                    # Positive update
                    reward_pos = self.reward_from_goodness(self.single_net(input_pos[t]), True)

                    weight_opter_stdp.zero_grad()
                    self.learner.step(reward_pos, on_grad=True)
                    weight_opter_stdp.step()
                self.learner.reset()

                reward_neg = 0.
                for t3 in range(self.time_step):
                    # Negative update
                    reward_neg = self.reward_from_goodness(self.single_net(input_neg[t3]), False)

                    weight_opter_stdp.zero_grad()
                    self.learner.step(reward_neg, on_grad=True)
                    weight_opter_stdp.step()
                self.learner.reset()
                # torch.cuda.empty_cache()
                self.learner.disable()
            functional.reset_net(self.single_net)
            # torch.cuda.empty_cache()

        goodness_pos = 0.
        for t in range(self.time_step):
            # Positive update
            # print(input_pos.max())
            goodness_pos += self.goodness_cal(self.single_net(input_pos[t]))

        goodness_pos = goodness_pos / self.time_step

        goodness_neg = 0.
        for t3 in range(self.time_step):
            # Negative update
            goodness_neg += self.goodness_cal(self.single_net(input_neg[t3]))

        goodness_neg = goodness_neg / self.time_step

        combined_pos = self.threshold_pos - goodness_pos - insight_pos
        combined_neg = - self.threshold_neg + goodness_neg - insight_neg

        loss_mixed = torch.log(torch.exp(torch.cat([combined_pos, combined_neg])) + 1).mean()
        weight_opter_surrogate.zero_grad()
        loss_mixed.backward()
        weight_opter_surrogate.step()
        functional.reset_net(self.single_net)

    def forward_withOUT_training(self, input_pos, input_neg):
        total_output_pos_list = []
        total_output_neg_list = []
        for t2 in range(self.time_step):
            total_output_pos_list.append((self.single_net(input_pos[t2])).detach())
            total_output_neg_list.append((self.single_net(input_neg[t2])).detach())

        total_output_pos = torch.stack(total_output_pos_list, dim=0)
        total_output_neg = torch.stack(total_output_neg_list, dim=0)
        return total_output_pos, total_output_neg

    def forward_withOUT_training_single(self, input_pos, firstflag):
        total_output_pos_list = []
        spike_counts = []
        if(firstflag==0):
            for t2 in range(self.time_step):
                out = self.single_net(input_pos[t2]).detach()
                total_output_pos_list.append(out)
                spike_counts.append((out > 0).sum().item())  # Count spikes at this time step
            total_output_pos = torch.stack(total_output_pos_list, dim=0)
        else:
            for t2 in range(self.time_step):
                out = self.single_net(input_pos[t2]).detach()
                total_output_pos_list.append(out)
                spike_counts.append((out > 0).sum().item())  # Count spikes at this time step
            total_output_pos = torch.stack(total_output_pos_list, dim=0)


        return total_output_pos,spike_counts





class NetOfLain(torch.nn.Module):
    """
    This class is used to instantiate the net object in the SFF algorithm, coordinate and call the training functions of each layer during training, so that they can perform local training independently.

    Member variables:
    lain_layers (LayerOfLain list): used to store layers for constructing SFF spiking neural network.
    insight_pos (float): The key constant for SFF to realize layer collaboration, which is the sum of the goodness of each layer after positive data propagation.
    """

    def __init__(self, lain_dimension, batch_size):
        super().__init__()
        self.lain_layers = []
        self.insight_pos = 0.
        self.insight_neg = 0.
        for d in range(len(lain_dimension) - 1):
            if (d == 0):
                layer = LayerOfLain(lain_dimension[d], lain_dimension[d + 1], pre_time_au=2., post_time_au=100.,
                                    batch_size=batch_size)
                self.lain_layers.append(layer)
            else:
                layer = LayerOfLain(lain_dimension[d], lain_dimension[d + 1], pre_time_au=2., post_time_au=100.,
                                    learning_rate=0.004, threshold_both=0.04, batch_size=batch_size)
                self.lain_layers.append(layer)

    def network_train_layers(self, train_loader, epo):
        # torch.cuda.empty_cache()
        for i, lain_layer in enumerate(self.lain_layers):
            print('training layer', i, '...')
            for features, labels in tqdm(train_loader):
                if (i == 0):
                    break
                # torch.cuda.empty_cache()
                features, labels = features.to(device), labels.to(device)
                features_pos = features
                features_pos = features_pos.permute(2, 0, 1)  # Rearrange to (time_step, batch_size, embed_dim)
                rnd = torch.randperm(features.size(0))
                features_neg = features[rnd]
                features_neg = features_neg.permute(2, 0, 1)
                #   expected_time_step = 50
                #   features_pos = features_pos[:, :expected_time_step, :]  # Truncate to (batch_size, 50, embed_dim)
                #   features_neg = features_neg[:, :expected_time_step, :]  # Truncate to (batch_size, 50, embed_dim)
                features_pos = features_pos.to(device)
                features_neg = features_neg.to(device)
                del features, labels
                # torch.cuda.empty_cache()
                # features_pos = features_pos.transpose(0, 1)
                # features_neg = features_neg.transpose(0, 1)
                self.insight_pos = self.network_collaboration(features_pos)
                self.insight_neg = self.network_collaboration(features_neg)
                positive_hidden, negative_hidden = features_pos, features_neg
                if (i > 0):
                    for o in range(i):
                        positive_hidden, negative_hidden = self.lain_layers[o].forward_withOUT_training(positive_hidden,
                                                                                                        negative_hidden)
                        functional.reset_net(self.lain_layers[o].single_net)
                # torch.cuda.empty_cache()
                if (i == 0):
                    lain_layer.forward_with_training(positive_hidden, negative_hidden, self.insight_pos,
                                                     self.insight_neg, stdpflag=False)
                else:
                    lain_layer.forward_with_training(positive_hidden, negative_hidden, self.insight_pos,
                                                     self.insight_neg, stdpflag=True)

    def network_predict(self, input):
        every_labels_goodness = []
        for label in range(2):
            hidden = input.permute(2, 0, 1)  # Reshape to (time_step, batch_size, embed_dim)
            hidden = hidden.to(device)  # Move to the correct device
            # torch.cuda.empty_cache()
            every_layer_goodness = []
            for p, lain_layer in enumerate(self.lain_layers):
                hidden,_ = lain_layer.forward_withOUT_training_single(hidden, p)
                goodnesstem = []
                for t in range(lain_layer.time_step):
                    goodnesstem.append((hidden[t].pow(2).mean(1)).unsqueeze(0))
                every_layer_goodness += [(torch.cat(goodnesstem, dim=0)).sum(0)]
            every_labels_goodness += [sum(every_layer_goodness).unsqueeze(1)]
            del hidden
            # for lain_layer in self.lain_layers:
            # functional.reset_net(lain_layer.single_net)
            # torch.cuda.empty_cache()
        every_labels_goodness = torch.cat(every_labels_goodness, 1)
        return every_labels_goodness.argmax(1)

    def network_collaboration(self, input):
        hidden = input.clone()
        every_layer_goodness = []
        for p, lain_layer in enumerate(self.lain_layers):
            hidden,_ = lain_layer.forward_withOUT_training_single(hidden, p)
            goodnesstem = []
            for t in range(lain_layer.time_step):
                goodnesstem.append((hidden[t].pow(2).mean(1)).unsqueeze(0))
            every_layer_goodness += [(torch.cat(goodnesstem, dim=0)).sum(0)]
            functional.reset_net(lain_layer.single_net)
        del hidden
        # torch.cuda.empty_cache()
        return sum(every_layer_goodness)







if __name__ == "__main__":
    all_preds = []
    torch.manual_seed(1000)
    torch.cuda.empty_cache()
    batch_size = 15
    embed_dim = 200
    max_tokens = 10
    train_loader, test_loader, input_dim, max_timestep = data_0()
    # train_loader=train_loader.permute(0, 2, 1)
    for features, labels in train_loader:
        print("Features shape:", features.shape)  # Shape of the spike train batch
        print("Labels shape:", labels.shape)  # Shape of the labels batch
        print("Features:", features)  # Print the actual spike train tensor
        print("Labels:", labels)  # Print the actual labels
        break  # Exit after the first batch to avoid printing everything

    alice = NetOfLain([input_dim, 500, 500], batch_size)
    for epo in range(1):
        print("Epoch:", epo)
        # torch.cuda.empty_cache()
        alice.network_train_layers(train_loader, epo)
        countT = 0.
        lossT = 0.
        for test_x, test_y in test_loader:
            test_x, test_y = test_x.to(device), test_y.to(device)
            # test_x: [batch, max_timestep, embed_dim]
            pred = alice.network_predict(test_x)
            lossT += 1.0 - pred.eq(test_y).float().mean().item()
            countT += 1
            all_preds.extend(pred.cpu().numpy())
            for lain_layer in alice.lain_layers:
                functional.reset_net(lain_layer.single_net)
        print('test error:', lossT / countT)

    all_labels = []
    all_scores = []

    for test_x, test_y in test_loader:
        test_x, test_y = test_x.to(device), test_y.to(device)
        # Get raw goodness scores for all classes
        every_labels_goodness = []
        for label in range(2):
            hidden = test_x.permute(2, 0, 1)  # Reshape to (time_step, batch_size, embed_dim)
            hidden = hidden.to(device)  # Move to the correct device
            every_layer_goodness = []
            for p, lain_layer in enumerate(alice.lain_layers):
                hidden,_ = lain_layer.forward_withOUT_training_single(hidden, p)
                goodnesstem = []
                for t in range(lain_layer.time_step):
                    goodnesstem.append((hidden[t].pow(2).mean(1)).unsqueeze(0))
                every_layer_goodness += [(torch.cat(goodnesstem, dim=0)).sum(0)]
            every_labels_goodness += [sum(every_layer_goodness).unsqueeze(1)]
            del hidden
        every_labels_goodness = torch.cat(every_labels_goodness, 1)  # shape: [batch, num_classes]
        all_scores.append(every_labels_goodness.cpu())
        all_labels.extend(test_y.cpu().numpy())
        for lain_layer in alice.lain_layers:
            functional.reset_net(lain_layer.single_net)

    all_scores = torch.cat(all_scores, dim=0).numpy()
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Compute MRR
    ranks = []
    for i, logits in enumerate(all_scores):
        sorted_indices = np.argsort(logits)[::-1]  # Descending order
        label = all_labels[i]
        idx = np.where(sorted_indices == label)[0]
        if len(idx) == 0:
            continue  # skip if label not found
        rank = idx[0] + 1
        ranks.append(1.0 / rank)
    mrr = np.mean(ranks)

    print(f"MRR after alice3 combination: {mrr:.4f}")
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)



# import locale
# locale.getpreferredencoding = lambda: "UTF-8"
# !pip install spikingjelly
import torch
import torch.nn as nn
import spikingjelly
import torchvision
import torch.utils.data as data
from tqdm import tqdm
from spikingjelly.activation_based import neuron, layer, learning, surrogate, encoding, functional
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import nltk
import re
import string
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import binary_embedding as embed






class LayerOfLain():
    """
    This class is used to instantiate the layer object in the SFF algorithm,
    provide a training function that is uniformly called by the network during training,
    and perform local training independently.

    Member variables:
    threshold_pos (float): used to determine whether goodness_pos is large enough and directly participate in training.
    threshold_neg (float): used to determine whether goodness_neg is small enough and directly participate in training.
    min_weight (float): Used to provide a hard bound when calling the STDP module.
    max_weight (float): Used to provide a hard bound when calling the STDP module.
    encoder (encoder): Poisson encoder, which converts traditional data into pulse shape data that conforms to Poisson distribution.
    time_step (int): The time step length of the simulation for each data sample.
    learning_rate (float): learning rate.
    pre_time_au (float): Spiking neural network hyperparameters, time constants related to membrane potential decay and STDP.
    post_time_au (float): spiking neural network hyperparameters, time constants related to membrane potential decay and STDP.
    learner (MSTDPLearner): reward-modulated STDP learner, called when the STDP module of SFF is started.
    """

    def __init__(self, N_input, N_output, pre_time_au=2.,
                 post_time_au=100., time_step=10,
                 batch_size=15, learning_rate=0.009, threshold_both=0.005):
        self.single_net = nn.Sequential(
            layer.Linear(N_input, N_output, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        ).to(device)
        self.threshold_pos = threshold_both
        self.threshold_neg = threshold_both
        self.min_weight = -1.
        self.max_weight = 1.
        self.encoder = encoding.PoissonEncoder()
        self.time_step = time_step
        self.learning_rate = learning_rate
        self.pre_time_au = pre_time_au
        self.post_time_au = post_time_au
        self.batch_size = batch_size
        self.N_output = N_output
        self.encoder = encoding.PoissonEncoder()
        self.learner = learning.MSTDPLearner(step_mode='s', batch_size=self.batch_size,
                                             synapse=self.single_net[0], sn=self.single_net[1],
                                             tau_pre=self.pre_time_au, tau_post=self.post_time_au,
                                             )
        self.learner.disable()

    def goodness_cal(self, output):
        goodness = output.pow(2).mean(1)
        # print(goodness)
        return goodness

    def reward_from_goodness(self, output, pos_flag):
        alpha_pos = 1.
        alpha_neg = 1.
        goodness = output.pow(2).mean(1)
        if (pos_flag == True):
            return alpha_pos * (goodness - self.threshold_pos)
        else:
            return alpha_neg * (self.threshold_neg - goodness)

    def forward_with_training(self, input_pos, input_neg, insight_pos, insight_neg, stdpflag=True):

        weight_opter_stdp = torch.optim.SGD(self.single_net.parameters(), lr=0.01, momentum=0.)
        weight_opter_surrogate = torch.optim.Adam(self.single_net.parameters(), lr=self.learning_rate)
        if (stdpflag == True):
            with torch.no_grad():
                self.learner.enable()
                reward_pos = 0.
                for t in range(self.time_step):
                    # Positive update
                    reward_pos = self.reward_from_goodness(self.single_net(input_pos[t]), True)

                    weight_opter_stdp.zero_grad()
                    self.learner.step(reward_pos, on_grad=True)
                    weight_opter_stdp.step()
                self.learner.reset()

                reward_neg = 0.
                for t3 in range(self.time_step):
                    # Negative update
                    reward_neg = self.reward_from_goodness(self.single_net(input_neg[t3]), False)

                    weight_opter_stdp.zero_grad()
                    self.learner.step(reward_neg, on_grad=True)
                    weight_opter_stdp.step()
                self.learner.reset()
                # torch.cuda.empty_cache()
                self.learner.disable()
            functional.reset_net(self.single_net)
            # torch.cuda.empty_cache()

        goodness_pos = 0.
        for t in range(self.time_step):
            # Positive update
            # print(input_pos.max())
            goodness_pos += self.goodness_cal(self.single_net(input_pos[t]))

        goodness_pos = goodness_pos / self.time_step

        goodness_neg = 0.
        for t3 in range(self.time_step):
            # Negative update
            goodness_neg += self.goodness_cal(self.single_net(input_neg[t3]))

        goodness_neg = goodness_neg / self.time_step

        combined_pos = self.threshold_pos - goodness_pos - insight_pos
        combined_neg = - self.threshold_neg + goodness_neg - insight_neg

        loss_mixed = torch.log(torch.exp(torch.cat([combined_pos, combined_neg])) + 1).mean()
        weight_opter_surrogate.zero_grad()
        loss_mixed.backward()
        weight_opter_surrogate.step()
        functional.reset_net(self.single_net)

    def forward_withOUT_training(self, input_pos, input_neg):
        total_output_pos_list = []
        total_output_neg_list = []
        for t2 in range(self.time_step):
            total_output_pos_list.append((self.single_net(input_pos[t2])).detach())
            total_output_neg_list.append((self.single_net(input_neg[t2])).detach())

        total_output_pos = torch.stack(total_output_pos_list, dim=0)
        total_output_neg = torch.stack(total_output_neg_list, dim=0)
        return total_output_pos, total_output_neg

    def forward_withOUT_training_single(self, input_pos, firstflag):
        total_output_pos_list = []
        spike_counts = []
        if(firstflag==0):
            for t2 in range(self.time_step):
                out = self.single_net(input_pos[t2]).detach()
                total_output_pos_list.append(out)
                spike_counts.append((out > 0).sum().item())  # Count spikes at this time step
            total_output_pos = torch.stack(total_output_pos_list, dim=0)
        else:
            for t2 in range(self.time_step):
                out = self.single_net(input_pos[t2]).detach()
                total_output_pos_list.append(out)
                spike_counts.append((out > 0).sum().item())  # Count spikes at this time step
            total_output_pos = torch.stack(total_output_pos_list, dim=0)


        return total_output_pos,spike_counts





class NetOfLain(torch.nn.Module):
    """
    This class is used to instantiate the net object in the SFF algorithm, coordinate and call the training functions of each layer during training, so that they can perform local training independently.

    Member variables:
    lain_layers (LayerOfLain list): used to store layers for constructing SFF spiking neural network.
    insight_pos (float): The key constant for SFF to realize layer collaboration, which is the sum of the goodness of each layer after positive data propagation.
    """

    def __init__(self, lain_dimension, batch_size):
        super().__init__()
        self.lain_layers = []
        self.insight_pos = 0.
        self.insight_neg = 0.
        for d in range(len(lain_dimension) - 1):
            if (d == 0):
                layer = LayerOfLain(lain_dimension[d], lain_dimension[d + 1], pre_time_au=2., post_time_au=100.,
                                    batch_size=batch_size)
                self.lain_layers.append(layer)
            else:
                layer = LayerOfLain(lain_dimension[d], lain_dimension[d + 1], pre_time_au=2., post_time_au=100.,
                                    learning_rate=0.004, threshold_both=0.04, batch_size=batch_size)
                self.lain_layers.append(layer)

    def network_train_layers(self, train_loader, epo):
        # torch.cuda.empty_cache()
        for i, lain_layer in enumerate(self.lain_layers):
            print('training layer', i, '...')
            for features, labels in tqdm(train_loader):
                if (i > 0):
                    break
                # torch.cuda.empty_cache()
                features, labels = features.to(device), labels.to(device)
                features_pos = features
                features_pos = features_pos.permute(2, 0, 1)  # Rearrange to (time_step, batch_size, embed_dim)
                rnd = torch.randperm(features.size(0))
                features_neg = features[rnd]
                features_neg = features_neg.permute(2, 0, 1)
                expected_time_step = 10
                features_pos = features_pos[:, :expected_time_step, :]  # Truncate to (batch_size, 50, embed_dim)
                features_neg = features_neg[:, :expected_time_step, :]  # Truncate to (batch_size, 50, embed_dim)
                features_pos = features_pos.to(device)
                features_neg = features_neg.to(device)
                del features, labels
                # torch.cuda.empty_cache()
                # features_pos = features_pos.transpose(0, 1)
                # features_neg = features_neg.transpose(0, 1)
                self.insight_pos = self.network_collaboration(features_pos)
                self.insight_neg = self.network_collaboration(features_neg)
                positive_hidden, negative_hidden = features_pos, features_neg
                if (i > 0):
                    for o in range(i):
                        positive_hidden, negative_hidden = self.lain_layers[o].forward_withOUT_training(positive_hidden,
                                                                                                        negative_hidden)
                        functional.reset_net(self.lain_layers[o].single_net)
                # torch.cuda.empty_cache()
                if (i == 0):
                    lain_layer.forward_with_training(positive_hidden, negative_hidden, self.insight_pos,
                                                     self.insight_neg, stdpflag=False)
                else:
                    lain_layer.forward_with_training(positive_hidden, negative_hidden, self.insight_pos,
                                                     self.insight_neg, stdpflag=True)

    def network_predict(self, input):
        every_labels_goodness = []
        for label in range(2):
            hidden = input.permute(2, 0, 1)  # Reshape to (time_step, batch_size, embed_dim)
            hidden = hidden.to(device)  # Move to the correct device
            # torch.cuda.empty_cache()
            every_layer_goodness = []
            for p, lain_layer in enumerate(self.lain_layers):
                hidden,_ = lain_layer.forward_withOUT_training_single(hidden, p)
                goodnesstem = []
                for t in range(lain_layer.time_step):
                    goodnesstem.append((hidden[t].pow(2).mean(1)).unsqueeze(0))
                every_layer_goodness += [(torch.cat(goodnesstem, dim=0)).sum(0)]
            every_labels_goodness += [sum(every_layer_goodness).unsqueeze(1)]
            del hidden
            # for lain_layer in self.lain_layers:
            # functional.reset_net(lain_layer.single_net)
            # torch.cuda.empty_cache()
        every_labels_goodness = torch.cat(every_labels_goodness, 1)
        return every_labels_goodness.argmax(1)

    def network_collaboration(self, input):
        hidden = input.clone()
        every_layer_goodness = []
        for p, lain_layer in enumerate(self.lain_layers):
            hidden,_ = lain_layer.forward_withOUT_training_single(hidden, p)
            goodnesstem = []
            for t in range(lain_layer.time_step):
                goodnesstem.append((hidden[t].pow(2).mean(1)).unsqueeze(0))
            every_layer_goodness += [(torch.cat(goodnesstem, dim=0)).sum(0)]
            functional.reset_net(lain_layer.single_net)
        del hidden
        # torch.cuda.empty_cache()
        return sum(every_layer_goodness)


if __name__ == "__main__":

    all_preds = []
    torch.manual_seed(1000)
    # torch.cuda.empty_cache()
    alice2 = NetOfLain([200, 500, 500], batch_size)
    print(alice2.lain_layers[1].single_net[0].weight.data.max())
    alice3 = NetOfLain([200, 500, 500], batch_size)
    alice3.lain_layers[0] = alice2.lain_layers[0]
    alice3.lain_layers[1] = alice.lain_layers[1]

    for epo in range(1):
        print("Epoch:", epo)
        # torch.cuda.empty_cache()
        alice3.network_train_layers(train_loader, epo)
        countT = 0.
        lossT = 0.
        for test_x, test_y in test_loader:
            test_x, test_y = test_x.to(device), test_y.to(device)
            x = alice3.network_predict(test_x)
            lossT += 1.0 - x.eq(test_y).float().mean().item()
            countT += 1
            all_preds.extend(x.cpu().numpy())
            for lain_layer in alice3.lain_layers:
                functional.reset_net(lain_layer.single_net)
        print('test error:', lossT / countT)

    all_labels = []
    all_scores = []

    for test_x, test_y in test_loader:
        test_x, test_y = test_x.to(device), test_y.to(device)
        # Get raw goodness scores for all classes
        every_labels_goodness = []
        for label in range(2):
            hidden = test_x.permute(2, 0, 1)  # Reshape to (time_step, batch_size, embed_dim)
            hidden = hidden.to(device)  # Move to the correct device
            every_layer_goodness = []
            for p, lain_layer in enumerate(alice3.lain_layers):
                hidden,_ = lain_layer.forward_withOUT_training_single(hidden, p)
                goodnesstem = []
                for t in range(lain_layer.time_step):
                    goodnesstem.append((hidden[t].pow(2).mean(1)).unsqueeze(0))
                every_layer_goodness += [(torch.cat(goodnesstem, dim=0)).sum(0)]
            every_labels_goodness += [sum(every_layer_goodness).unsqueeze(1)]
            del hidden
        every_labels_goodness = torch.cat(every_labels_goodness, 1)  # shape: [batch, num_classes]
        all_scores.append(every_labels_goodness.cpu())
        all_labels.extend(test_y.cpu().numpy())
        for lain_layer in alice3.lain_layers:
            functional.reset_net(lain_layer.single_net)

    all_scores = torch.cat(all_scores, dim=0).numpy()
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Compute MRR
    ranks = []
    for i, logits in enumerate(all_scores):
        sorted_indices = np.argsort(logits)[::-1]  # Descending order
        label = all_labels[i]
        idx = np.where(sorted_indices == label)[0]
        if len(idx) == 0:
            continue  # skip if label not found
        rank = idx[0] + 1
        ranks.append(1.0 / rank)
    mrr = np.mean(ranks)

    print(f"MRR after alice3 combination: {mrr:.4f}")
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    total_spike_counts = [0 for _ in range(len(alice.lain_layers))]
    num_batches = 0


