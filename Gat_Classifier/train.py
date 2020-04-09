
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from model import GAT
from comments import build_karate_club_graph


embed = nn.Embedding(34, 5)  # 34 nodes with embedding dim equal to 5
inputs = embed.weight
labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
labels = torch.tensor([0, 1])  # their labels are different

edge_index = torch.from_numpy(build_karate_club_graph()).long()

net = GAT(4, 5, 5, 2)

optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
all_logits = []
for epoch in range(5000):
    logits = net(inputs,edge_index)
    # we save the logits for visualization later
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
