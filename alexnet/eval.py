import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def eval_binary(net, loader, device):
    net.eval()
    label_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot_loss = 0
    tot_misses = 0
    examples = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch[0]
            labels = batch[1]

            imgs = imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=label_type)

            with torch.no_grad():
                labels_pred = torch.squeeze(net(imgs))

            tot_loss += F.binary_cross_entropy_with_logits(labels_pred, labels).item()

            labels_pred_human = (nn.Sigmoid()(labels_pred) > 0.5).type(torch.int)
            tot_misses += np.count_nonzero(torch.abs(labels - labels_pred_human).cpu().numpy())
            examples += len(labels)

            pbar.update()

    return tot_loss / n_val, 1 - (tot_misses * 1.0/examples)