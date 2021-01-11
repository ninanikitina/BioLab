import logging
import os

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from alexnet.eval import eval_binary
import torchvision.models.alexnet as alexnet
from torchvision.datasets import ImageFolder
from torchvision import transforms as tf

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

root_train_dir = '../src/data_binary/train'
root_val_dir = '../src/data_binary/val'
dir_checkpoint = '../src/checkpoints/'


def train_net(net,
              device,
              epochs=150,
              batch_size=4,
              lr=0.001,
              val_percent=0.1,
              save_cp=True):
    data_transforms = tf.Compose([tf.Resize(256),
                                  tf.ToTensor(),
                                  # tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                  ])

    train_dataset = ImageFolder(root_train_dir, transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)

    val_dataset = ImageFolder(root_val_dir, transform=data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=8)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0


    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Checkpoints:     {save_cp}
            Device:          {device.type}
        ''')

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=True)

    for epoch in range(epochs):
        # net.train_net()

        epoch_loss = 0
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch[0]
                labels = batch[1]

                imgs = imgs.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)

                labels_pred = torch.squeeze(net(imgs))
                loss = criterion(labels_pred, labels)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

        # Validation
        val_loss, val_acc = eval_binary(net, val_loader, device)
        scheduler.step(val_loss)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        logging.info('Validation cross entropy: {}'.format(val_loss))
        writer.add_scalar('Loss/test', val_loss, global_step)

        logging.info('Validation accuracy: {}'.format(val_acc))
        writer.add_scalar('Acc/test', val_acc, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass

            if epoch % 10 == 9:
                torch.save(net.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = alexnet(num_classes=1)
    net.to(device=device)
    train_net(net, device)
