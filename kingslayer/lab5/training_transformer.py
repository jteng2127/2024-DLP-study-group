import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, args, train_loader, epoch):
        self.model.train()
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch}")
        for i, img in enumerate(pbar):
            self.optim.zero_grad()
            
            img = img.to(args.device)
            logits, gt = self.model(img)
            # logits: bs, d, 1025
            # gt: bs, d
            
            # print("logits: ", logits.shape, "gt: ", gt.shape)
            logits = logits.reshape(-1, logits.shape[-1]) # bs*d, 1025
            gt = gt.reshape(-1) # bs*d
            
            loss = F.cross_entropy(logits, gt)
            
            loss.backward()
            self.optim.step(self.scheduler)
            
            pbar.set_postfix_str(f'Loss: {loss:.3f}')

    def eval_one_epoch(self, args, val_loader):
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(val_loader, total=len(val_loader))
            pbar.set_description(f"Validation ")
            t = 0
            loss_tot = 0.0
            for i, img in enumerate(pbar):
                t += 1
                
                img = img.to(args.device)
                logits, gt = self.model(img)
                
                # print("logits: ", logits.shape, "gt: ", gt.shape)
                logits = logits.reshape(-1, logits.size(-1))
                gt = gt.reshape(-1)
                # print("logits: ", logits.shape, "gt: ", gt.shape)
                loss = F.cross_entropy(logits, gt)
                
                pbar.set_postfix_str(f'Loss: {loss:.3f}')
                
                loss_tot += loss
            loss_tot = loss_tot / t
        
        return loss_tot

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.transformer.parameters(), lr=args.learning_rate)
        scheduler = None
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab5_dataset/cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)

    min_loss = float('inf')
#TODO2 step1-5:    
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_transformer.train_one_epoch(args, train_loader, epoch)
        loss = train_transformer.eval_one_epoch(args, val_loader)
        print(f'Validation loss on epoch {epoch}: {loss:.3f}')
        
        if epoch % args.save_per_epoch == 0:
            torch.save(train_transformer.model.transformer.state_dict(), os.path.join('transformer_checkpoints', f'transformer_epoch_{epoch}.pth'))
        
        if loss < min_loss:
            min_loss = loss
            torch.save(train_transformer.model.transformer.state_dict(), os.path.join('transformer_checkpoints', f'transformer_best.pth'))
            
        