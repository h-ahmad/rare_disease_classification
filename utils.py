# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 23:34:28 2025

@author: hussain
"""

import os
import torch
from torchvision.models import maxvit_t, MaxVit_T_Weights
import clip
from model.ConvNet import ConvNet
from model.SwinTransformer import SwinTransformer
from model.clip import CLIP
from torchvision import models
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import timm
from torchvision.transforms import transforms
from sklearn.utils.multiclass import unique_labels

CLIP_MEAN=(0.48145466, 0.4578275, 0.40821073)
CLIP_STD=(0.26862954, 0.26130258, 0.27577711)

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

def get_transformation(opt):
    img_crop_size = 0.0
    if(opt.model == 'convnet'): 
        img_crop_size = 150 
    else: img_crop_size = 224
    
    if(opt.model == 'clip_with_lora' or opt.model == 'clip_without_lora'):
        transformer = transforms.Compose([        
            transforms.RandomResizedCrop(img_crop_size), # for convnet, 150, while for others, 224
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ])
    elif opt.model == 'resnet50':
        transformer = transforms.Compose([        
            transforms.RandomResizedCrop(img_crop_size), # for convnet, 150, while for others, 224
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
            ])
    else:
        transformer = transforms.Compose([        
            transforms.RandomResizedCrop(img_crop_size), # for convnet, 150, while for others, 224
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1]
                                [0.5,0.5,0.5]),
            ])
    
    return transformer

class CLIPFineTuner(nn.Module):
    def __init__(self, encoder, out_dim):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(512, out_dim)  # 512 for ViT-B/32

    def forward(self, x):
        x = self.encoder(x)  # [B, 512]
        return self.fc(x)

def compute_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def get_model(opt, device):
    if opt.model == 'convnet':
        model = ConvNet(num_classes=len(os.listdir(opt.dataset_train)))
    elif opt.model == 'maxvit':
        model = maxvit_t(weights=MaxVit_T_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                               torch.nn.Flatten(),
                                               torch.nn.LayerNorm(512),
                                               torch.nn.Linear(512, 512),
                                               torch.nn.Tanh(),
                                               torch.nn.Linear(512, len(os.listdir(opt.dataset_train)), bias=False),)
    elif opt.model == 'swin':
        model = SwinTransformer(num_classes=len(os.listdir(opt.dataset_train)))
    elif opt.model == 'efficientnet':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(os.listdir(opt.dataset_train)))
    elif opt.model == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(os.listdir(opt.dataset_train)))
    elif opt.model == 'mobilenetv2':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.last_channel, len(os.listdir(opt.dataset_train)))
    elif opt.model == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, len(os.listdir(opt.dataset_train)))
    elif opt.model == 'resnet101':
        model = models.resnet101(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(os.listdir(opt.dataset_train))) 
    elif opt.model == 'deit_large':
        model = timm.create_model('deit3_large_patch16_224', pretrained=True)
        model.head = nn.Linear(model.head.in_features, len(os.listdir(opt.dataset_train))) 
    elif opt.model == 'deit_base':
        model = timm.create_model('deit3_base_patch16_224', pretrained=True)
        model.head = nn.Linear(model.head.in_features, len(os.listdir(opt.dataset_train)))  # Adjust final layer
    elif opt.model == 'volo_d1_224':
        model = timm.create_model('volo_d1_224', pretrained=True)
        model.reset_classifier(num_classes=len(os.listdir(opt.dataset_train)))
    elif opt.model == 'clip_without_lora':
        model, preprocess = clip.load(opt.clip_version, device=device)
        image_encoder = model.visual  
        for param in image_encoder.parameters():
            param.requires_grad = True      
        model = CLIPFineTuner(image_encoder, out_dim = len(os.listdir(opt.dataset_train)))
        model = model.to(device).float()
    elif opt.model == 'clip_with_lora':
        model = CLIP(dataset=opt.dataset_name, is_lora_image=opt.is_lora_image, is_lora_text=opt.is_lora_text,
                     clip_download_dir=opt.clip_download_dir, clip_version=opt.clip_version,)
        model = model.to(device).float()
    
    if (opt.model != 'clip_without_lora' and opt.model != 'clip_with_lora'):
        for param in model.parameters():
            param.requires_grad = True          
        model = model.to(device)
        
    return model

def train(model_name, model, train_loader, validate_loader, loss_fn, optimiser, num_epochs,
          batch_size, learning_rate, device, log_dir, scheduler):
    max_val_acc = 0.0
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    bestEpoch = 0
    print('--------------------------------------------------------------')
    # Loop along epochs to do the training
    for i in range(num_epochs):
        train_accuracy = 0.0
        train_loss = 0.0
        print('\nTRAINING')     
        model.train()
        for images, labels in train_loader:
            print('\rEpoch[' + str(i+1) + '/' + str(num_epochs) + ']: ', end='')
            images, labels = images.to(device), labels.to(device)
            labels = labels.to(device)
            optimiser.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimiser.step()
            # scheduler.step()
            train_accuracy += compute_accuracy(predictions, labels).item()
            train_loss += loss.item()
            
        scheduler.step()
        # validation
        val_accuracy = 0.0
        val_loss = 0.0
        model.eval()
        print('\n\nVALIDATION')
        for images, labels in validate_loader:
            print('\rEpoch[' + str(i+1) + '/' + str(num_epochs) + ']: ', end='')            
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            val_accuracy += compute_accuracy(predictions, labels).item()
            val_loss += loss.item()
            
        train_accs.append(train_accuracy / len(train_loader))
        val_accs.append(val_accuracy / len(validate_loader))
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(validate_loader))
            
        if (val_accuracy / len(validate_loader)) > max_val_acc:
            torch.save(model.state_dict(), log_dir+"/"+model_name+"/checkpoint.pth")
            print('\n')
            print(f'- Train Acc: {(train_accuracy / len(train_loader))*100:.2f}%')
            print(f'- Val Acc: {(val_accuracy / len(validate_loader))*100:.2f}%')
            print(f'- Train Loss: {train_loss / len(train_loader):.3f}')
            print(f'- Val Loss: {val_loss / len(validate_loader):.3f}')
            print(f'\nAccuracy increased ({max_val_acc*100:.6f}% ---> {(val_accuracy / len(validate_loader))*100:.6f}%) \nModel saved')
            max_val_acc = val_accuracy / len(validate_loader)
            bestEpoch = i
        print("--------------------------------------------------------------")
        # Save losses and accuracies
        # max_val_acc #.... within each epoch
    # out of epoch
    plot_train_acc_loss(log_dir, model_name, train_accs, val_accs, train_losses, val_losses, bestEpoch)
    
def test(output_path, model_name, model, dataloader, device, num_classes, is_multiclass=False):
    state_dict = torch.load(output_path+"/"+model_name+'/checkpoint.pth', map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    test_accuracy = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    # Convert to numpy arrays
    # all_labels = torch.tensor(all_labels).numpy()
    # all_preds = torch.tensor(all_preds).numpy()
    # all_probs = torch.tensor(all_probs).numpy()
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    # === Classification Metrics ===
    averaging = 'binary' 
    if num_classes > 2:
        averaging = 'macro'
    f1 = f1_score(all_labels, all_preds, average=averaging)
    precision = precision_score(all_labels, all_preds, average=averaging)
    recall = recall_score(all_labels, all_preds, average=averaging)
    accuracy = accuracy_score(all_labels, all_preds)
    try:
        # For multiclass AUROC, we use one-vs-rest
        if num_classes > 2:
            from sklearn.preprocessing import label_binarize
            binary_labels = label_binarize(all_labels, classes=list(range(num_classes)))
            auroc = roc_auc_score(binary_labels, all_probs, average='macro', multi_class='ovr')
        else:
            auroc = roc_auc_score(all_labels, [p[1] for p in all_probs])
    except Exception as e:
        auroc = f"Could not calculate AUROC: {str(e)}"

    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n=== Evaluation Metrics ===")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"AUROC        : {auroc}")
    print(f"Confusion Matrix:\n{cm}")  
    
    metrics_file = os.path.join(output_path, model_name, "evaluation_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUROC: {auroc:.4f}\n")
        print(f"Metrics saved to {metrics_file}")
        
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['colo_aca', 'colon_n'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels(all_labels))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.grid(False)
    plt.savefig(os.path.join(output_path, model_name, 'confusion_matrix.pdf'))
        
def plot_train_acc_loss(log_dir, model_name, train_acc, val_acc, train_loss, val_loss, best_epoch):
    epochs = np.arange(torch.tensor(train_loss).shape[0])
    
    plt.figure()
    plt.plot(epochs, train_loss, label="Training loss", c='b')
    plt.plot(epochs, val_loss, label="Validation loss", c='r')
    plt.plot(best_epoch, val_loss[best_epoch], label="Best epoch", c='y', marker='.', markersize=10)
    plt.text(best_epoch+.01, val_loss[best_epoch]+.01, str(best_epoch) + ' - ' + str(round(val_loss[best_epoch], 3)), fontsize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss along epochs')
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(log_dir, model_name, 'loss.pdf'))
    
    plt.figure()
    plt.plot(epochs, train_acc, label="Training accuracy", c='b')
    plt.plot(epochs, val_acc, label="Validation accuracy", c='r')
    plt.plot(best_epoch, val_acc[best_epoch], label="Best epoch", c='y', marker='.', markersize=10)
    plt.text(best_epoch+.001, val_acc[best_epoch]+.001, str(best_epoch) + ' - ' + str(round(val_acc[best_epoch], 3)), fontsize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy along epochs')
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(log_dir, model_name, 'accuracy.pdf'))
    
    # plt.show()
        