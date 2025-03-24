#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

import os
from helper_code import *
import numpy as np
import torch
import torch.nn.functional as F

from config import *
from data import Preprocessor, PCGDataset
from torch.utils.data import DataLoader
from HMSSNet import Hierachical_MS_Net
from utils import AverageMeter, calc_accuracy, load_patient_features
from loss import LabelSmoothingCrossEntropy

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    verbose = verbose >= 1
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build Datasets and Loaders
    if verbose: 
        print('Loading datasets...')
    train_preprocessor = Preprocessor(**PREPROCESSING_CFG, 
                                      mode = 'train')
    
    train_dataset = PCGDataset(data_folder, 
                               preprocessor = train_preprocessor, 
                               classes = DATASET_CFG['systolic_murmur_class'],
                               target = 'Systolic murmur timing')
    train_loader = DataLoader(train_dataset, 
                              shuffle=True,
                              drop_last=True, 
                              **DATALOADER_CFG)
    
    if verbose:
        print('Building up Torch CNN and optimizer...')
    timing_classifier = Hierachical_MS_Net(num_classes=DATASET_CFG['num_systolic_murmur_class'], **MODEL_CFG).to(device)
    optimizer = torch.optim.AdamW(timing_classifier.parameters(), **OPTIMIZER_CFG)
    criterion = LabelSmoothingCrossEntropy(TRAINING_CFG['label_smoothing'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, min_lr=1e-7, verbose=verbose)

    # Stage 1: Train the classifier for timing classification
    if verbose:
        print('Training model for Systolic murmur timing classification...')
    for epoch in range(TRAINING_CFG['epochs']):
        if verbose:
            print(f'Epoch {epoch} starts...')
        train_epoch(train_loader, timing_classifier, optimizer, criterion, scheduler, device, TRAINING_CFG['print_freq'])
        if verbose:
            print('\n')
        save_challenge_model(model_folder, timing_classifier, file_name='timing_classifier')
        
    # Stage 2: Train the classifier for Outcome classification
 
        
    if verbose:
        print('Done.')
            
def train_epoch(dataloader, model, optimizer, criterion, scheduler=None, device='cuda', print_freq=10, verbose=False):
    model.train()
    acc_meter, loss_meter = AverageMeter(), AverageMeter()

    for i, (multi_scale_specs, patient_features, targets) in enumerate(dataloader):
        multi_scale_specs = [s.to(device) for s in multi_scale_specs]
        patient_features = patient_features.to(device)
        targets = targets.to(device)
        batch_size = targets.size(0)
        preds = model(multi_scale_specs, patient_features)
        targets = targets.to(torch.int64)  # Ensure targets are integers

        batch_loss = criterion(preds, targets)
        batch_acc = calc_accuracy(preds, targets)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        loss_meter.update(batch_loss.item(), batch_size)
        acc_meter.update(batch_acc.item(), batch_size)
            
        if verbose and i != 0 and i % print_freq == 0:
            print(f'Training Iteration: {i} '\
                  f'Loss: {loss_meter.avg:.6f} \t'\
                  f'Accuracy: {acc_meter.avg*100:.4f}')
            
    print(f'Training Loss: {loss_meter.avg:.6f} \t'\
          f'Accuracy: {acc_meter.avg*100:.4f}')
            
    if scheduler:
        scheduler.step(loss_meter.avg)


def calc_pred_locations(preds, window_size=3, interval=0.5, freq=2000):
    interval = int(interval * freq)
    window_size = int(window_size * freq)
    recording_length = window_size + (preds.shape[0] - 1) * interval
    unknown_probs = []
    
    location_preds = np.zeros((recording_length, preds.shape[1]))
    for i in range(len(preds)):
        location_preds[i*interval: i*interval + window_size, :] += preds[i]
    location_preds = np.argmax(location_preds, -1)
    return location_preds


@torch.no_grad()
def recording_timing_diagnose(multi_scale_specs, timing_classifier, systolic_murmur_class, interval):
    timing_logits = timing_classifier(multi_scale_specs)
    timing_probs = F.softmax(timing_logits, -1).cpu().numpy()
    location_preds = calc_pred_locations(timing_probs, 
                                        window_size=PREPROCESSING_CFG['length'],
                                        interval=interval,
                                        freq=PREPROCESSING_CFG['frequency'])
    class_duration = np.bincount(location_preds, minlength=len(systolic_murmur_class)) / PREPROCESSING_CFG['frequency']
    
    if class_duration[1] / sum(class_duration) > 0.8:
        pred = 1
    else:
        if class_duration[0] >= 3:
            pred = 0
        else:
            pred = 2
    return pred
    

@torch.no_grad()
def run_challenge_model(model, data, recordings, verbose):
    (device, preprocessor, timing_classifier, systolic_murmur_class) = model  # Removed outcome-related components
    interval = 1.0
    recording_timing_counts = np.zeros(len(systolic_murmur_class), dtype=np.int_)

    patient_features = torch.from_numpy(load_patient_features(data)).unsqueeze(0).to(device)
    recording_timing_preds = np.zeros(len(recordings), dtype=np.int_)

    for i in range(len(recordings)):
        multi_scale_specs, timings = preprocessor(recordings[i], 4000, interval=interval)
        multi_scale_specs = [s.to(device) for s in multi_scale_specs]
        recording_timing_preds[i] = recording_timing_diagnose(multi_scale_specs, timing_classifier, systolic_murmur_class, interval)

    recording_timing_counts = np.bincount(recording_timing_preds, minlength=len(systolic_murmur_class))

    # Assign murmur labels based on the most frequent prediction
    timing_labels = np.zeros(len(systolic_murmur_class), dtype=np.int_)
    timing_labels[np.argmax(recording_timing_counts)] = 1
    classes=systolic_murmur_class
    probabilities = recording_timing_counts / np.sum(recording_timing_counts) if np.sum(recording_timing_counts) > 0 else np.zeros(len(systolic_murmur_class))

    return classes, timing_labels, probabilities
    
# ################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, model, file_name='timing_classifier'):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, os.path.join(model_folder, f'{file_name}.pth'))
    

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    preprocessor = Preprocessor(mode='test', **PREPROCESSING_CFG)
    
    timing_checkpoint = torch.load(os.path.join(model_folder, 'timing_classifier.pth'), map_location=device)
    timing_classifier = Hierachical_MS_Net(num_classes=DATASET_CFG['num_systolic_murmur_class'], **MODEL_CFG).to(device)
    timing_classifier.load_state_dict(timing_checkpoint['model_state_dict'])
    timing_classifier.eval()
    

    
    systolic_murmur_class = DATASET_CFG['systolic_murmur_class']
   
    
    return (device, preprocessor, timing_classifier, systolic_murmur_class)
