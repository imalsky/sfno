import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch_harmonics.examples.sfno import PdeDataset
import json
import os

from model import SFNO
from train import train_model

def main():
    # Load configuration
    config_path = 'inputs/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_p = config['model_params']
    train_p = config['training_params']
    data_p = config['dataset_params']
    enable_amp = config['enable_amp']
    num_examples = 32

    # Create output directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('model', exist_ok=True)

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)
    print(f"Using device: {device}")

    # Dataset and DataLoader
    dt_hours = data_p['dt_hours']
    dt_solver_seconds = data_p['dt_solver_seconds']
    
    dt = dt_hours * 3600
    nsteps = dt // dt_solver_seconds

    if model_p['img_size'][0] != data_p['dims'][0] or model_p['img_size'][1] != data_p['dims'][1]:
        print(f"Warning: Model img_size {model_p['img_size']} and dataset dims {data_p['dims']} differ.")
        print(f"Using dataset dims {data_p['dims']} for model img_size for consistency.")
        model_p['img_size'] = data_p['dims']

    dataset = PdeDataset(dt=dt,
                         nsteps=nsteps,
                         dims=tuple(data_p['dims']),
                         device=device,
                         normalize=True,
                         num_examples=num_examples)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, persistent_workers=False)
    solver = dataset.solver.to(device) 

    nlat = dataset.nlat
    nlon = dataset.nlon

    model_p['img_size'] = (nlat, nlon) 
    model = SFNO(
        img_size=tuple(model_p['img_size']),
        grid=model_p['grid'],
        num_layers=model_p['num_layers'],
        scale_factor=model_p['scale_factor'],
        embed_dim=model_p['embed_dim'],
        big_skip=model_p['big_skip'],
        pos_embed=model_p['pos_embed'],
        use_mlp=model_p['use_mlp'],
        normalization_layer=model_p['normalization_layer']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_p['lr'], weight_decay=train_p['weight_decay'])
    
    torch.manual_seed(333)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(333)

    print("Starting training...")
    train_model(
        model=model,
        solver=solver,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=None,
        nepochs=train_p['nepochs'],
        nfuture=train_p['nfuture'],
        num_examples=train_p['num_examples'],
        num_valid=train_p['num_valid'],
        loss_fn_name=train_p['loss_fn'],
        enable_amp=enable_amp,
        device=device
    )

    model_save_path = 'model/final_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    print("Running inference for saving sample data...")
    dataloader.dataset.set_initial_condition('random') 
    dataloader.dataset.set_num_examples(4) 

    torch.manual_seed(0) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    with torch.inference_mode():
        inp, tar = next(iter(dataloader))
        inp, tar = inp.to(device), tar.to(device)
        out = model(inp).detach()

    sample_data_path = 'data/data.pt'
    torch.save({'input': inp.cpu(), 'target': tar.cpu(), 'prediction': out.cpu()}, sample_data_path)
    print(f"Sample inference data saved to {sample_data_path}")

    print("Main script finished. Model and sample data saved.")

if __name__ == '__main__':
    main()