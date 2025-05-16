import torch
import time
from tqdm import tqdm
from utils import l2loss_sphere, spectral_l2loss_sphere
def train_model(model, solver, dataloader, optimizer, scheduler=None, 
                nepochs=20, nfuture=0, num_examples=256, num_valid=8, 
                loss_fn_name='l2', enable_amp=False, device='cpu'):

    train_start = time.time()

    for epoch in tqdm(range(nepochs)):
        epoch_start = time.time()

        # Set dataset for training phase
        dataloader.dataset.set_initial_condition('random')
        dataloader.dataset.set_num_examples(num_examples)

        # Training
        acc_loss = 0
        model.train()
        for inp, tar in dataloader:
            inp, tar = inp.to(device), tar.to(device) 
            
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=enable_amp):
                prd = model(inp)
                for _ in range(nfuture):
                    prd = model(prd)
                
                if loss_fn_name == 'l2':
                    loss = l2loss_sphere(solver, prd, tar)
                elif loss_fn_name == "spectral_l2":
                    loss = spectral_l2loss_sphere(solver, prd, tar)
                else:
                    raise ValueError(f"Unknown loss function: {loss_fn_name}")

            loss.backward()
            optimizer.step()
            
            acc_loss += loss.item() * inp.size(0)

        if scheduler is not None:
            scheduler.step()

        acc_loss = acc_loss / len(dataloader.dataset)

        # Set dataset for validation phase
        dataloader.dataset.set_initial_condition('random') 
        dataloader.dataset.set_num_examples(num_valid)

        # Validation
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for inp, tar in dataloader:
                inp, tar = inp.to(device), tar.to(device)
                
                with torch.amp.autocast(device_type=device.type, enabled=enable_amp):
                    prd = model(inp)
                    for _ in range(nfuture):
                        prd = model(prd)
                    
                    # Validation loss is l2loss_sphere with relative=True as per original
                    loss = l2loss_sphere(solver, prd, tar, relative=True)

                valid_loss += loss.item() * inp.size(0)
        
        valid_loss = valid_loss / len(dataloader.dataset) 

        epoch_time = time.time() - epoch_start

        print(f'--------------------------------------------------------------------------------')
        print(f'Epoch {epoch+1}/{nepochs} summary:')
        print(f'Time taken: {epoch_time:.2f} seconds')
        print(f'Accumulated training loss ({loss_fn_name}): {acc_loss:.4f}')
        print(f'Relative validation loss (l2): {valid_loss:.4f}')

    train_time = time.time() - train_start
    print(f'--------------------------------------------------------------------------------')
    print(f'Done. Training took {train_time:.2f} seconds.')
    
    return valid_loss # Return last validation loss