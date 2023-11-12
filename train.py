import numpy as np
import torch
import tqdm
import hydra
from model.deeponet import GraphDeepOnet, NodeDeepOnet
from datasets.dts import WaterDataset, collate_fn
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits as bce_wl
from utils.lib import model_structure, model_save

@hydra.main("config", "base", None)
def main(cfg):
    work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    epochs = 500
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = NodeDeepOnet(3, 256, 1, activation = "gelu", dropout = 0.0, layer = 6).to(device)
    print("\n".join(model_structure(net)))
    dts = WaterDataset("../data/ice_16A/ice_16A_small_train.hdf5", testing = False)
    test_dts = WaterDataset("../data/ice_16A/ice_16A_small_test.hdf5")
    dtl = DataLoader(dts, batch_size = 8, shuffle = True, collate_fn = collate_fn)
    test_dtl = DataLoader(test_dts, batch_size= 32, collate_fn = collate_fn)
    opt = torch.optim.Adam(net.parameters(), lr = 1e-4)
    lower = np.inf

    for epoch in range(epochs):
        bar = tqdm.tqdm(dtl)
        losses = []
        bar.set_description(f"TRAIN [{epoch+1}/{epochs}]")
        
        net.train()
        net.requires_grad_(True)
        
        for i, (file_name, gk, gp, gn) in enumerate(bar):
            gk = gk.to(device)
            gp = gp.to(device)
            gn = gn.to(device)
            pos_pos = net(gk.ndata['pos'], gp.ndata['pos'], gk, gp)
            neg_pos = net(gk.ndata['pos'], gn.ndata['pos'], gk, gn)
            
            loss = bce_wl(pos_pos, torch.ones_like(pos_pos)) + \
                bce_wl(neg_pos, torch.zeros_like(neg_pos))
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            losses.append(loss.item())
            
            if i % 10 == 0:
                mean_loss = np.mean(losses)
                losses = []
                bar.set_postfix({'Loss': f"{mean_loss:.2e}"})
        
        bar.close()
        
        bar = tqdm.tqdm(test_dtl)
        bar.set_description(f"TEST [{epoch+1}/{epochs}]")
        losses = []
        net.eval()
        net.requires_grad_(False)
        
        for i, (file_name, gk, gp, gn) in enumerate(bar):
            gk = gk.to(device)
            gp = gp.to(device)
            gn = gn.to(device)
            pos_pos = net(gk.ndata['pos'], gp.ndata['pos'], gk, gp)
            neg_pos = net(gk.ndata['pos'], gn.ndata['pos'], gk, gn)
            
            loss = bce_wl(pos_pos, torch.ones_like(pos_pos)) + \
                bce_wl(neg_pos, torch.zeros_like(neg_pos))
            
            losses.append(loss.item())
            if i % 10 == 0:
                bar.set_postfix({'Loss': loss.item()})
        
        bar.close()
        mean_loss = np.mean(losses)
        if mean_loss < lower:
            lower = mean_loss
            model_save(net, f"{work_dir}/net_{epoch+1}_{mean_loss:.1f}.pkl")
            saved = "saved"
        else:
            saved = "not saved"
            
        
        print(f"Testing [{epoch+1}/{epochs}]: Loss {mean_loss:.3f}, Model {saved}.")

if __name__ == "__main__":
    main()