import numpy as np
import torch
import tqdm
import hydra
import matplotlib.pyplot as plt
import dgl

from model.deeponet import GraphDeepOnet, NodeDeepOnet
from datasets.dts import WaterDataset, collate_fn
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits as bce_wl
from utils.lib import model_structure, model_save, model_load

@hydra.main("config", "base", None)
def main(cfg):
    work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    epochs = 500
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = NodeDeepOnet(3, 256, 1, activation = "gelu", dropout = 0.0, layer = 12).to(device)
    #model_load(net, "./outputs/2023-11-12/17-51-16/net_2_0.5.pkl")
    print("\n".join(model_structure(net)))
    dts = WaterDataset("../data/ice_16A_small_train.hdf5", testing = False)
    test_dts = WaterDataset("../data/ice_16A_small_test.hdf5")
    dtl = DataLoader(dts, batch_size = 32, shuffle = True, collate_fn = collate_fn)
    test_dtl = DataLoader(test_dts, batch_size= 32, collate_fn = collate_fn)
    opt = torch.optim.Adam([
        {'params': [p for p in net.parameters() if "brunch" in p.names], 'lr': 1e-3},
        {'params': [p for p in net.parameters() if "brunch" not in p.names], 'lr': 1e-4}
        ])
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, 0.95)
    lower = np.inf

    for epoch in range(epochs):
        bar = tqdm.tqdm(dtl)
        losses = []
        grads = []
        bar.set_description(f"TRAIN [{epoch+1}/{epochs}]")
        
        net.train()
        net.requires_grad_(True)
        #TRAIN
        for i, (file_name, known_graph, test_points, test_labels) in enumerate(bar):
            known_graph = known_graph.to(device, non_blocking = True)
            test_points = test_points.to(device, non_blocking = True)
            test_labels = test_labels.to(device, non_blocking = True)
            
            pred_labels = net(known_graph, test_points)

            loss = bce_wl(pred_labels, test_labels)
            
            opt.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            
            
            losses.append(loss.item())
            grads.append(grad.item())
            
            if i % 10 == 0:
                mean_loss = np.mean(losses)
                mean_grad = np.mean(grads)
                losses = []
                grads = []
                bar.set_postfix({'Loss': f"{mean_loss:.2e}", "Grad": f"{mean_grad:.2e}"})
        
        sch.step()
        bar.close()
        
        bar = tqdm.tqdm(test_dtl)
        bar.set_description(f"TEST [{epoch+1}/{epochs}]")
        losses = []
        net.eval()
        net.requires_grad_(False)
        #TEST
        for i, (file_name, known_graph, test_points, test_labels) in enumerate(bar):
            known_graph = known_graph.to(device, non_blocking = True)
            test_points = test_points.to(device, non_blocking = True)
            test_labels = test_labels.to(device, non_blocking = True)
            
            pred_labels = net(known_graph, test_points)

            loss = bce_wl(pred_labels, test_labels)
            
            losses.append(loss.item())
            if i % 10 == 0:
                bar.set_postfix({'Loss': loss.item()})
        
        bar.close()
        mean_loss = np.mean(losses)
        if mean_loss < lower:
            lower = mean_loss
            model_save(net, f"{work_dir}/net_{epoch+1}_{mean_loss:.2e}.pkl")
            saved = "saved"
        else:
            saved = "not saved"
            
        print(f"Testing [{epoch+1}/{epochs}]: Loss {mean_loss:.3f}, Model {saved}.")
        #SAMPLE
        net.eval()
        net.requires_grad_(False)
        file_name, known_graph, test_points, test_labels = test_dts[0]
        known_graph = known_graph.to(device)
        
        test_points = test_points[test_labels.squeeze()==1]
        sample_points = torch.rand(10000, 3, device = device) * 2 - 1
        g = dgl.graph(([], []), num_nodes = len(sample_points))
        g.ndata['pos'] = sample_points
        
        conf = net(known_graph, sample_points[None])
        conf = conf.sigmoid().squeeze()
        mask = conf > 0.5
        
        conf = conf[mask]
        sample_points = sample_points[mask]
        
        sample_points = sample_points[torch.argsort(conf, descending = True)]
        
        fig = plt.figure(figsize = (10, 5))
        ax1 = fig.add_subplot(121, projection = "3d")
        ax2 = fig.add_subplot(122, projection = "3d")
        
        ax1.set_title("Ground Truth")
        ax2.set_title("Prediction")
        
        known_graph = known_graph.to('cpu')
        
        known_points = known_graph.ndata['pos']
        ax = ax1.scatter(test_points[:, 0], test_points[:, 1], test_points[:, 2], c = "r", s = 1, alpha = 0.5, label = "all")
        ax = ax1.scatter(known_points[:, 0], known_points[:, 1], known_points[:, 2], c = "b", s = 1, alpha = 1.0, label = "known")
        
        ax = ax2.scatter(test_points[:, 0], test_points[:, 1], test_points[:, 2], c = "r", s = 1, alpha = 0.5, label = "all")
        ax = ax2.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], c = conf, s = 1, alpha = 0.3,label = "prediction")
        plt.colorbar(ax, ax = ax2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{work_dir}/sample_{epoch+1}_{mean_loss:.1f}.png")
        plt.close()

if __name__ == "__main__":
    main()