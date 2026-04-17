import torch
import wandb
from tqdm import tqdm
import argparse
import yaml
from src.model import Miniflow
from src.dataset import MoonDataset
from src.utils import show_field

def main():

    # args
    wandb.init()
    parser = argparse.ArgumentParser(description="Miniflow arguments")
    parser.add_argument("--config_path", type=str, default="./mini_flow/configs/miniflow.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path))
    config["config_path"] = args.config_path


    # setup training
    data = MoonDataset(config)
    model = Miniflow(config).to("cuda")
    optimizer = torch.optim.AdamW(lr=float(config["training"]["lr"]),
                                  params=model.parameters(),)
    dataloader = torch.utils.data.DataLoader(dataset=data,
                                             shuffle=True,
                                             batch_size=config["training"]["batch_size"],
                                             )


    # train
    model.to("cuda")
    for epoch in range(config["training"]["epochs"]):
        # eval every 10th one
        if epoch % 10 == 0:
            show_field(model, epoch)
        # train velocity field
        epoch_loss = 0
        n_samples = 0
        for x_1 in tqdm(dataloader, f"train epoch {epoch}"):
            x_1 = x_1.cuda()
            optimizer.zero_grad()
            b = x_1.shape[0]

            t = torch.rand(b, 1, device="cuda")
            x_0 = torch.randn(b, 2, device="cuda")
            x_t = (1-t)*x_0 + t*x_1

            input = torch.cat([x_t, t], dim=1)
            vel_pred = model(input)
            vel_target = x_1 - x_0

            loss = ((vel_pred - vel_target)**2).mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_samples += b

        wandb.log({"loss": epoch_loss / n_samples})
        print("loss:", epoch_loss / n_samples)
    
    torch.save(model, f"./model_{epoch}_{epoch_loss:04.4f}.pth")
    pass



    
    
            

if __name__ == "__main__":
    main()