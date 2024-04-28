from diffusers import DDPMScheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from tqdm.auto import tqdm
import argparse
from data.dataset import Nomad_Dataset
from model.nomad import *

lr = 1e-4
epoch = 30
batch_size = 64
val_ratio=0.2
alpha = 1e-4
k = 10
seed = 42
dir = '/content/go_stanford_5'
log_path = r'/content/drive/MyDrive/nomad_log'

def train(lr, epoch, batch_size, val_ratio, alpha, k, seed, dir, log_path):
    train_set = Nomad_Dataset(dir, len_traj_pred=8, ct_len = 6, max_distance=20, mode='train',val_ratio=val_ratio, seed=seed)
    valid_set = Nomad_Dataset(dir, len_traj_pred=8, ct_len = 6, max_distance=20, mode='val',val_ratio=val_ratio, seed=seed)
    train_lens = len(train_set)
    valid_lens = len(valid_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)
    train_loader_len = len(train_loader.dataset)//256
    writer = SummaryWriter(log_path)

    encoder_net, dist_net, noise_net = VisionEncoder(), DistPredNet(), NoisePredNet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder_net.to(device)
    dist_net.to(device)
    noise_net.to(device)

    optimizer = optim.AdamW([*encoder_net.parameters(),
                            *dist_net.parameters(),
                            *noise_net.parameters()], lr=lr)

    noise_scheduler = DDPMScheduler(num_train_timesteps=k)

    for i in range(epoch):

        encoder_net.train()
        dist_net.train()
        noise_net.train()

        for _iter, data in enumerate(tqdm(train_loader)):
            obs_imgs, goal_img, actions, distance, goal_mask =data
            obs_imgs = obs_imgs.to(device)
            goal_img = goal_img.to(device)
            actions = actions.to(device)
            distance = distance.to(device)
            goal_mask = goal_mask.to(device)
            optimizer.zero_grad()

            ct = encoder_net(obs_imgs,goal_img,goal_mask)
            dist_logit = dist_net(ct)

            noise = torch.randn(actions.shape).to(device)
            timesteps = torch.randint(0,k,(actions.shape[0],),device=device)
            noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

            noise_pred = noise_net(noisy_actions, ct, timesteps)

            loss = alpha*(F.l1_loss(dist_logit,distance))+(1-alpha)*(F.mse_loss(noise_pred,noise))

            writer.add_scalar("Loss/train", loss, i*train_loader_len+_iter+1)
            writer.flush()

            loss.backward()
            optimizer.step()

        encoder_net.eval()
        dist_net.eval()
        noise_net.eval()

        count_ = 0
        cos_sim = 0
        with torch.no_grad():
            for data in tqdm(valid_loader):
                obs_imgs, goal_img, actions, distance, goal_mask =data
                obs_imgs = obs_imgs.to(device)
                goal_img = goal_img.to(device)
                actions = actions.to(device)
                distance = distance.to(device)
                goal_mask = goal_mask.to(device)

                ct = encoder_net(obs_imgs,goal_img,goal_mask)

                noise = torch.randn(actions.shape).to(device)

                for k_tmp in noise_scheduler.timesteps[:]:
                    timestep=k_tmp.unsqueeze(-1).repeat(noise.shape[0]).to(device)
                    noise_pred = noise_net(noise, ct, timestep)

                    # inverse diffusion step (remove noise)
                    noise = noise_scheduler.step(model_output=noise_pred,
                                                timestep=k_tmp,
                                                sample=noise).prev_sample

                pred_pos = torch.cumsum(noise,dim=1)
                pos = torch.cumsum(actions,dim=1)

                cosine_similarity = F.cosine_similarity(pred_pos, pos,dim=-1).mean()
                cos_sim += cosine_similarity.item()
                count_ +=1
        cos_sim_acc = cos_sim/count_
        writer.add_scalar('cos_sim_acc', cos_sim_acc, i)
        writer.flush()

        torch.save(encoder_net.state_dict(),os.path.join(log_path,f"encoder_net{i:05}.pt"))
        torch.save(encoder_net.state_dict(),os.path.join(log_path,f"dist_net{i:05}.pt"))
        torch.save(encoder_net.state_dict(),os.path.join(log_path,f"noise_net{i:05}.pt"))




if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr',  type=float, default=1e-4)
    parser.add_argument('--epoch',  type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_ratio',  type=float, default=0.2)
    parser.add_argument('--alpha',  type=float, default=1e-4)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dir',  type=str, default='./dataset')
    parser.add_argument('--log_path',  type=str, default='./log/01')
    args = parser.parse_args()
    print(args)
    train(args.lr, args.epoch,  args.batch_size, args.val_ratio, args.alpha, args.k, args.seed, args.dir, args.log_path)