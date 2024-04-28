from torch.utils.data import Dataset
from torchvision import transforms
from sklearn import model_selection
import glob
import os
import pickle
from PIL import Image
import numpy as np
import torch

class Nomad_Dataset(Dataset):
    def __init__(self,dir, len_traj_pred=8, ct_len = 6, max_distance=20, mode='train',val_ratio=0.2, seed=42):
        self.dir = dir
        self.len_traj_pred=8
        self.ct_len=6
        self.max_distance=20
        self.mode=mode
        self.val_ratio = val_ratio
        self.seed = seed
        self.nomad_transform = transforms.Compose([transforms.Resize(96),
                                                   transforms.CenterCrop((96,96)),
                                                   transforms.ToTensor()])
        self.setup()

    def setup(self):
        traj_dirs = glob.glob(self.dir +'/*')
        traj_dirs.sort()
        train_traj_dirs, val_traj_dirs = model_selection.train_test_split(traj_dirs, test_size=self.val_ratio, random_state=self.seed)

        if self.mode=='train':
            train_traj_dirs.sort()
            self.traj_dirs = train_traj_dirs

        elif self.mode =='val':
            val_traj_dirs.sort()
            self.traj_dirs = val_traj_dirs

        self.img_paths = []
        for traj_dir in self.traj_dirs:
            traj_path = os.path.join(traj_dir,'traj_data.pkl')
            with open(traj_path, mode='rb') as f:
                traj_data = pickle.load(f)
            traj_len = len(traj_data['position'])
            tmp = [x for x in glob.glob(traj_dir+'/*.jpg') if (int(os.path.splitext(os.path.basename(x))[0])>=5 and int(os.path.splitext(os.path.basename(x))[0])<traj_len-8)]
            self.img_paths += tmp

    def abs2relative(self, curr_pos,curr_yaw,positions): # reverse transform
        #y_axis yaw = 0
        curr_yaw=curr_yaw[0]
        rot_matrix = np.array([[np.cos(curr_yaw), -np.sin(curr_yaw)],
                               [np.sin(curr_yaw), np.cos(curr_yaw)]])
        return (positions-curr_pos)@rot_matrix

    def pos2action(self, pos):
        action = np.concatenate([np.zeros((1,pos.shape[-1])), pos], axis=0)
        action = action[1:] - action[:-1]
        return action
    def idx2img_path(self,img_path,i):
        curr_time = os.path.basename(img_path)
        curr_time = os.path.splitext(curr_time)[0]
        curr_time = int(curr_time)
        img_path = img_path.split('/')
        img_path[-1]=str(curr_time+i)+'.jpg'
        img_path = '/'.join(img_path)
        return img_path

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,i):
        obs_paths = []
        for idx in range(-5,1,1):
            obs_paths.append(self.idx2img_path(self.img_paths[i],idx))

        #obs_paths = self.img_paths[i-5:i+1]
        traj_dir = os.path.dirname(self.img_paths[i])
        traj_path = os.path.join(traj_dir,'traj_data.pkl')
        with open(traj_path, mode='rb') as f:
            traj_data = pickle.load(f)


        curr_time = os.path.basename(obs_paths[-1])
        curr_time = os.path.splitext(curr_time)[0]
        curr_time = int(curr_time)

        traj_max_time = len(traj_data['position'])-1
        goal_time = curr_time + np.random.randint(3,self.max_distance)
        goal_time = min(traj_max_time,goal_time)
        goal_path = self.idx2img_path(obs_paths[-1],goal_time-curr_time)
        distance = goal_time - curr_time


        pos = self.abs2relative(traj_data['position'][curr_time],
                                    traj_data['yaw'][curr_time],
                                    traj_data['position'][curr_time+1:curr_time+self.len_traj_pred+1])

        #pos = pos.astype(np.float32)
        goal_mask = torch.randint(0,2,(1,)).squeeze()

        if not goal_mask and distance < 8:
            pos[distance-1::]=pos[distance-1]


        actions = self.pos2action(pos)
        actions = actions.astype(np.float32)
        actions = torch.from_numpy(actions)
        distance = torch.tensor([distance/self.max_distance])

        for j,c in enumerate(obs_paths):
            tmp = Image.open(c)
            tmp = self.nomad_transform(tmp)
            if j==0:
                obs_imgs = tmp.unsqueeze(0)
            else:
                obs_imgs = torch.cat((tmp.unsqueeze(0),obs_imgs),dim=0)


        goal_img = Image.open(goal_path)
        goal_img = self.nomad_transform(goal_img)

        return obs_imgs, goal_img, actions, distance, goal_mask
