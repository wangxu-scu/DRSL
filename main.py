import torch
# from model import TextDNN, ImageDNN, DNN, CNN
from model import Model
import numpy as np
# np_seed = 1111
# torch_seed = 2345

# torch.manual_seed(torch_seed)
# np.random.seed(np_seed)



from custom_dataset import MyCustomDataset
from torch.utils.data import DataLoader



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
dataset_config = {
    'dataset_name': 'xmedianet_deep',
    'class_number': 200}

batch_size = {'train':200, 'test':200}
dataset = {x: MyCustomDataset(dataset=dataset_config['dataset_name'], state=x)
           for x in ['train', 'test']}
is_shuffle={'train': True, 'test': False}
dataloaders = {x: DataLoader(dataset[x], batch_size=batch_size[x],
                             shuffle=is_shuffle[x], num_workers=1)
               for x in ['train', 'test']}

dataset_sizes = {x: len(dataset[x]) for x in ['train', 'test']}


model = Model(
    input_dim_I=4096,
    input_dim_T=300,
    hidden_dim_I=1024,
    hidden_dim_T=1024,
    hidden_dim_R=1024,
    output_dim_I=300,
    output_dim_T=300,
    output_dim_R=1
)
model.to(device)

import train
model = train.train2(model, dataloaders, device, dataset_sizes, num_epochs=20, retreival=True)