import torch
import torch.utils.data
from utils.data import SurrealDataset
from utils import Config

config = Config('./config')

data_path = "/run/media/bdvllrs/Data/Documents/Supelec/MVA/Image/SURREAL2/data"
batch_size = 4

device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)

test_dataset = torch.utils.data.DataLoader(SurrealDataset(data_path, 'test', 'run0'),
                                           batch_size=batch_size, shuffle=False)
val_dataset = torch.utils.data.DataLoader(SurrealDataset(data_path, 'val', 'run0'),
                                          batch_size=batch_size, shuffle=False)
train_dataset = torch.utils.data.DataLoader(SurrealDataset(data_path, 'train', 'run0'),
                                            batch_size=batch_size, shuffle=True)

