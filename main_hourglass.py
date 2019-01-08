import torch
import torch.utils.data
from utils.data import SurrealDataset
from utils import Config
from models import StackedHourGlass
from utils import StackedHourglassTrainer

config = Config('./config')

device_type = "cuda" if torch.cuda.is_available() and config.device_type == "cuda" else "cpu"

print("Using", device_type)

config_video_constraints = config.video_constraints
config = config.hourglass
device = torch.device(device_type)

test_set, val_set, train_set = [], [], []

if config.data_type == "surreal":
    test_set = SurrealDataset(config.data_path, 'test', config.run)
    val_set = SurrealDataset(config.data_path, 'val', config.run)
    train_set = SurrealDataset(config.data_path, 'train', config.run)

test_dataset = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
val_dataset = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
# train_dataset = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)

model = StackedHourGlass(config.n_channels, config.n_stack, config.n_modules, config.n_reductions, config.n_joints)
model = model.to(device)


