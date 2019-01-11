import torch
import torch.utils.data
from utils.data import SurrealDatasetWithVideoContinuity as SurrealDataset
from utils import Config
from models import StackedHourGlass, Linear
from utils import SurrealTrainer as Trainer

config = Config('./config')

device_type = "cuda" if torch.cuda.is_available() and config.device_type == "cuda" else "cpu"

print("Using", device_type)

config_video_constraints = config.video_constraints
config_surreal = config.surreal

config = config.hourglass
device = torch.device(device_type)

test_set, val_set, train_set = [], [], []

# Load dataset
print("Loading datasets...")
if config.data_type == "surreal":
    test_set = SurrealDataset(config_surreal.data_path, 'test', config_surreal.run,
                              frames_before=config_video_constraints.frames_before,
                              frames_after=config_video_constraints.frames_after)
    val_set = SurrealDataset(config_surreal.data_path, 'val', config_surreal.run,
                             frames_before=config_video_constraints.frames_before,
                             frames_after=config_video_constraints.frames_after)
    train_set = SurrealDataset(config_surreal.data_path, 'train', config_surreal.run,
                               frames_before=config_video_constraints.frames_before,
                               frames_after=config_video_constraints.frames_after)

# Define Torch dataset
test_dataset = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
val_dataset = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
train_dataset = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
print("Loaded.")

print("Loading models...")
# Load pretrained model of stacked hourglass for 2D pose estimation
hg_model = StackedHourGlass(config.n_channels, config.n_stack, config.n_modules, config.n_reductions,
                            config.n_joints)
hg_model.to(device)
hg_model.load_state_dict(torch.load(config.pretrained_path, map_location=device)['model_state'])
hg_model.eval()

number_frames = config_video_constraints.frames_before + config_video_constraints.frames_after + 1
model = Linear(input_size=2 * config.n_joints * number_frames, hidden_size=1024,
               output_size=48).to(device)
print("Loaded.")

optimizer = torch.optim.Adam(model.parameters())

trainer = Trainer(train_dataset, test_dataset, optimizer, model, hg_model,
                  save_folder='builds', plot_logs=config.plot_logs,
                  video_constraints=config_video_constraints.use,
                  frames_before=config_video_constraints.frames_before,
                  frames_after=config_video_constraints.frames_after,
                  regularization_video_constraints=config_video_constraints.regularization).to(device)

trainer.train(config.n_epochs)

# trainer.load('./builds/2019-01-03 15:28:25')
# trainer.val()
