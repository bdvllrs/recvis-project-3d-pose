import torch.utils.data
from utils.data import Human36M
from utils import Config
from models.linear import Linear
from utils.trainer import Trainer


config = Config("./config")

device_type = "cuda" if torch.cuda.is_available() and config.device_type == "cuda" else "cpu"
config_video_constraints = config.video_constraints
config = config.main

print('Using', device_type)
device = torch.device(device_type)

dataset = Human36M('../dataset/h36m/', max_video_length=config.max_video_length,
                   video_constraints=config_video_constraints.use,
                   use_hourglass=True,
                   frames_before=config_video_constraints.frames_before, frames_after=config_video_constraints.frames_after)

train_set = torch.utils.data.DataLoader(dataset.train_set, batch_size=config.batch_size, shuffle=True)
test_set = torch.utils.data.DataLoader(dataset.test_set, batch_size=config.batch_size)

number_frames = config_video_constraints.frames_before + config_video_constraints.frames_after + 1
model = Linear(input_size=32 * number_frames, hidden_size=1024, output_size=48).to(device)

optimizer = torch.optim.Adam(model.parameters())

trainer = Trainer(train_set, test_set, optimizer, model, dataset,
                  save_folder='builds', plot_logs=config.plot_logs,
                  video_constraints=config_video_constraints.use,
                  frames_before=config_video_constraints.frames_before,
                  frames_after=config_video_constraints.frames_after,
                  regularization_video_constraints=config_video_constraints.regularization).to(device)

trainer.train(config.n_epochs)

# trainer.load('./builds/2019-01-03 15:28:25')
# trainer.val()
