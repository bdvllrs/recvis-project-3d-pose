import torch
import torch.utils.data
from utils.data import SurrealDataset
from utils import Config
from models import VideoContinuitySHourglass
from utils import StackedHourglassTrainer as Trainer
import matplotlib.pyplot as plt

config = Config('./config')

device_type = "cuda" if torch.cuda.is_available() and config.device_type == "cuda" else "cpu"

print("Using", device_type)

config_video_constraints = config.video_constraints
config = config.hourglass
device = torch.device(device_type)

test_set, val_set, train_set = [], [], []

if config.data_type == "surreal":
    test_set = SurrealDataset(config.data_path, 'test', config.run,
                              frames_before=config_video_constraints.frames_before,
                              frames_after=config_video_constraints.frames_after)

    # frame, joints, _ = test_set[0]
    # frame = frame[0].transpose((1, 2, 0))
    # plt.imshow(frame)
    # for k in range(joints.shape[1]):
    #     circle = plt.Circle((joints[0, k], joints[1, k]), radius=1, color='red')
    #     plt.gcf().gca().add_artist(circle)
    # plt.show()
    # print(frame.shape)
    val_set = SurrealDataset(config.data_path, 'val', config.run,
                             frames_before=config_video_constraints.frames_before,
                             frames_after=config_video_constraints.frames_after)
    train_set = SurrealDataset(config.data_path, 'train', config.run,
                               frames_before=config_video_constraints.frames_before,
                               frames_after=config_video_constraints.frames_after)

test_dataset = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
val_dataset = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
train_dataset = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)

n_frames = 1 + config_video_constraints.frames_before + config_video_constraints.frames_after

model = VideoContinuitySHourglass(config.n_channels, config.n_stack, config.n_modules, config.n_reductions,
                                  config.n_joints, n_frames)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters())

for data, joints_2d, joints_3d in train_dataset:
    pass
# trainer = Trainer(train_dataset, test_dataset, optimizer, model,
#                   save_folder='builds/hourglass', plot_logs=config.plot_logs,
#                   video_constraints=config_video_constraints.use,
#                   frames_before=config_video_constraints.frames_before,
#                   frames_after=config_video_constraints.frames_after,
#                   regularization_video_constraints=config_video_constraints.regularization).to(device)
#
# trainer.train(config.n_epochs)
