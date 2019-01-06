import torch.utils.data
from utils.data import Human36M
from utils import Config
from utils.models import Linear
from utils.trainer import Trainer

config = Config()
config.set("batch_size", 64, "Batch Size")
config.set("n_epochs", 200, "Number of epochs")
config.set("plot_logs", False, "Number of epochs")
config.set("max_video_length", -1, "Length max of video")
config.set("video_constraints", True, "If we use videos")
config.set("video_constraints_regularization", 0.01, "Regularization of continuity constraints on video")
config.set("frames_before", 1, "Number of frames before")
config.set("frames_after", 1, "Number of frames after")

device_type = "cuda" if torch.cuda.is_available() else "cpu"
print('Using', device_type)
device = torch.device(device_type)

dataset = Human36M('../dataset/h36m/', max_video_length=config.max_video_length,
                   video_constraints=config.video_constraints,
                   frames_before=config.frames_before, frames_after=config.frames_after)

train_set = torch.utils.data.DataLoader(dataset.train_set, batch_size=config.batch_size, shuffle=True)
test_set = torch.utils.data.DataLoader(dataset.test_set, batch_size=config.batch_size)

number_frames = config.frames_before + config.frames_after + 1
model = Linear(input_size=32 * number_frames, hidden_size=1024, output_size=48).to(device)

optimizer = torch.optim.Adam(model.parameters())

trainer = Trainer(train_set, test_set, optimizer, model, dataset,
                  save_folder='builds', plot_logs=config.plot_logs,
                  video_constraints=config.video_constraints,
                  frames_before=config.frames_before,
                  frames_after=config.frames_after,
                  regularization_video_constraints=config.video_constraints_regularization).to(device)

trainer.train(config.n_epochs)

# trainer.load('./builds/2019-01-03 15:28:25')
# trainer.val()
