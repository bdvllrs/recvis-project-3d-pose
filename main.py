import torch.utils.data
from utils.data import Human36M
from utils import Config
from utils.models import Linear
from utils.trainer import Trainer

config = Config()
config.set("batch_size", 64, "Batch Size")
config.set("n_epochs", 200, "Number of epochs")
config.set("plot_logs", False, "Number of epochs")

device_type = "cuda" if torch.cuda.is_available() else "cpu"
print('Using', device_type)
device = torch.device(device_type)

dataset = Human36M('../dataset/h36m/')

train_set = torch.utils.data.DataLoader(dataset.train_set, batch_size=config.batch_size, shuffle=True)
test_set = torch.utils.data.DataLoader(dataset.test_set, batch_size=config.batch_size)

model = Linear(input_size=32, hidden_size=1024, output_size=48).to(device)

optimizer = torch.optim.Adam(model.parameters())


trainer = Trainer(train_set, test_set, optimizer, model, dataset,
                  save_folder='builds', plot_logs=config.plot_logs).to(device)

trainer.train(config.n_epochs)
