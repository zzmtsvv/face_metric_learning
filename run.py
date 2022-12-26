from utils import seed_everything
from train_loop import Trainer

seed_everything()
t = Trainer()
t.fit()
