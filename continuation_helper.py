from torch.utils.tensorboard import SummaryWriter

from utils.parameters import Params


class ContinuationHelper:

    tb_writer: SummaryWriter = None

    def __init__(self, params):
        self.params = Params(**params)
        if self.params.tb:
            self.tb_writer = SummaryWriter(log_dir=f'continuation_helper_runs/{self.params.name}')
