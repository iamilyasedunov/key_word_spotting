import sys
sys.path.append(sys.path[0] + "/..")

from utils.utils import *

class LogMelspec():
    
    def __init__(self, is_train, config):
        # with augmentations
        if is_train:
            self.melspec = nn.Sequential(
                torchaudio.transforms.MelSpectrogram(sample_rate=config.sample_rate,  n_mels=config.n_mels),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
                torchaudio.transforms.TimeMasking(time_mask_param=35),
            ).to(config.device)

        # no augmentations
        else:
            self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_mels=config.n_mels
            ).to(config.device)
            

    def __call__(self, batch):
        return torch.log(self.melspec(batch).clamp_(min=1e-7, max=1e9))  # already on device