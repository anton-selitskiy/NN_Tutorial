import argparse
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from itertools import islice
import pathlib

parser = argparse.ArgumentParser(description = "Test")
parser.add_argument('--file', type=str, default="./recordings/0_george_22.wav", help='Path to your file')
par = parser.parse_args()

path_to_data = pathlib.Path(par.file)

class AudioMnist(Dataset):
    def __init__(self, path_to_data):
        #self.path_to_data = pathlib.Path(path_to_data)
        #self.paths = list(self.path_to_data.rglob('*.wav'))
        self.paths = path_to_data
        self.featurizer = torchaudio.transforms.MelSpectrogram( \
            sample_rate=16000, n_fft=1024, win_length=1024,hop_length=256, n_mels=80)
        
    def __getitem__(self, index):
        path_to_wav = self.paths[index].as_posix()
        wav, _ = torchaudio.load(path_to_wav)
        mel_spec = self.featurizer(wav).squeeze(dim=0).clamp(1e-5).log()
        label = int(path_to_wav.split('/')[-1].split('_')[0])
        return mel_spec, label
        
    def __len__(self):
        return len(self.paths)

dataset = AudioMnist([path_to_data])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(input_size=80, hidden_size=256, batch_first=True)
        self.clf = nn.Linear(256, 10)
        self.s = nn.Softmax(dim=1)
    def forward(self, input):
        output, _ = self.rnn(input.transpose(-1,-2))
        output = self.clf(output[:,-1])
        output = self.s(output)
        return output

model = Model()
state_dict = torch.load('./weights.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()
wav = dataset[0][0].unsqueeze(dim = 0)
with torch.no_grad():
    predict = int(model(wav).detach().argmax().numpy())
print(f"Real value: {dataset[0][1]}, predicted value: {predict}")
