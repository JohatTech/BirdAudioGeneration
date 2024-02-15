import torch  
import torchaudio 
import torch.optim as optim
from torch.utils.data import Dataset
from torchaudio.models import WaveRNN
import os
import pandas as pd 

metadata = pd.read_csv("bird_songs_metadata.csv")
print(metadata['filename'])
class CustomDataset(Dataset):
    def __init__(self,audio_files, filenames):
        self.audio_dir =  filenames
        self.audio_files = audio_files
    def __len__(self):
        return len(self.audio_dir)

    def __getitem__(self, index):
        file_path = os.path.join(self.audio_files, self.audio_dir[index])
        waveform, sample_rate = torchaudio.load(file_path, format='wav')
        spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)
        return waveform, spectrogram

dataset = CustomDataset('wavfiles', metadata['filename'])
data_loader = torch.utils.data.DataLoader(dataset, batch_size = 8, shuffle = True )

model = WaveRNN(upsample_scales=[5,5,8], n_classes=512, hop_length=200)
optimizer =  optim.Adam(model.parameters(), lr = 0.001)
criterion = torch.nn.CrossEntropyLoss()

epochs = 3 
print_every = 10 

for epoch in range(epochs):
    for batch_id, (waveforms_batch, spectrograms_batch) in enumerate(data_loader):
        
        optimizer.zero_grad()
        output = model.forward(waveforms_batch, spectrograms_batch)
        loss = criterion(output, spectrograms_batch)
        loss.backward()
        optimizer.step()

        if batch_id % print_every ==0:
            print(f"Epoch :{epoch},  batch_id{batch_id}, loss: {loss.item()}")


torch.save(model.state_dict, 'birdmodel.pth')
