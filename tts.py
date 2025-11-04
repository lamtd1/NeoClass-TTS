from transformers import VitsModel, AutoTokenizer
import torch

model = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

text = "1 2 3 4 5 6 7 8 9 10"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform
import scipy
import numpy as np


output_numpy = output.squeeze().cpu().numpy()

scipy.io.wavfile.write(f"number.wav", rate=model.config.sampling_rate, data=output_numpy)

print("Đã lưu file techno.wav thành công!")