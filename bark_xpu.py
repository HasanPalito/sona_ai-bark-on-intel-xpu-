from transformers import AutoProcessor, AutoModel
#make sure you install intel_extension_for_pytorch
import intel_extension_for_pytorch as torch
import torch as tp
import scipy
import time

# Check device availability
device = "xpu" if torch.xpu.is_available() else "cpu"
print(f"Device selected: {device}")

# Load model and processor
model = AutoModel.from_pretrained("suno/bark-small")
#using half precission ,horrible result
#model = AutoModel.from_pretrained("suno/bark-small", torch_dtype=tp.float16)

processor = AutoProcessor.from_pretrained("suno/bark-small")

# Get sampling rate from model config
# sampling_rate = 24000
sampling_rate = model.generation_config.sample_rate

# using bettertransformer(optional)
model = model.to_bettertransformer()
# enable CPU offload
#model.enable_cpu_offload() not working :(

#load to xpu
model = model.to(device)

# Prepare inputs
text_prompt = "Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."
inputs = processor(text_prompt).to(device)

start=time.time()
# Generate speech values on the same device
speech_values = model.generate(**inputs, do_sample=True)
end=time.time()

#if u are using half precission
#Convert speech_values to float32 because scipy doesnt support fp16
#speech_values_float32 = speech_values.to(tp.float32)
#speech_values_numpy = speech_values_float32.cpu().numpy().squeeze()

#if u are using full precission
speech_values_numpy = speech_values.cpu().numpy().squeeze()

# Write to WAV file
scipy.io.wavfile.write("bark_out_better_transformer_fp16.wav", rate=sampling_rate, data=speech_values_numpy)

# Print device to confirm
print(f"Model loaded on {device}. generation time:{end-start}")