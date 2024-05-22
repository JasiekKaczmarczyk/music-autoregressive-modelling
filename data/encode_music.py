import glob
import os
import torch
import torchaudio
from tqdm import tqdm

from transformers import EncodecModel, AutoProcessor

def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == "__main__":
    makedir_if_not_exists("quantized")

    fp = glob.glob("music/**.mp3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EncodecModel.from_pretrained("facebook/encodec_32khz").to(device)
    processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")

    for f in tqdm(fp, total=len(fp)):
        name = f.split("/")[-1].split(".")[0]

        x, sr = torchaudio.load(f)
        x = x.mean(dim=0)
        inputs = processor(raw_audio=x, sampling_rate=processor.sampling_rate, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.no_grad():
            z = model.encode(**inputs)
        
        codes = z.audio_codes.squeeze(0).squeeze(0).cpu()

        print(f"Audio: {x.shape} -> Latent: {codes.shape}")

        torch.save(
            codes,
            f"quantized/{name}.pt",
        )