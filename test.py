from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr')

# for custom file, change path
est_sources = model.separate_file(path='speechbrain/sepformer-wsj02mix/test_mixture.wav')

torchaudio.save("source1hat.wav", est_sources[:, :, 0].detach().cpu(), 8000)
torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)