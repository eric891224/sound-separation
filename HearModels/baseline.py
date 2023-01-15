import torch

from torch import Tensor
from s3prl.nn import S3PRLUpstream

from typing import Tuple

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

HOP_SIZE = 20 # ms

class Baseline(torch.nn.Module):
	def __init__(self, speech_model_type: str, sound_model_type: str):
		super(Baseline, self).__init__()

		self.speech_model = S3PRLUpstream(speech_model_type, refresh=False)
		self.sound_model = S3PRLUpstream(sound_model_type, refresh=False)

	def forward(self, x, x_len):
		speech_feat, speech_feat_len = self.speech_model(x, x_len)
		sound_feat, sound_feat_len = self.sound_model(x, x_len)

		all_feat = [torch.cat([speech_feat[-1], sound_feat[-1]], dim=2)]
		all_len = speech_feat_len

		return all_feat, all_len

def load_model(
	model_file_path: str = "", 
	speech_model_type: str = "hubert_large_ll60k",
	sound_model_type: str = "mae_ast_frame"
) -> torch.nn.Module:
	"""
	Returns a torch.nn.Module that produces embeddings for audio.

	Args:
		model_file_path: Ignored.
	Returns:
		Model
	"""

	model = Baseline(speech_model_type, sound_model_type)
	model = model.to(device)

	model.sample_rate = 16000
	if speech_model_type == "hubert_large_ll60k" and sound_model_type == "mae_ast_frame":
		model.scene_embedding_size = 1024 + 768
		model.timestamp_embedding_size = 1024 + 768
	elif model_type == "hubert" and sound_model_type == "mae_ast_frame":
		model.scene_embedding_size = 768 + 768
		model.timestamp_embedding_size = 768 + 768

	return model

def get_timestamp_embeddings(
	audio: Tensor,
	model: torch.nn.Module,
) -> Tuple[Tensor, Tensor]:
	"""
	This function returns embeddings at regular intervals centered at timestamps. Both
	the embeddings and corresponding timestamps (in milliseconds) are returned.

	Args:
		audio: n_sounds x n_samples of mono audio in the range [-1, 1].
		model: Loaded model.
	Returns:
		- Tensor: embeddings, A float32 Tensor with shape (n_sounds, n_timestamps,
			model.timestamp_embedding_size).
		- Tensor: timestamps, Centered timestamps in milliseconds corresponding
			to each embedding in the output. Shape: (n_sounds, n_timestamps).
	"""

	audio = audio.to(device)
	audio_len = torch.LongTensor([audio.shape[1]] * audio.shape[0]).to(device)

	model.eval()
	with torch.no_grad():
		embeddings, embeddings_len = model(audio, audio_len)
		embeddings = embeddings[-1] # last layer output

	audio_ms = int(audio.shape[1] / model.sample_rate * 1000)
	# n_timestamps = (audio_ms - 10) // HOP_SIZE
	n_timestamps = audio_ms / HOP_SIZE
	n_timestamps = int(-(-n_timestamps // 1))	# round up
	# last_center = 10 + (n_timestamps - 1) * HOP_SIZE
	end = 10 + n_timestamps * HOP_SIZE

	timestamps = torch.arange(10, end, HOP_SIZE)
	assert len(timestamps) == n_timestamps
	timestamps = timestamps.expand((embeddings.shape[0], timestamps.shape[0]))
	print(audio.shape, timestamps.shape, embeddings.shape)
	assert timestamps.shape[1] == embeddings.shape[1]
	
	return embeddings, timestamps

def get_scene_embeddings(
	audio: Tensor,
	model: torch.nn.Module,
) -> Tensor:
	"""
	This function returns a single embedding for each audio clip. In this baseline
	implementation we simply summarize the temporal embeddings from
	get_timestamp_embeddings() using torch.mean().
	
	Args:
		audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in
			a batch will be padded/trimmed to the same length.
		model: Loaded model.
	Returns:
		- embeddings, A float32 Tensor with shape
			(n_sounds, model.scene_embedding_size).
	"""

	embeddings, _ = get_timestamp_embeddings(audio, model)
	embeddings = torch.mean(embeddings, dim=1)

	return embeddings