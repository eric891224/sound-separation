import torch

from torch import Tensor
from s3prl.nn import S3PRLUpstream

from typing import Tuple

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

HOP_SIZE = 20 # ms

def load_model(model_file_path: str = "") -> torch.nn.Module:
	"""
	Returns a torch.nn.Module that produces embeddings for audio.

	Args:
		model_file_path: Ignored.
	Returns:
		Model
	"""

	model = S3PRLUpstream("mae_ast_frame", refresh=False)
	model = model.to(device)

	model.sample_rate = 16000
	model.scene_embedding_size = 768
	model.timestamp_embedding_size = 768

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