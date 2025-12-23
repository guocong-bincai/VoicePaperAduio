import os
import re
import tempfile
import numpy as np
from typing import Generator, Optional
from huggingface_hub import snapshot_download
from .model.voxcpm import VoxCPMModel, LoRAConfig

class VoxCPM:
    def __init__(self,
            voxcpm_model_path : str,
            zipenhancer_model_path : str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
            enable_denoiser : bool = True,
            optimize: bool = True,
            lora_config: Optional[LoRAConfig] = None,
            lora_weights_path: Optional[str] = None,
        ):
        """Initialize VoxCPM TTS pipeline.

        Args:
            voxcpm_model_path: Local filesystem path to the VoxCPM model assets
                (weights, configs, etc.). Typically the directory returned by
                a prior download step.
            zipenhancer_model_path: ModelScope acoustic noise suppression model
                id or local path. If None, denoiser will not be initialized.
            enable_denoiser: Whether to initialize the denoiser pipeline.
            optimize: Whether to optimize the model with torch.compile. True by default, but can be disabled for debugging.
            lora_config: LoRA configuration for fine-tuning. If lora_weights_path is
                provided without lora_config, a default config will be created.
            lora_weights_path: Path to pre-trained LoRA weights (.pth file or directory
                containing lora_weights.ckpt). If provided, LoRA weights will be loaded.
        """
        print(f"voxcpm_model_path: {voxcpm_model_path}, zipenhancer_model_path: {zipenhancer_model_path}, enable_denoiser: {enable_denoiser}")

        # If lora_weights_path is provided but no lora_config, create a default one
        if lora_weights_path is not None and lora_config is None:
            lora_config = LoRAConfig(
                enable_lm=True,
                enable_dit=True,
                enable_proj=False,
            )
            print(f"Auto-created default LoRAConfig for loading weights from: {lora_weights_path}")

        self.tts_model = VoxCPMModel.from_local(voxcpm_model_path, optimize=optimize, lora_config=lora_config)

        # Load LoRA weights if path is provided
        if lora_weights_path is not None:
            print(f"Loading LoRA weights from: {lora_weights_path}")
            loaded_keys, skipped_keys = self.tts_model.load_lora_weights(lora_weights_path)
            print(f"Loaded {len(loaded_keys)} LoRA parameters, skipped {len(skipped_keys)}")

        self.text_normalizer = None
        if enable_denoiser and zipenhancer_model_path is not None:
            from .zipenhancer import ZipEnhancer
            self.denoiser = ZipEnhancer(zipenhancer_model_path)
        else:
            self.denoiser = None
        if optimize:
            print("Warm up VoxCPMModel...")
            self.tts_model.generate(
                target_text="Hello, this is the first test sentence.",
                max_len=10,
            )

    @classmethod
    def from_pretrained(cls,
            hf_model_id: str = "openbmb/VoxCPM1.5",
            load_denoiser: bool = True,
            zipenhancer_model_id: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
            cache_dir: str = None,
            local_files_only: bool = False,
            optimize: bool = True,
            lora_config: Optional[LoRAConfig] = None,
            lora_weights_path: Optional[str] = None,
            **kwargs,
        ):
        """Instantiate ``VoxCPM`` from a Hugging Face Hub snapshot.

        Args:
            hf_model_id: Explicit Hugging Face repository id (e.g. "org/repo") or local path.
            load_denoiser: Whether to initialize the denoiser pipeline.
            optimize: Whether to optimize the model with torch.compile. True by default, but can be disabled for debugging.
            zipenhancer_model_id: Denoiser model id or path for ModelScope
                acoustic noise suppression.
            cache_dir: Custom cache directory for the snapshot.
            local_files_only: If True, only use local files and do not attempt
                to download.
            lora_config: LoRA configuration for fine-tuning. If lora_weights_path is
                provided without lora_config, a default config will be created with
                enable_lm=True and enable_dit=True.
            lora_weights_path: Path to pre-trained LoRA weights (.pth file or directory
                containing lora_weights.ckpt). If provided, LoRA weights will be loaded
                after model initialization.
        Kwargs:
            Additional keyword arguments passed to the ``VoxCPM`` constructor.

        Returns:
            VoxCPM: Initialized instance whose ``voxcpm_model_path`` points to
            the downloaded snapshot directory.

        Raises:
            ValueError: If neither a valid ``hf_model_id`` nor a resolvable
                ``hf_model_id`` is provided.
        """
        repo_id = hf_model_id
        if not repo_id:
            raise ValueError("You must provide hf_model_id")

        # Load from local path if provided
        if os.path.isdir(repo_id):
            local_path = repo_id
        else:
            # Otherwise, try from_pretrained (Hub); exit on failure
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        return cls(
            voxcpm_model_path=local_path,
            zipenhancer_model_path=zipenhancer_model_id if load_denoiser else None,
            enable_denoiser=load_denoiser,
            optimize=optimize,
            lora_config=lora_config,
            lora_weights_path=lora_weights_path,
            **kwargs,
        )

    def generate(self, *args, **kwargs) -> np.ndarray:
        return next(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[np.ndarray, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    def generate_long(self, text: str, chunk_size: int = 3000, **kwargs) -> np.ndarray:
        """Generate speech for long text (up to 20000 characters).

        Automatically splits long text into manageable chunks and concatenates results.
        Recommended for texts exceeding 4000 characters.

        Args:
            text: Input text (up to 20000 characters)
            chunk_size: Maximum characters per chunk (default 3000)
            **kwargs: Additional arguments passed to generate()

        Returns:
            Combined numpy array of generated speech
        """
        return next(self._generate_long_text(text, chunk_size=chunk_size, streaming=False, **kwargs))

    def generate_long_streaming(self, text: str, chunk_size: int = 3000, **kwargs) -> Generator[np.ndarray, None, None]:
        """Generate long text with streaming output."""
        return self._generate_long_text(text, chunk_size=chunk_size, streaming=True, **kwargs)

    def _split_text_smart(self, text: str, max_chars: int = 3000) -> list:
        """
        Intelligently split text by sentences while preserving semantic integrity.

        Splits by Chinese punctuation (。！？；) and English punctuation (. ! ? ;)

        Args:
            text: Input text
            max_chars: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        # Normalize line breaks
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r'\s+', ' ', text)

        # Split by Chinese and English punctuation
        # 使用更聪明的分割策略：先按句号分割，再合并成指定大小
        sentences = re.split(r'(。|！|？|；|\.|\!|\?|;)', text)

        chunks = []
        current_chunk = ""

        i = 0
        while i < len(sentences):
            sentence = sentences[i].strip()

            # 如果下一个是标点，合并到当前句子
            if i + 1 < len(sentences) and sentences[i + 1] in '。！？；.!?;':
                sentence += sentences[i + 1]
                i += 2
            else:
                i += 1

            # 跳过空句子
            if not sentence:
                continue

            # 如果当前块加上新句子不超过限制，直接加入
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += sentence
            else:
                # 否则保存当前块，开始新块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        # 添加最后的块
        if current_chunk:
            chunks.append(current_chunk.strip())

        return [c for c in chunks if c]

    def _generate_long_text(self,
            text: str,
            prompt_wav_path: str = None,
            prompt_text: str = None,
            cfg_value: float = 2.0,
            inference_timesteps: int = 10,
            chunk_size: int = 3000,
            normalize: bool = False,
            denoise: bool = False,
            streaming: bool = False,
        ) -> Generator[np.ndarray, None, None]:
        """
        Generate speech for long text (up to 20000 characters).

        Automatically splits text into chunks and processes sequentially.

        Args:
            text: Input text (max 20000 characters)
            prompt_wav_path: Optional reference audio for voice style
            prompt_text: Text corresponding to reference audio
            cfg_value: Guidance scale
            inference_timesteps: Number of inference steps per chunk
            chunk_size: Target characters per chunk (default 3000)
            normalize: Whether to normalize text
            denoise: Whether to denoise prompt audio
            streaming: Whether to yield chunks as they're generated

        Yields:
            numpy.ndarray: Audio chunks if streaming=True, else final combined audio
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        if len(text) > 20000:
            raise ValueError(f"Text too long: {len(text)} characters. Maximum is 20000.")

        # Split text into manageable chunks
        chunks = self._split_text_smart(text, max_chars=chunk_size)

        print(f"[VoxCPM-LongText] Processing {len(text)} characters in {len(chunks)} chunks (chunk_size={chunk_size})")

        if len(chunks) == 1:
            # Short text, use standard generation
            for wav in self._generate(
                text=chunks[0],
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                denoise=denoise,
                streaming=streaming,
            ):
                yield wav
        else:
            # Long text, process chunks sequentially
            all_wavs = []

            for i, chunk in enumerate(chunks, 1):
                print(f"[VoxCPM-LongText] Chunk {i}/{len(chunks)}: {len(chunk)} characters")

                # Only use external prompt for first chunk
                # Subsequent chunks use internal consistency
                if i == 1:
                    curr_prompt_wav = prompt_wav_path
                    curr_prompt_text = prompt_text
                else:
                    curr_prompt_wav = None
                    curr_prompt_text = None

                # Generate this chunk
                wav = self.generate(
                    text=chunk,
                    prompt_wav_path=curr_prompt_wav,
                    prompt_text=curr_prompt_text,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    normalize=normalize,
                    denoise=denoise,
                )

                all_wavs.append(wav)

                if streaming:
                    # Yield chunk immediately in streaming mode
                    yield wav

            if not streaming:
                # Concatenate all chunks and yield once
                combined = np.concatenate(all_wavs)
                print(f"[VoxCPM-LongText] Combined {len(chunks)} chunks into final audio")
                yield combined

    def _generate(self,
            text : str,
            prompt_wav_path : str = None,
            prompt_text : str = None,
            cfg_value : float = 2.0,
            inference_timesteps : int = 10,
            min_len : int = 2,
            max_len : int = 4096,
            normalize : bool = False,
            denoise : bool = False,
            retry_badcase : bool = True,
            retry_badcase_max_times : int = 3,
            retry_badcase_ratio_threshold : float = 6.0,
            streaming: bool = False,
        ) -> Generator[np.ndarray, None, None]:
        """Synthesize speech for the given text and return a single waveform.

        This method optionally builds and reuses a prompt cache. If an external
        prompt (``prompt_wav_path`` + ``prompt_text``) is provided, it will be
        used for all sub-sentences. Otherwise, the prompt cache is built from
        the first generated result and reused for the remaining text chunks.

        Args:
            text: Input text. Can include newlines; each non-empty line is
                treated as a sub-sentence.
            prompt_wav_path: Path to a reference audio file for prompting.
            prompt_text: Text content corresponding to the prompt audio.
            cfg_value: Guidance scale for the generation model.
            inference_timesteps: Number of inference steps.
            max_len: Maximum token length during generation.
            normalize: Whether to run text normalization before generation.
            denoise: Whether to denoise the prompt audio if a denoiser is
                available.
            retry_badcase: Whether to retry badcase.
            retry_badcase_max_times: Maximum number of times to retry badcase.
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio.
            streaming: Whether to return a generator of audio chunks.
        Returns:
            Generator of numpy.ndarray: 1D waveform array (float32) on CPU.
            Yields audio chunks for each generations step if ``streaming=True``,
            otherwise yields a single array containing the final audio.
        """
        if not text.strip() or not isinstance(text, str):
            raise ValueError("target text must be a non-empty string")

        if prompt_wav_path is not None:
            if not os.path.exists(prompt_wav_path):
                raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")

        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")

        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)
        temp_prompt_wav_path = None

        try:
            if prompt_wav_path is not None and prompt_text is not None:
                if denoise and self.denoiser is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        temp_prompt_wav_path = tmp_file.name
                    self.denoiser.enhance(prompt_wav_path, output_path=temp_prompt_wav_path)
                    prompt_wav_path = temp_prompt_wav_path
                fixed_prompt_cache = self.tts_model.build_prompt_cache(
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text
                )
            else:
                fixed_prompt_cache = None  # will be built from the first inference

            if normalize:
                if self.text_normalizer is None:
                    from .utils.text_normalize import TextNormalizer
                    self.text_normalizer = TextNormalizer()
                text = self.text_normalizer.normalize(text)

            generate_result = self.tts_model._generate_with_prompt_cache(
                            target_text=text,
                            prompt_cache=fixed_prompt_cache,
                            min_len=min_len,
                            max_len=max_len,
                            inference_timesteps=inference_timesteps,
                            cfg_value=cfg_value,
                            retry_badcase=retry_badcase,
                            retry_badcase_max_times=retry_badcase_max_times,
                            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                            streaming=streaming,
                        )

            for wav, _, _ in generate_result:
                yield wav.squeeze(0).cpu().numpy()

        finally:
            if temp_prompt_wav_path and os.path.exists(temp_prompt_wav_path):
                try:
                    os.unlink(temp_prompt_wav_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------ #
    # LoRA Interface (delegated to VoxCPMModel)
    # ------------------------------------------------------------------ #
    def load_lora(self, lora_weights_path: str) -> tuple:
        """Load LoRA weights from a checkpoint file.

        Args:
            lora_weights_path: Path to LoRA weights (.pth file or directory
                containing lora_weights.ckpt).

        Returns:
            tuple: (loaded_keys, skipped_keys) - lists of loaded and skipped parameter names.

        Raises:
            RuntimeError: If model was not initialized with LoRA config.
        """
        if self.tts_model.lora_config is None:
            raise RuntimeError(
                "Cannot load LoRA weights: model was not initialized with LoRA config. "
                "Please reinitialize with lora_config or lora_weights_path parameter."
            )
        return self.tts_model.load_lora_weights(lora_weights_path)

    def unload_lora(self):
        """Unload LoRA by resetting all LoRA weights to initial state (effectively disabling LoRA)."""
        self.tts_model.reset_lora_weights()

    def set_lora_enabled(self, enabled: bool):
        """Enable or disable LoRA layers without unloading weights.

        Args:
            enabled: If True, LoRA layers are active; if False, only base model is used.
        """
        self.tts_model.set_lora_enabled(enabled)

    def get_lora_state_dict(self) -> dict:
        """Get current LoRA parameters state dict.

        Returns:
            dict: State dict containing all LoRA parameters (lora_A, lora_B).
        """
        return self.tts_model.get_lora_state_dict()

    @property
    def lora_enabled(self) -> bool:
        """Check if LoRA is currently configured."""
        return self.tts_model.lora_config is not None