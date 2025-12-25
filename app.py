import os
import numpy as np
import torch
import gradio as gr
from typing import Optional, Tuple
from funasr import AutoModel
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Disable MPS to avoid "Output channels > 65536 not supported" error on macOS
torch.backends.mps.enabled = False

if os.environ.get("HF_REPO_ID", "").strip() == "":
    os.environ["HF_REPO_ID"] = "openbmb/VoxCPM1.5"

import voxcpm


class VoxCPMDemo:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Running on device: {self.device}")

        # ASR model for prompt text recognition
        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_model: Optional[AutoModel] = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level='DEBUG',
            device="cuda:0" if self.device == "cuda" else "cpu",
        )

        # TTS model (lazy init)
        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self.default_local_model_dir = "./models/VoxCPM1.5"

    # ---------- Model helpers ----------
    def _resolve_model_dir(self) -> str:
        """
        Resolve model directory:
        1) Use local checkpoint directory if exists
        2) If HF_REPO_ID env is set, download into models/{repo}
        3) Fallback to 'models'
        """
        if os.path.isdir(self.default_local_model_dir):
            return self.default_local_model_dir

        repo_id = os.environ.get("HF_REPO_ID", "").strip()
        if len(repo_id) > 0:
            target_dir = os.path.join("models", repo_id.replace("/", "__"))
            if not os.path.isdir(target_dir):
                try:
                    from huggingface_hub import snapshot_download  # type: ignore
                    os.makedirs(target_dir, exist_ok=True)
                    print(f"Downloading model from HF repo '{repo_id}' to '{target_dir}' ...")
                    snapshot_download(repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False)
                except Exception as e:
                    print(f"Warning: HF download failed: {e}. Falling back to 'data'.")
                    return "models"
            return target_dir
        return "models"

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        print("Model not loaded, initializing...")
        model_dir = self._resolve_model_dir()
        print(f"Using model dir: {model_dir}")
        self.voxcpm_model = voxcpm.VoxCPM(voxcpm_model_path=model_dir)
        print("Model loaded successfully.")
        return self.voxcpm_model

    # ---------- Functional endpoints ----------
    def prompt_wav_recognition(self, prompt_wav):
        if prompt_wav is None:
            return ""
        res = self.asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
        text = res[0]["text"].split('|>')[-1]
        return text

    def generate_tts_audio(
        self,
        text_input,
        prompt_wav_path_input = None,
        prompt_text_input = None,
        cfg_value_input = 2.0,
        inference_timesteps_input = 10,
        do_normalize = True,
        denoise = True,
    ):
        """
        Generate speech from text using VoxCPM; optional reference audio for voice style guidance.
        Returns (sample_rate, waveform_numpy)
        """
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        prompt_wav_path = prompt_wav_path_input if prompt_wav_path_input else None
        prompt_text = prompt_text_input if prompt_text_input else None

        print(f"DEBUG: generate_tts_audio called with text='{text[:20]}...', prompt='{prompt_wav_path}'")
        print(f"Generating audio for text: '{text[:60]}...'")
        wav = current_model.generate(
            text=text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=int(inference_timesteps_input),
            normalize=do_normalize,
            denoise=denoise,
        )
        return (current_model.tts_model.sample_rate, wav)


# ---------- UI Builders ----------

def create_demo_interface(demo: VoxCPMDemo):
    """Build the Gradio UI for VoxCPM demo with language switching."""
    # static assets (logo path)
    # assets_dir = Path(__file__).parent / "assets"
    # gr.set_static_paths(paths=[assets_dir])

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
        ),
        css="""
        .logo-container {
            text-align: center;
            margin: 0.5rem 0 1rem 0;
        }
        .logo-container img {
            height: 80px;
            width: auto;
            max-width: 200px;
            display: inline-block;
        }
        /* Bold accordion labels */
        #acc_quick_en details > summary,
        #acc_quick_zh details > summary,
        #acc_tips_en details > summary,
        #acc_tips_zh details > summary {
            font-weight: 600 !important;
            font-size: 1.1em !important;
        }
        /* Bold labels for specific checkboxes */
        #chk_denoise_en label,
        #chk_denoise_en span,
        #chk_normalize_en label,
        #chk_normalize_en span,
        #chk_denoise_zh label,
        #chk_denoise_zh span,
        #chk_normalize_zh label,
        #chk_normalize_zh span {
            font-weight: 600;
        }
        """,
        analytics_enabled=False
    ) as interface:
        # State to track current language
        lang_state = gr.State("en")

        # Header logo
        logo_path = Path(__file__).parent / "assets" / "voxcpm_logo.png"
        gr.HTML(f'<div class="logo-container"><img src="file={logo_path}" alt="VoxCPM Logo"></div>')

        # Language selector
        with gr.Row():
            with gr.Column(scale=10):
                pass
            with gr.Column(scale=1, min_width=150):
                lang_selector = gr.Radio(
                    choices=[("English", "en"), ("中文", "zh")],
                    value="en",
                    label="Language",
                    interactive=True
                )

        # ==================== ENGLISH VERSION ====================
        with gr.Group(elem_id="group_en") as group_en:
            # Quick Start
            with gr.Accordion("📋 Quick Start Guide", open=False, elem_id="acc_quick_en"):
                gr.Markdown("""
### How to Use
1. **(Optional) Provide a Voice Prompt** - Upload or record an audio clip to provide the desired voice characteristics for synthesis.
2. **(Optional) Enter prompt text** - If you provided a voice prompt, enter the corresponding transcript here (auto-recognition available).
3. **Enter target text** - Type the text you want the model to speak.
4. **Generate Speech** - Click the "Generate" button to create your audio.
                """)

            # Pro Tips
            with gr.Accordion("💡 Pro Tips", open=False, elem_id="acc_tips_en"):
                gr.Markdown("""
### Prompt Speech Enhancement
- **Enable** to remove background noise for a clean voice, with an external ZipEnhancer component. However, this will limit the audio sampling rate to 16kHz, restricting the cloning quality ceiling.
- **Disable** to preserve the original audio's all information, including background atmosphere, and support audio cloning up to 44.1kHz sampling rate.

### Text Normalization
- **Enable** to process general text with an external WeTextProcessing component.
- **Disable** to use VoxCPM's native text understanding ability.

### CFG Value
- **Lower CFG** if the voice prompt sounds strained or expressive, or instability occurs with long text input.
- **Higher CFG** for better adherence to the prompt speech style or input text, or instability occurs with too short text input.

### Inference Timesteps
- **Lower** for faster synthesis speed.
- **Higher** for better synthesis quality.
                """)

            # Main controls
            with gr.Row():
                with gr.Column():
                    prompt_wav_en = gr.Audio(
                        sources=["upload", 'microphone'],
                        type="filepath",
                        label="Prompt Speech (Optional, or let VoxCPM improvise)",
                        value="./examples/example.wav",
                    )
                    DoDenoisePromptAudio_en = gr.Checkbox(
                        value=False,
                        label="Prompt Speech Enhancement",
                        elem_id="chk_denoise_en",
                        info="We use ZipEnhancer model to denoise the prompt audio."
                    )
                    with gr.Row():
                        prompt_text_en = gr.Textbox(
                            value="Just by listening a few minutes a day, you'll be able to eliminate negative thoughts by conditioning your mind to be more positive.",
                            label="Prompt Text",
                            placeholder="Please enter the prompt text. Automatic recognition is supported, and you can correct the results yourself..."
                        )
                    run_btn_en = gr.Button("Generate Speech", variant="primary")

                with gr.Column():
                    cfg_value_en = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        value=2.0,
                        step=0.1,
                        label="CFG Value (Guidance Scale)",
                        info="Higher values increase adherence to prompt, lower values allow more creativity"
                    )
                    inference_timesteps_en = gr.Slider(
                        minimum=4,
                        maximum=30,
                        value=10,
                        step=1,
                        label="Inference Timesteps",
                        info="Number of inference timesteps for generation (higher values may improve quality but slower)"
                    )
                    with gr.Row():
                        text_en = gr.Textbox(
                            value="VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly realistic speech.",
                            label="Target Text",
                        )
                    with gr.Row():
                        DoNormalizeText_en = gr.Checkbox(
                            value=False,
                            label="Text Normalization",
                            elem_id="chk_normalize_en",
                            info="We use wetext library to normalize the input text."
                        )
                    audio_output_en = gr.Audio(label="Output Audio")

            # Wiring for English
            run_btn_en.click(
                fn=demo.generate_tts_audio,
                inputs=[text_en, prompt_wav_en, prompt_text_en, cfg_value_en, inference_timesteps_en, DoNormalizeText_en, DoDenoisePromptAudio_en],
                outputs=[audio_output_en],
                show_progress=True,
                api_name="generate_en"
            )
            prompt_wav_en.change(fn=demo.prompt_wav_recognition, inputs=[prompt_wav_en], outputs=[prompt_text_en], api_name="recognize_en")

        # ==================== CHINESE VERSION ====================
        with gr.Group(elem_id="group_zh", visible=False) as group_zh:
            # Quick Start
            with gr.Accordion("📋 快速入门", open=False, elem_id="acc_quick_zh"):
                gr.Markdown("""
### 使用说明
1. **（可选）提供参考声音** - 上传或录制一段音频，为声音合成提供音色、语调和情感等个性化特征
2. **（可选项）输入参考文本** - 如果提供了参考语音，请输入其对应的文本内容（支持自动识别）。
3. **输入目标文本** - 输入您希望模型朗读的文字内容。
4. **生成语音** - 点击"生成"按钮，即可为您创造出音频。
                """)

            # Pro Tips
            with gr.Accordion("💡 使用建议", open=False, elem_id="acc_tips_zh"):
                gr.Markdown("""
### 参考语音降噪
- **启用**：通过 ZipEnhancer 组件消除背景噪音，但会将音频采样率限制在16kHz，限制克隆上限。
- **禁用**：保留原始音频的全部信息，包括背景环境声，最高支持44.1kHz的音频复刻。

### 文本正则化
- **启用**：使用 WeTextProcessing 组件，可支持常见文本的正则化处理。
- **禁用**：将使用 VoxCPM 内置的文本理解能力。如，支持音素输入（如中文转拼音：{ni3}{hao3}；英文转CMUDict：{HH AH0 L OW1}）和公式符号合成，尝试一下！

### CFG 值
- **调低**：如果提示语音听起来不自然或过于夸张，或者长文本输入出现稳定性问题。
- **调高**：为更好地贴合提示音频的风格或输入文本， 或者极短文本输入出现稳定性问题。

### 推理时间步
- **调低**：合成速度更快。
- **调高**：合成质量更佳。
                """)

            # Main controls
            with gr.Row():
                with gr.Column():
                    prompt_wav_zh = gr.Audio(
                        sources=["upload", 'microphone'],
                        type="filepath",
                        label="参考语音（可选，或让 VoxCPM 自由发挥）",
                        value="./examples/example.wav",
                    )
                    DoDenoisePromptAudio_zh = gr.Checkbox(
                        value=False,
                        label="参考语音降噪",
                        elem_id="chk_denoise_zh",
                        info="我们使用 ZipEnhancer 模型对参考语音进行降噪。"
                    )
                    with gr.Row():
                        prompt_text_zh = gr.Textbox(
                            value="只需每天花几分钟时间收听，您就能通过调节思维方式来消除负面想法，让心态更积极。",
                            label="参考文本",
                            placeholder="请输入参考文本。支持自动识别，您也可以自己修正结果..."
                        )
                    run_btn_zh = gr.Button("生成语音", variant="primary")

                with gr.Column():
                    cfg_value_zh = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        value=2.0,
                        step=0.1,
                        label="CFG 值（引导强度）",
                        info="更高的值会增强对参考音频的遵循，更低的值允许更多创意"
                    )
                    inference_timesteps_zh = gr.Slider(
                        minimum=4,
                        maximum=30,
                        value=10,
                        step=1,
                        label="推理时间步",
                        info="生成的推理时间步数（更高的值可能提高质量但速度更慢）"
                    )
                    with gr.Row():
                        text_zh = gr.Textbox(
                            value="VoxCPM 是一个创新的端到端文本转语音模型，来自 ModelBest，旨在生成高度逼真的语音。",
                            label="目标文本",
                        )
                    with gr.Row():
                        DoNormalizeText_zh = gr.Checkbox(
                            value=False,
                            label="文本正则化",
                            elem_id="chk_normalize_zh",
                            info="我们使用 wetext 库来规范化输入文本。"
                        )
                    audio_output_zh = gr.Audio(label="输出音频")

            # Wiring for Chinese
            run_btn_zh.click(
                fn=demo.generate_tts_audio,
                inputs=[text_zh, prompt_wav_zh, prompt_text_zh, cfg_value_zh, inference_timesteps_zh, DoNormalizeText_zh, DoDenoisePromptAudio_zh],
                outputs=[audio_output_zh],
                show_progress=True,
                api_name="generate_zh"
            )
            prompt_wav_zh.change(fn=demo.prompt_wav_recognition, inputs=[prompt_wav_zh], outputs=[prompt_text_zh], api_name="recognize_zh")

        # Language switching logic
        def switch_language(lang: str):
            return {
                group_en: gr.update(visible=(lang == "en")),
                group_zh: gr.update(visible=(lang == "zh")),
            }

        lang_selector.change(
            fn=switch_language,
            inputs=[lang_selector],
            outputs=[group_en, group_zh]
        )

    return interface


def run_demo(server_name: str = "127.0.0.1", server_port: int = 7860, show_error: bool = True):
    demo = VoxCPMDemo()
    interface = create_demo_interface(demo)
    # Launch with queue enabled
    assets_dir = Path(__file__).parent / "assets"
    interface.queue().launch(
        server_name=server_name,
        server_port=server_port,
        show_error=show_error,
        debug=True,
        share=False,
        allowed_paths=[str(assets_dir)]
    )


if __name__ == "__main__":
    run_demo()
