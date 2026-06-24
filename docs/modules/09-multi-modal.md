# 多模态处理详细设计

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| **统一抽象** | ASR/TTS/图像/文件解析收敛到同一接口，Agent 侧零感知差异 |
| **即插即用** | 新增编码器 = 实现基类 + 注册工厂，不影响已有代码 |
| **异步优先** | 所有方法均为 async，适配 IO 密集型操作 |
| **流式支持** | ASR/TTS 支持流式处理，降低首字延迟 |
| **格式自适应** | 文件解析自动识别 MIME type，选择对应解析器 |

---

## 2. 协议设计 `multimodal/protocol.py`

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Modality(str, Enum):
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"
    FILE = "file"


@dataclass(frozen=True)
class MultimodalInput:
    """多模态输入统一格式"""
    modality: Modality
    data: bytes | str                          # 原始数据（二进制或 base64）
    mime_type: str | None = None               # MIME type，如 "audio/wav"、"image/png"
    metadata: dict[str, Any] | None = None     # 附加元信息


@dataclass(frozen=True)
class MultimodalOutput:
    """多模态处理结果"""
    modality: Modality
    data: bytes | str                          # 处理后数据
    mime_type: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ASRResult:
    """语音识别结果"""
    text: str                                  # 识别文本
    language: str | None = None                # 检测到的语言
    confidence: float | None = None            # 置信度 [0, 1]
    words: list[dict[str, Any]] | None = None  # 逐词时间戳 [{"word": "...", "start": 0.1, "end": 0.5}]


@dataclass(frozen=True)
class TTSResult:
    """语音合成结果"""
    audio_data: bytes                          # 音频二进制数据
    mime_type: str = "audio/wav"               # 输出格式
    duration: float | None = None              # 语音时长（秒）


@dataclass(frozen=True)
class ImageAnalysisResult:
    """图像分析结果"""
    description: str                           # 图像描述文本
    labels: list[str] | None = None            # 识别标签
    ocr_text: str | None = None                # OCR 提取的文本
    objects: list[dict[str, Any]] | None = None  # 检测到的对象 [{"label": "cat", "bbox": [...], "confidence": 0.95}]


@dataclass(frozen=True)
class FileParseResult:
    """文件解析结果"""
    text_content: str                          # 提取的文本内容
    mime_type: str                             # 原始 MIME type
    pages: int | None = None                   # 页数（PDF/Word）
    metadata: dict[str, Any] | None = None     # 文档元信息（作者、标题等）
```

---

## 3. 基类设计 `multimodal/base.py`

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator
from multimodal.protocol import (
    MultimodalInput, MultimodalOutput, ASRResult, TTSResult,
    ImageAnalysisResult, FileParseResult,
)


class BaseEncoder(ABC):
    """
    所有多模态编码器的基类。

    设计原则：
    - 所有方法均为 async，统一异步 IO
    - 输入输出使用内部协议对象，不暴露 provider SDK 类型
    - 每个实例绑定一个 provider + model，线程安全
    """

    @property
    @abstractmethod
    def encoder_name(self) -> str:
        """编码器标识，如 "whisper"、"faster-whisper""""
        ...

    @property
    @abstractmethod
    def supported_modalities(self) -> list[str]:
        """支持的模态类型"""
        ...


class BaseASR(BaseEncoder, ABC):
    """语音识别接口"""

    @abstractmethod
    async def transcribe(
        self,
        audio_data: bytes,
        mime_type: str | None = None,
        language: str | None = None,
    ) -> ASRResult:
        """
        非流式语音识别。

        Args:
            audio_data: 音频二进制数据
            mime_type:  MIME type，如 "audio/wav"、"audio/mp3"
            language:   指定语言代码（zh, en 等），None 则自动检测

        Returns:
            ASRResult
        """
        ...

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str | None = None,
    ) -> AsyncIterator[str]:
        """
        流式语音识别（增量输出）。

        默认实现：收集所有 chunk 后调用 transcribe()。
        可被子类覆盖为真正的流式处理。

        Yields:
            识别文本增量
        """
        collected = b""
        async for chunk in audio_chunks:
            collected += chunk
        result = await self.transcribe(collected, language=language)
        yield result.text


class BaseTTS(BaseEncoder, ABC):
    """语音合成接口"""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """
        文本转语音。

        Args:
            text:   要合成的文本
            voice:  音色标识（None 使用默认）
            speed:  语速倍数 [0.5, 4.0]

        Returns:
            TTSResult
        """
        ...

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> AsyncIterator[bytes]:
        """
        流式语音合成（增量输出音频 chunk）。

        默认实现：调用 synthesize() 后一次性返回。
        可被子类覆盖为真正的流式处理。

        Yields:
            音频数据增量
        """
        result = await self.synthesize(text, voice=voice, speed=speed)
        yield result.audio_data


class BaseImageAnalyzer(BaseEncoder, ABC):
    """图像分析接口"""

    @abstractmethod
    async def analyze(
        self,
        image_data: bytes,
        mime_type: str | None = None,
        tasks: list[str] | None = None,
    ) -> ImageAnalysisResult:
        """
        图像分析。

        Args:
            image_data: 图像二进制数据
            mime_type:  MIME type，如 "image/png"、"image/jpeg"
            tasks:      要执行的分析任务列表：
                        - "describe": 生成描述文本
                        - "ocr":     OCR 文字识别
                        - "labels":  标签分类
                        - "objects": 对象检测
                        None 则执行所有可用任务

        Returns:
            ImageAnalysisResult
        """
        ...


class BaseFileParser(BaseEncoder, ABC):
    """文件解析接口"""

    @abstractmethod
    async def parse(
        self,
        file_data: bytes,
        mime_type: str | None = None,
        filename: str | None = None,
    ) -> FileParseResult:
        """
        文件内容提取。

        Args:
            file_data: 文件二进制数据
            mime_type: MIME type（None 则自动检测）
            filename:  文件名（用于辅助判断格式）

        Returns:
            FileParseResult

        Raises:
            ValueError: 不支持的文件类型
        """
        ...
```

---

## 4. Encoder 实现

### 4.1 Whisper ASR `multimodal/whisper_asr.py`

```python
from typing import AsyncIterator
import asyncio
import tempfile
import os

from multimodal.base import BaseASR
from multimodal.protocol import ASRResult


class WhisperASR(BaseASR):
    """
    OpenAI Whisper 语音识别。

    支持两种模式：
    - API 模式：调用 OpenAI Whisper API（需 api_key）
    - 本地模式：使用 openai-whisper 包本地推理（需 GPU）
    """

    def __init__(
        self,
        mode: str = "api",                    # "api" | "local"
        model: str = "whisper-1",             # API 模式固定 whisper-1；本地模式可选 tiny/base/small/medium/large
        api_key: str | None = None,
        language: str | None = None,          # 默认语言
    ):
        self._mode = mode
        self._model = model
        self._default_language = language

        if mode == "api":
            import openai
            self._client = openai.AsyncOpenAI(api_key=api_key)
        else:
            import whisper
            self._model_instance = whisper.load_model(model)

    @property
    def encoder_name(self) -> str:
        return f"whisper-{self._mode}"

    @property
    def supported_modalities(self) -> list[str]:
        return ["audio"]

    async def transcribe(
        self,
        audio_data: bytes,
        mime_type: str | None = None,
        language: str | None = None,
    ) -> ASRResult:
        if self._mode == "api":
            return await self._transcribe_api(audio_data, mime_type, language)
        else:
            return await self._transcribe_local(audio_data, language)

    async def _transcribe_api(
        self, audio_data: bytes, mime_type: str | None, language: str | None
    ) -> ASRResult:
        ext = "wav"
        if mime_type:
            ext_map = {"audio/mp3": "mp3", "audio/mpeg": "mp3", "audio/wav": "wav", "audio/flac": "flac"}
            ext = ext_map.get(mime_type, "wav")

        async with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            file_obj = open(temp_path, "rb")
            resp = await self._client.audio.transcriptions.create(
                model="whisper-1",
                file=file_obj,
                language=language or self._default_language,
                response_format="verbose_json",  # 获取逐词时间戳
            )
            file_obj.close()

            return ASRResult(
                text=resp.text,
                language=resp.language,
                words=[{"word": w["word"], "start": w["start"], "end": w["end"]} for w in getattr(resp, "words", []) or []],
            )
        finally:
            os.unlink(temp_path)

    async def _transcribe_local(
        self, audio_data: bytes, language: str | None
    ) -> ASRResult:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._model_instance.transcribe(
                audio_data if isinstance(audio_data, (bytes, bytearray)) else audio_data,
                language=language or self._default_language,
                verbose=False,
            ),
        )

        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append({"word": w["text"], "start": w["start"], "end": w["end"]})

        return ASRResult(
            text=result["text"],
            language=language or self._default_language,
            confidence=None,  # local whisper 不提供整体置信度
            words=words if words else None,
        )


class FasterWhisperASR(BaseASR):
    """
    faster-whisper 语音识别（CTranslate2 加速版）。

    比原版 whisper 快 4x，内存占用少 50%。
    适合本地部署场景。
    """

    def __init__(
        self,
        model: str = "large-v3",
        device: str = "auto",   # "cpu" | "cuda" | "auto"
    ):
        from faster_whisper import WhisperModel
        self._model_instance = WhisperModel(model, device=device)
        self._model_name = model

    @property
    def encoder_name(self) -> str:
        return f"faster-whisper-{self._model_name}"

    @property
    def supported_modalities(self) -> list[str]:
        return ["audio"]

    async def transcribe(
        self,
        audio_data: bytes,
        mime_type: str | None = None,
        language: str | None = None,
    ) -> ASRResult:
        import tempfile
        import os

        # faster-whisper 接受文件路径或 numpy array
        async with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            segments, info = self._model_instance.transcribe(
                temp_path,
                language=language,
                beam_size=5,
            )

            text_parts = []
            words = []
            for seg in segments:
                text_parts.append(seg.text)
                for w in seg.words or []:
                    words.append({"word": w.word, "start": w.start, "end": w.end})

            return ASRResult(
                text="".join(text_parts),
                language=info.language,
                confidence=info.language_probability,
                words=words if words else None,
            )
        finally:
            os.unlink(temp_path)
```

### 4.2 OpenAI TTS `multimodal/openai_tts.py`

```python
from typing import AsyncIterator
import httpx

from multimodal.base import BaseTTS
from multimodal.protocol import TTSResult


class OpenAITTS(BaseTTS):
    """
    OpenAI TTS 语音合成。

    支持模型：tts-1, tts-1-hd
    支持音色：alloy, echo, fable, onyx, nova, shimmer
    """

    def __init__(
        self,
        api_key: str,
        model: str = "tts-1",
        voice: str = "alloy",
        base_url: str | None = None,
    ):
        self._api_key = api_key
        self._model = model
        self._default_voice = voice
        self._base_url = base_url or "https://api.openai.com/v1"
        self._client = httpx.AsyncClient(timeout=60.0)

    @property
    def encoder_name(self) -> str:
        return "openai-tts"

    @property
    def supported_modalities(self) -> list[str]:
        return ["text"]  # text → audio

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> TTSResult:
        resp = await self._client.post(
            f"{self._base_url}/audio/speech",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "input": text,
                "voice": voice or self._default_voice,
                "speed": speed,
                "response_format": "wav",
            },
        )
        resp.raise_for_status()

        import math
        # 粗略估算时长：中文约 4 字/秒，英文约 2 词/秒
        estimated_duration = len(text) / (4 if any('\u4e00' <= c <= '\u9fff' for c in text) else 15)

        return TTSResult(
            audio_data=resp.content,
            mime_type="audio/wav",
            duration=estimated_duration,
        )

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> AsyncIterator[bytes]:
        """OpenAI TTS API 不支持真正的流式输出，这里分段模拟"""
        # 按句子切分，逐段合成
        import re
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        for sentence in sentences:
            if not sentence.strip():
                continue
            result = await self.synthesize(sentence, voice=voice, speed=speed)
            yield result.audio_data

    async def __del__(self):
        await self._client.aclose()


class EdgeTTS(BaseTTS):
    """
    Microsoft Edge TTS（免费、无需 API Key）。

    通过 edge-tts 包调用 Edge 浏览器的内置 TTS 服务。
    适合低成本场景，但稳定性和速度不如商业 API。
    """

    def __init__(
        self,
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ):
        self._default_voice = voice
        self._rate = rate
        self._pitch = pitch

    @property
    def encoder_name(self) -> str:
        return "edge-tts"

    @property
    def supported_modalities(self) -> list[str]:
        return ["text"]

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> TTSResult:
        import asyncio
        import tempfile
        import os
        import edge_tts

        communicate = edge_tts.Communicate(
            text,
            voice or self._default_voice,
            rate=self._rate,
            pitch=self._pitch,
        )

        async with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name

        try:
            async with edge_tts.StreamMux() as mux:
                mux.register_sink(temp_path)
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        pass  # 已写入文件通过 mux
            audio_data = open(temp_path, "rb").read()
        finally:
            os.unlink(temp_path)

        return TTSResult(
            audio_data=audio_data,
            mime_type="audio/mp3",
        )
```

### 4.3 图像分析 `multimodal/image_analyzer.py`

```python
from multimodal.base import BaseImageAnalyzer
from multimodal.protocol import ImageAnalysisResult


class OpenAIImageAnalyzer(BaseImageAnalyzer):
    """
    使用 GPT-4o / GPT-4V 进行图像分析。

    支持：描述生成、OCR、标签分类、对象检测。
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        import openai
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model

    @property
    def encoder_name(self) -> str:
        return f"openai-vision-{self._model}"

    @property
    def supported_modalities(self) -> list[str]:
        return ["image"]

    async def analyze(
        self,
        image_data: bytes,
        mime_type: str | None = None,
        tasks: list[str] | None = None,
    ) -> ImageAnalysisResult:
        import base64

        if not mime_type:
            mime_type = "image/png"

        b64_image = base64.b64encode(image_data).decode("utf-8")
        image_url = f"data:{mime_type};base64,{b64_image}"

        tasks = tasks or ["describe", "ocr", "labels"]
        prompt_parts = []
        if "describe" in tasks:
            prompt_parts.append("请详细描述这张图片的内容。")
        if "ocr" in tasks:
            prompt_parts.append("提取图片中所有可见文字（OCR）。如果没有文字，此项留空。")
        if "labels" in tasks:
            prompt_parts.append("列出图片中的主要标签/类别（逗号分隔）。")

        prompt = "\n".join(prompt_parts) + "\n\n请以 JSON 格式回复：{\"description\": \"...\", \"ocr_text\": \"...\", \"labels\": [...]}。没有的内容用 null 表示。"

        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "你是一个图像分析助手，请以 JSON 格式返回结果。"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            max_tokens=2048,
        )

        import json
        text = resp.choices[0].message.content or "{}"
        # 提取 JSON（处理可能包含 markdown code block 的情况）
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {}

        return ImageAnalysisResult(
            description=data.get("description", ""),
            ocr_text=data.get("ocr_text"),
            labels=data.get("labels"),
        )


class LocalOCRAnalyzer(BaseImageAnalyzer):
    """
    本地 OCR 分析器（PaddleOCR / Tesseract）。

    适合离线场景，中文识别推荐 PaddleOCR。
    """

    def __init__(self, engine: str = "paddleocr"):
        self._engine = engine
        if engine == "paddleocr":
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        else:
            import pytesseract
            self._tesseract = pytesseract

    @property
    def encoder_name(self) -> str:
        return f"local-ocr-{self._engine}"

    @property
    def supported_modalities(self) -> list[str]:
        return ["image"]

    async def analyze(
        self,
        image_data: bytes,
        mime_type: str | None = None,
        tasks: list[str] | None = None,
    ) -> ImageAnalysisResult:
        import tempfile
        import os
        import asyncio

        loop = asyncio.get_event_loop()

        async with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_data)
            temp_path = f.name

        try:
            if self._engine == "paddleocr":
                result = await loop.run_in_executor(None, self._ocr.ocr, temp_path, cls=True)
                ocr_text = ""
                for line in (result[0] or []):
                    ocr_text += line[1][0] + "\n"
            else:
                import PIL.Image
                img = PIL.Image.open(temp_path)
                ocr_text = await loop.run_in_executor(None, self._tesseract.image_to_string, img, lang="chi_sim+eng")

            return ImageAnalysisResult(
                description="",  # 本地 OCR 不做描述
                ocr_text=ocr_text.strip() or None,
            )
        finally:
            os.unlink(temp_path)
```

### 4.4 文件解析 `multimodal/file_parser.py`

```python
from multimodal.base import BaseFileParser
from multimodal.protocol import FileParseResult


class UniversalFileParser(BaseFileParser):
    """
    通用文件解析器。

    支持格式：PDF、DOCX、XLSX、TXT、CSV、Markdown、HTML、图片(OCR)。
    自动检测 MIME type，路由到对应解析器。
    """

    def __init__(self, ocr_engine: BaseImageAnalyzer | None = None):
        self._ocr = ocr_engine
        # MIME type → 解析方法映射
        self._handlers: dict[str, str] = {
            "application/pdf": "_parse_pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "_parse_docx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "_parse_xlsx",
            "text/plain": "_parse_text",
            "text/csv": "_parse_csv",
            "text/markdown": "_parse_text",
            "text/html": "_parse_html",
            "image/png": "_parse_image",
            "image/jpeg": "_parse_image",
            "image/gif": "_parse_image",
        }

    @property
    def encoder_name(self) -> str:
        return "universal-file-parser"

    @property
    def supported_modalities(self) -> list[str]:
        return ["file"]

    async def parse(
        self,
        file_data: bytes,
        mime_type: str | None = None,
        filename: str | None = None,
    ) -> FileParseResult:
        # 自动检测 MIME type
        if not mime_type:
            import magic
            mime_type = magic.from_buffer(file_data, mime=True)

        handler_name = self._handlers.get(mime_type)
        if not handler_name:
            # 尝试通过文件名判断
            if filename:
                ext = filename.rsplit(".", 1)[-1].lower()
                ext_map = {
                    "pdf": "_parse_pdf", "docx": "_parse_docx", "xlsx": "_parse_xlsx",
                    "txt": "_parse_text", "csv": "_parse_csv", "md": "_parse_text",
                    "html": "_parse_html", "htm": "_parse_html",
                    "png": "_parse_image", "jpg": "_parse_image", "jpeg": "_parse_image",
                }
                handler_name = ext_map.get(ext)

        if not handler_name:
            raise ValueError(f"Unsupported file type: {mime_type} (file: {filename})")

        handler = getattr(self, handler_name)
        return await handler(file_data, mime_type)

    # ---------- 各格式解析器 ----------

    async def _parse_pdf(self, data: bytes, mime_type: str) -> FileParseResult:
        import asyncio
        loop = asyncio.get_event_loop()

        def _extract():
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            pages = len(reader.pages)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            metadata = {}
            if reader.metadata:
                for key, value in reader.metadata.items():
                    metadata[key.replace("/", "")] = str(value)
            return text, pages, metadata

        text, pages, metadata = await loop.run_in_executor(None, _extract)
        return FileParseResult(text_content=text.strip(), mime_type=mime_type, pages=pages, metadata=metadata or None)

    async def _parse_docx(self, data: bytes, mime_type: str) -> FileParseResult:
        import asyncio
        loop = asyncio.get_event_loop()

        def _extract():
            import io
            from docx import Document
            doc = Document(io.BytesIO(data))
            text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
            return text, len(doc.paragraphs)

        text, _ = await loop.run_in_executor(None, _extract)
        return FileParseResult(text_content=text.strip(), mime_type=mime_type)

    async def _parse_xlsx(self, data: bytes, mime_type: str) -> FileParseResult:
        import asyncio
        loop = asyncio.get_event_loop()

        def _extract():
            import io
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
            lines = []
            for sheet in wb.worksheets:
                lines.append(f"## Sheet: {sheet.title}")
                for row in sheet.iter_rows(values_only=True):
                    lines.append("\t".join(str(v) if v is not None else "" for v in row))
            return "\n".join(lines), len(wb.worksheets)

        text, pages = await loop.run_in_executor(None, _extract)
        return FileParseResult(text_content=text.strip(), mime_type=mime_type, pages=pages)

    async def _parse_text(self, data: bytes, mime_type: str) -> FileParseResult:
        # 尝试多种编码
        for encoding in ["utf-8", "gbk", "latin-1"]:
            try:
                text = data.decode(encoding)
                return FileParseResult(text_content=text.strip(), mime_type=mime_type)
            except (UnicodeDecodeError, LookupError):
                continue
        raise ValueError(f"Cannot decode text file with any known encoding")

    async def _parse_csv(self, data: bytes, mime_type: str) -> FileParseResult:
        import io
        text = await self._parse_text(data, "text/csv")
        return FileParseResult(text_content=text.text_content, mime_type=mime_type)

    async def _parse_html(self, data: bytes, mime_type: str) -> FileParseResult:
        import asyncio
        loop = asyncio.get_event_loop()

        def _extract():
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(data, "html.parser")
            # 移除脚本和样式
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            return text

        text = await loop.run_in_executor(None, _extract)
        return FileParseResult(text_content=text.strip(), mime_type=mime_type)

    async def _parse_image(self, data: bytes, mime_type: str) -> FileParseResult:
        if not self._ocr:
            raise ValueError("Image parsing requires an OCR engine. Provide ocr_engine in constructor.")
        result = await self._ocr.analyze(data, mime_type=mime_type, tasks=["ocr"])
        return FileParseResult(
            text_content=result.ocr_text or "(No text detected)",
            mime_type=mime_type,
        )
```

---

## 5. 工厂与注册 `multimodal/factory.py`

```python
from typing import Type
from multimodal.base import BaseEncoder, BaseASR, BaseTTS, BaseImageAnalyzer, BaseFileParser


_ENCODER_REGISTRY: dict[str, Type[BaseEncoder]] = {}


def register_encoder(name: str, cls: Type[BaseEncoder]):
    """注册编码器"""
    _ENCODER_REGISTRY[name] = cls


def _auto_register():
    from multimodal.whisper_asr import WhisperASR, FasterWhisperASR
    from multimodal.openai_tts import OpenAITTS, EdgeTTS
    from multimodal.image_analyzer import OpenAIImageAnalyzer, LocalOCRAnalyzer
    from multimodal.file_parser import UniversalFileParser

    register_encoder("whisper-api", WhisperASR)
    register_encoder("faster-whisper", FasterWhisperASR)
    register_encoder("openai-tts", OpenAITTS)
    register_encoder("edge-tts", EdgeTTS)
    register_encoder("openai-vision", OpenAIImageAnalyzer)
    register_encoder("local-ocr", LocalOCRAnalyzer)
    register_encoder("universal-parser", UniversalFileParser)


_auto_register()


def create_encoder(encoder_type: str, config: dict) -> BaseEncoder:
    """根据类型 + 配置创建编码器实例"""
    cls = _ENCODER_REGISTRY.get(encoder_type)
    if cls is None:
        available = ", ".join(_ENCODER_REGISTRY.keys())
        raise ValueError(f"Unknown encoder '{encoder_type}'. Available: {available}")

    import inspect
    sig = inspect.signature(cls.__init__)
    params = {k: v for k, v in config.items() if k in sig.parameters and k != "self"}
    return cls(**params)


def list_encoders() -> list[str]:
    """列出已注册的编码器"""
    return list(_ENCODER_REGISTRY.keys())
```

---

## 6. Agent 集成 `multimodal/integration.py`

```python
"""
多模态模块与 Agent Core 的集成层。

职责：
- 将多模态输入转换为 LLM 可理解的 Message 格式
- 注册为 Tool，供 Agent 在对话中调用
- 处理流式 ASR → Agent → TTS 的全链路语音交互
"""

from typing import AsyncIterator
import base64
import logging

from tool.base import BaseTool, ToolResult, ToolDefinition
from multimodal.base import BaseASR, BaseTTS, BaseImageAnalyzer, BaseFileParser
from multimodal.protocol import Modality, MultimodalInput

logger = logging.getLogger(__name__)


class ASRTool(BaseTool):
    """语音识别工具，Agent 可调用"""

    name = "speech_to_text"
    description = "将音频转换为文本。支持 wav、mp3、flac 等格式。"
    parameters = {
        "type": "object",
        "properties": {
            "audio_base64": {"type": "string", "description": "音频数据的 base64 编码"},
            "mime_type": {"type": "string", "description": "MIME type，如 audio/wav"},
            "language": {"type": "string", "description": "语言代码（可选）：zh, en, ja..."},
        },
        "required": ["audio_base64"],
    }

    def __init__(self, asr_engine: BaseASR):
        super().__init__()
        self._asr = asr_engine

    async def execute(self, arguments: dict) -> ToolResult:
        audio_data = base64.b64decode(arguments["audio_base64"])
        result = await self._asr.transcribe(
            audio_data,
            mime_type=arguments.get("mime_type"),
            language=arguments.get("language"),
        )
        return ToolResult(
            success=True,
            content=result.text,
            metadata={"language": result.language} if result.language else None,
        )


class TTSTool(BaseTool):
    """语音合成工具，Agent 可调用"""

    name = "text_to_speech"
    description = "将文本转换为语音。返回音频的 base64 编码。"
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要合成的文本"},
            "voice": {"type": "string", "description": "音色标识（可选）"},
            "speed": {"type": "number", "description": "语速倍数 [0.5, 4.0]，默认 1.0"},
        },
        "required": ["text"],
    }

    def __init__(self, tts_engine: BaseTTS):
        super().__init__()
        self._tts = tts_engine

    async def execute(self, arguments: dict) -> ToolResult:
        result = await self._tts.synthesize(
            text=arguments["text"],
            voice=arguments.get("voice"),
            speed=float(arguments.get("speed", 1.0)),
        )
        audio_b64 = base64.b64encode(result.audio_data).decode("utf-8")
        return ToolResult(
            success=True,
            content=f"data:{result.mime_type};base64,{audio_b64}",
            metadata={"duration": result.duration},
        )


class ImageAnalysisTool(BaseTool):
    """图像分析工具，Agent 可调用"""

    name = "analyze_image"
    description = "分析图片内容：描述、OCR 文字识别、标签分类。"
    parameters = {
        "type": "object",
        "properties": {
            "image_base64": {"type": "string", "description": "图片数据的 base64 编码"},
            "mime_type": {"type": "string", "description": "MIME type，如 image/png"},
            "tasks": {
                "type": "array",
                "items": {"type": "string", "enum": ["describe", "ocr", "labels"]},
                "description": "分析任务列表（可选）",
            },
        },
        "required": ["image_base64"],
    }

    def __init__(self, analyzer: BaseImageAnalyzer):
        super().__init__()
        self._analyzer = analyzer

    async def execute(self, arguments: dict) -> ToolResult:
        image_data = base64.b64decode(arguments["image_base64"])
        result = await self._analyzer.analyze(
            image_data,
            mime_type=arguments.get("mime_type"),
            tasks=arguments.get("tasks"),
        )
        parts = []
        if result.description:
            parts.append(f"描述：{result.description}")
        if result.ocr_text:
            parts.append(f"OCR 文字：\n{result.ocr_text}")
        if result.labels:
            parts.append(f"标签：{', '.join(result.labels)}")

        return ToolResult(
            success=True,
            content="\n\n".join(parts) if parts else "(No analysis results)",
        )


class FileParseTool(BaseTool):
    """文件解析工具，Agent 可调用"""

    name = "parse_file"
    description = "提取文件中的文本内容。支持 PDF、Word、Excel、TXT、CSV、HTML、图片(OCR)。"
    parameters = {
        "type": "object",
        "properties": {
            "file_base64": {"type": "string", "description": "文件数据的 base64 编码"},
            "mime_type": {"type": "string", "description": "MIME type（可选，自动检测）"},
            "filename": {"type": "string", "description": "文件名（可选）"},
        },
        "required": ["file_base64"],
    }

    def __init__(self, parser: BaseFileParser):
        super().__init__()
        self._parser = parser

    async def execute(self, arguments: dict) -> ToolResult:
        file_data = base64.b64decode(arguments["file_base64"])
        result = await self._parser.parse(
            file_data,
            mime_type=arguments.get("mime_type"),
            filename=arguments.get("filename"),
        )
        # 如果内容过长，截断并提示
        content = result.text_content
        truncated = False
        if len(content) > 10000:
            content = content[:10000] + "\n\n... (文件内容较长，已截断)"
            truncated = True

        return ToolResult(
            success=True,
            content=content,
            metadata={
                "mime_type": result.mime_type,
                "pages": result.pages,
                "truncated": truncated,
            },
        )


# ==================== 语音交互链路 ====================

class VoiceInteractionPipeline:
    """
    ASR → Agent → TTS 全链路语音交互。

    Usage:
        pipeline = VoiceInteractionPipeline(asr_engine, agent_core, tts_engine)
        async for audio_chunk in pipeline.process_voice(audio_data):
            websocket.send_bytes(audio_chunk)
    """

    def __init__(self, asr: BaseASR, agent, tts: BaseTTS):
        self._asr = asr
        self._agent = agent       # AgentCore 实例
        self._tts = tts

    async def process_voice(
        self,
        audio_data: bytes,
        session_id: str,
        mime_type: str | None = None,
    ) -> AsyncIterator[bytes]:
        """
        处理语音输入，返回语音输出。

        Flow:
            音频 → ASR → 文本 → Agent → 回复文本 → TTS → 音频流
        """
        # Step 1: ASR
        logger.info("ASR transcription started")
        asr_result = await self._asr.transcribe(audio_data, mime_type=mime_type)
        logger.info(f"ASR result: {asr_result.text[:50]}...")

        # Step 2: Agent processing
        response_text = ""
        async for chunk in self._agent.process_stream(session_id, asr_result.text):
            if chunk.delta_content:
                response_text += chunk.delta_content

        if not response_text.strip():
            return

        # Step 3: TTS streaming
        logger.info("TTS synthesis started")
        async for audio_chunk in self._tts.synthesize_stream(response_text):
            yield audio_chunk
```

---

## 7. 架构总览

```
                    ┌─────────────────────┐
                    │     Agent Core      │
                    │   (通过 Tool 接口    │
                    │    调用多模态能力)    │
                    └──────────┬──────────┘
                               │ Tool Registry
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │ ASRTool     │  │ TTSTool     │  │ ImageTool   │
     │ (语音→文本)  │  │ (文本→语音)  │  │ (图片分析)   │
     └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
            ▼                ▼                 ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │ WhisperASR  │  │ OpenAITTS   │  │ GPT-4V      │
     │ FasterWhisp │  │ EdgeTTS     │  │ PaddleOCR   │
     └─────────────┘  └─────────────┘  └─────────────┘

     ┌──────────────────────────────────────────┐
     │       FileParseTool (文件解析)            │
     │  UniversalFileParser                     │
     │  PDF → pypdf    DOCX → python-docx      │
     │  XLSX → openpyxl HTML → BeautifulSoup   │
     └──────────────────────────────────────────┘

     ┌──────────────────────────────────────────┐
     │       VoiceInteractionPipeline           │
     │  ASR → Agent → TTS 全链路语音交互         │
     └──────────────────────────────────────────┘
```

---

## 8. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **统一抽象** | `BaseEncoder` ABC，所有编码器收敛到同一接口 |
| **即插即用** | Factory + Registry，新增编码器 = 1 个文件 |
| **Agent 集成** | 封装为 Tool，通过 ToolRegistry 注册，Agent 自动发现 |
| **语音链路** | `VoiceInteractionPipeline` 串联 ASR → Agent → TTS |
| **流式支持** | ASR/TTS 均提供 stream 方法，降低首字延迟 |
| **格式自适应** | `UniversalFileParser` 自动检测 MIME type，路由到对应解析器 |
| **本地/云端双模式** | Whisper API / Local、OpenAI TTS / Edge TTS 可配置切换 |
