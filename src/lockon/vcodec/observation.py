from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np

from lockon.protos.gym_env import gym_env_pb2
from lockon.utils import tensor_from_array

try:
    import av
except ImportError:  # pragma: no cover - runtime dependency
    av = None


RGB_TENSOR_DTYPE = "uint8"
JPEG_TENSOR_DTYPE = "jpeg_bytes"
H264_TENSOR_DTYPE = "h264_annexb"


class ObservationFormat(str, Enum):
    RGB = "rgb"
    JPEG = "jpeg"
    H264 = "h264"

    @classmethod
    def from_value(cls, value: str) -> "ObservationFormat":
        try:
            return cls(value)
        except ValueError as exc:
            supported = ", ".join(member.value for member in cls)
            raise ValueError(f"unsupported observation format: {value}, expected one of {supported}") from exc


@dataclass
class ObservationCodecConfig:
    observation_format: ObservationFormat = ObservationFormat.RGB
    jpeg_quality: int = 80
    h264_bitrate_kbps: int = 4000
    h264_gop: int = 30


def _tensor_from_bytes(data: bytes, dtype: str) -> gym_env_pb2.Tensor:
    return gym_env_pb2.Tensor(data=data, shape=[len(data)], dtype=dtype)


def _array_from_rgb_tensor(tensor: gym_env_pb2.Tensor) -> np.ndarray:
    if tensor.dtype != RGB_TENSOR_DTYPE:
        raise ValueError(f"expected rgb tensor dtype {RGB_TENSOR_DTYPE}, got {tensor.dtype}")
    return np.frombuffer(tensor.data, dtype=np.uint8).reshape(tuple(tensor.shape))


class ObservationEncoder:
    tensor_dtype: str

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def encode(self, frame_rgb: np.ndarray) -> tuple[gym_env_pb2.Tensor, dict[str, Any]]:
        raise NotImplementedError


class ObservationDecoder:
    tensor_dtype: str

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def decode(self, observation: gym_env_pb2.Tensor, info: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError


class RgbObservationEncoder(ObservationEncoder):
    tensor_dtype = RGB_TENSOR_DTYPE

    def encode(self, frame_rgb: np.ndarray) -> tuple[gym_env_pb2.Tensor, dict[str, Any]]:
        frame = np.asarray(frame_rgb, dtype=np.uint8)
        height, width = frame.shape[:2]
        return tensor_from_array(frame), {
            "frame_codec": ObservationFormat.RGB.value,
            "width": int(width),
            "height": int(height),
        }


class RgbObservationDecoder(ObservationDecoder):
    tensor_dtype = RGB_TENSOR_DTYPE

    def decode(self, observation: gym_env_pb2.Tensor, info: dict[str, Any]) -> np.ndarray:
        return _array_from_rgb_tensor(observation).copy()


class JpegObservationEncoder(ObservationEncoder):
    tensor_dtype = JPEG_TENSOR_DTYPE

    def __init__(self, quality: int = 80) -> None:
        self.quality = int(quality)

    def encode(self, frame_rgb: np.ndarray) -> tuple[gym_env_pb2.Tensor, dict[str, Any]]:
        frame = np.asarray(frame_rgb, dtype=np.uint8)
        height, width = frame.shape[:2]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            raise RuntimeError("failed to encode jpeg observation")

        return _tensor_from_bytes(encoded.tobytes(), self.tensor_dtype), {
            "frame_codec": ObservationFormat.JPEG.value,
            "width": int(width),
            "height": int(height),
        }


class JpegObservationDecoder(ObservationDecoder):
    tensor_dtype = JPEG_TENSOR_DTYPE

    def decode(self, observation: gym_env_pb2.Tensor, info: dict[str, Any]) -> np.ndarray:
        encoded = np.frombuffer(observation.data, dtype=np.uint8)
        frame_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError("failed to decode jpeg observation")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


class H264ObservationEncoder(ObservationEncoder):
    tensor_dtype = H264_TENSOR_DTYPE

    def __init__(self, bitrate_kbps: int = 4000, gop: int = 30) -> None:
        self.bitrate_kbps = int(bitrate_kbps)
        self.gop = int(gop)
        self._codec = None
        self._width: int | None = None
        self._height: int | None = None
        self._pts = 0

    def reset(self) -> None:
        self.close()
        self._pts = 0

    def close(self) -> None:
        if self._codec is None:
            return
        try:
            self._codec.encode(None)
        except Exception:
            pass
        self._codec = None
        self._width = None
        self._height = None

    def _create_codec(self, width: int, height: int):
        if av is None:
            raise RuntimeError("PyAV is required for h264 observation format")

        last_error: Exception | None = None
        for codec_name in ("libx264", "h264"):
            try:
                codec = av.CodecContext.create(codec_name, "w")
                codec.width = width
                codec.height = height
                codec.pix_fmt = "yuv420p"
                codec.gop_size = self.gop
                codec.max_b_frames = 0

                # Setting codec.bit_rate on the bare CodecContext makes libx264
                # output luma-only frames in our runtime setup. Keep color
                # fidelity first, and tune quality with CRF while preserving the
                # existing bitrate knob for future transport tuning.
                crf = "18" if self.bitrate_kbps >= 4000 else "23"
                codec.options = {
                    "preset": "ultrafast",
                    "tune": "zerolatency",
                    "g": str(self.gop),
                    "keyint": str(self.gop),
                    "sc_threshold": "0",
                    "crf": crf,
                }
                codec.open()
                return codec
            except Exception as exc:  # pragma: no cover - depends on runtime ffmpeg build
                last_error = exc

        raise RuntimeError("failed to initialize h264 encoder") from last_error

    def _ensure_codec(self, width: int, height: int) -> None:
        if self._codec is not None and self._width == width and self._height == height:
            return
        self.reset()
        self._codec = self._create_codec(width, height)
        self._width = width
        self._height = height

    def encode(self, frame_rgb: np.ndarray) -> tuple[gym_env_pb2.Tensor, dict[str, Any]]:
        frame = np.asarray(frame_rgb, dtype=np.uint8)
        height, width = frame.shape[:2]
        self._ensure_codec(width, height)

        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame = video_frame.reformat(width=width, height=height, format="yuv420p")
        video_frame.pts = self._pts
        self._pts += 1

        packets = self._codec.encode(video_frame)
        data = b"".join(bytes(packet) for packet in packets)
        if not data:
            raise RuntimeError("h264 encoder produced no packet for current frame")

        return _tensor_from_bytes(data, self.tensor_dtype), {
            "frame_codec": ObservationFormat.H264.value,
            "keyframe": any(packet.is_keyframe for packet in packets),
            "pts": int(video_frame.pts),
            "width": int(width),
            "height": int(height),
        }


class H264ObservationDecoder(ObservationDecoder):
    tensor_dtype = H264_TENSOR_DTYPE

    def __init__(self) -> None:
        self._codec = None

    def reset(self) -> None:
        self.close()

    def close(self) -> None:
        self._codec = None

    def _ensure_codec(self):
        if self._codec is not None:
            return self._codec
        if av is None:
            raise RuntimeError("PyAV is required for h264 observation format")
        self._codec = av.CodecContext.create("h264", "r")
        return self._codec

    def decode(self, observation: gym_env_pb2.Tensor, info: dict[str, Any]) -> np.ndarray:
        if observation.dtype != self.tensor_dtype:
            raise ValueError(f"expected h264 tensor dtype {self.tensor_dtype}, got {observation.dtype}")

        codec = self._ensure_codec()

        try:
            frames = codec.decode(av.Packet(observation.data))
        except Exception as exc:  # pragma: no cover - runtime codec behavior
            raise RuntimeError("failed to decode h264 observation") from exc

        if not frames:
            raise RuntimeError("h264 decoder produced no frame for current observation")

        return frames[-1].to_ndarray(format="rgb24")


def create_observation_encoder(
    observation_format: str | ObservationFormat,
    *,
    jpeg_quality: int = 80,
    h264_bitrate_kbps: int = 4000,
    h264_gop: int = 30,
) -> ObservationEncoder:
    codec_format = (
        observation_format
        if isinstance(observation_format, ObservationFormat)
        else ObservationFormat.from_value(observation_format)
    )
    if codec_format is ObservationFormat.RGB:
        return RgbObservationEncoder()
    if codec_format is ObservationFormat.JPEG:
        return JpegObservationEncoder(quality=jpeg_quality)
    return H264ObservationEncoder(bitrate_kbps=h264_bitrate_kbps, gop=h264_gop)


def create_observation_decoder(tensor_dtype: str) -> ObservationDecoder:
    if tensor_dtype == RGB_TENSOR_DTYPE:
        return RgbObservationDecoder()
    if tensor_dtype == JPEG_TENSOR_DTYPE:
        return JpegObservationDecoder()
    if tensor_dtype == H264_TENSOR_DTYPE:
        return H264ObservationDecoder()
    raise ValueError(f"unsupported observation tensor dtype: {tensor_dtype}")
