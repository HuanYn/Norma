# Third-party notices

## Optional multilingual OpenCLIP model

Norma can download and use
[`laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k`](https://huggingface.co/laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k)
through `open-clip-torch`. The model card lists the model license as MIT. Model
weights and the PyTorch/OpenCLIP runtime are optional downloads and are not
bundled in this repository.

## Optional OpenCV Zoo face models

Norma's default people-analysis provider can download the pinned OpenCV Zoo
[YuNet 2023mar face detector](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
and
[SFace 2021dec face recognizer](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface).
The YuNet model files are distributed under the MIT License, and the SFace
model files are distributed under the Apache License 2.0. Their weights are
downloaded on demand, verified against fixed SHA-256 values, and are not
bundled in this repository.

## Optional Qwen3-VL grounded-generation model

Norma can use the official
[`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
snapshot for local grounded multimodal generation. The model card declares the
weights under the Apache License 2.0. The multi-gigabyte weights are an optional
local download and are never bundled in this repository. Norma accepts only the
pinned model revision and content manifest documented by the runtime; it loads
with `local_files_only=True` and does not fall back to a model Hub during an API
request. Source photos and generated audit records stay on the local machine.

## Public multimodal smoke fixture

The optional reproducible smoke scripts download
[`Gothic-architecture-banner.jpg`](https://commons.wikimedia.org/wiki/File:Gothic-architecture-banner.jpg),
"Roof of Milan cathedral," by Traveler100. Wikimedia Commons lists the image
under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0). The image
is not bundled in this repository; `scripts/download_public_smoke_image.py`
pins its 1920-pixel derivative by byte size and SHA-256 and writes the complete
attribution beside the downloaded file.

## Controlled preference-study image fixture

The controlled contextual-preference and CAPU-PDRR experiments use 70 images
from a 72-image Wikimedia Commons fixture. The image binaries are not bundled
in this repository. The version-controlled
`fixtures/contextual_preference_wikimedia_20260814.json` records, for every
image, the Commons source page, author/credit metadata, reported license,
historical derivative URL, byte size, and SHA-256 observed by the experiment.
License terms vary by image and must be followed individually.

Wikimedia thumbnail URLs are not immutable archives. A 2026-08-29 audit found
that some current responses no longer reproduce the historical bytes or pixels;
the manifest therefore remains an audit record and fail-closed verifier, not a
promise that mutable upstream URLs can reconstruct the old experiment forever.
Exact historical files stay local and untracked unless a separately licensed,
content-addressed research archive is published.

## pianke-desktop / pianke-core

Norma vendors and adapts portions of
[`xishouyunxing/pianke-desktop`](https://github.com/xishouyunxing/pianke-desktop),
including its fast image hash, quality scoring, and clustering core.

MIT License

Copyright (c) 2026 xishouyunxing

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
