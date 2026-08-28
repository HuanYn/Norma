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
