# LingBot inference adapter attribution

`runtime.py` is an inference adaptation referencing Robbyant's LingBot-World-v2
at commit `1895d300d8ac936401689b26389f51cbd36530eb`, notably the causal-fast
sampling and conditioning conventions in `wan/image2video.py`.

Upstream: https://github.com/Robbyant/lingbot-world-v2

The adaptation in `runtime.py` is provided under **CC BY-NC-SA 4.0**, with the
license text in `LINGBOT_LICENSE.txt`. It is not offered under a more permissive
project-wide license. Upstream model weights and implementation remain external
dependencies; preserve their original notices and applicable terms. Do not claim
endorsement by Robbyant. Changes here add stateful bounded action requests and
full-prefix VAE decoding, not new model training. Non-commercial demo use only.

The separately written HTTP/session protocol and elementary camera mathematics
do not redistribute upstream weights or source files.
