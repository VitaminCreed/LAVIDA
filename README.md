# No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection (LAVIDA)

[![arXiv](https://img.shields.io/badge/arXiv-2602.19248-b31b1b.svg?logo=arXiv)](http://arxiv.org/abs/2602.19248)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is the ```official open-source``` implementation of [No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection](http://arxiv.org/abs/2602.19248) by Zunkai Dai, Ke Li, Jiajia Liu, Jie Yang, Yuanyuan Qiao. 

## 📌 Description

Video anomaly detection (VAD) has long faced challenges due to the rare occurrence and spatio-temporal scarcity of anomalous events. Existing methods often struggle in open-world scenarios because of limited dataset diversity and an inadequate understanding of context-dependent semantics.

**To address these issues, we propose LAVIDA, an end-to-end zero-shot video anomaly detection framework.**

### Key Contributions:
* We propose an end-to-end zero-shot VAD framework, LAVIDA, which leverages MLLMs to extract video anomaly semantic representations and enables frame- /pixel-level open-world anomaly detection.
* We introduce an Anomaly Exposure Sampler: a training strategy that repurposes segmentation targets as pseudo-anomalies, enabling training without VAD data and improving adaptability to diverse scenarios.
* We design a token compression method for LLM-based VAD model, which mitigates background interference and reduces computational costs for LLMs.

Extensive experiments show that our method achieves state-of-the-art zero-shot performance, and achieves competitive results w.r.t. unsupervised VAD methods at the frame level, and competitive zero-shot performance at the pixel level.


---

## 📅 Todo / Coming Soon
- [x] Release train and inference code.
- [ ] Add instructions for dataset preparation.
- [ ] Provide usage instructions.

## 🛠️ Installation
*(Wait for further updates)*

## 🚀 Usage
*(Wait for further updates)*

## ✉️ Citation
If you find our work helpful for your research, please consider citing:
```bibtex
@misc{dai2026needrealanomalymllm,
      title={No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection}, 
      author={Zunkai Dai and Ke Li and Jiajia Liu and Jie Yang and Yuanyuan Qiao},
      year={2026},
      eprint={2602.19248},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.19248}, 
}