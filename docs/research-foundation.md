# Research Foundation for DeepForensics

## Project Title

**Explainable Forensic Framework for Deepfake Detection through Visual–Semantic Consistency Analysis**

## Project Summary

DeepForensics is a deepfake forensics system that combines multiple state-of-the-art detection models into one practical web application. The system accepts image and video inputs, runs detector services through a FastAPI gateway, fuses their outputs using ensemble methods, and presents the result through a forensic-style frontend with confidence scores, evidence summaries, and Grad-CAM-style visual explanations.

The research foundation of the project is based on generalizable deepfake detection, CLIP-based fake image recognition, video deepfake detection using EfficientNet and Vision Transformers, and ensemble fusion. The current implementation uses active Docker services for image and video detection, while additional model folders provide optional extension paths for audio and more image detectors.

## Research Objectives

1. Detect whether an uploaded image or video is likely authentic or manipulated.
2. Combine multiple detector outputs into a single final verdict.
3. Improve interpretability by showing confidence, model votes, evidence summaries, and visual attention maps.
4. Build a modular architecture where new detectors can be added without rewriting the full application.
5. Provide a practical full-stack deployment using Docker, FastAPI, React, and Nginx.

## Active Research Papers Used

### 1. Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection

**Authors:** Chuangchuang Tan, Huan Liu, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, Yunchao Wei  
**Link:** https://arxiv.org/abs/2312.10461  
**Implemented as:** NPR Deepfake Detection  
**Repository reference:** `models/image/npr_deepfakedetection/`

This paper introduces Neighboring Pixel Relationships, or NPR, for detecting artifacts produced by up-sampling operations in CNN-based generative models. The method is useful because many synthetic images contain subtle local pixel dependencies created during generation. DeepForensics uses this model as one of its active image detectors.

**Role in DeepForensics:**

- Detects local structural artifacts in generated images.
- Helps identify synthetic images created by GAN or diffusion-based generators.
- Provides one model vote and probability score for ensemble fusion.

### 2. Towards Universal Fake Image Detectors that Generalize Across Generative Models

**Authors:** Utkarsh Ojha, Yuheng Li, Yong Jae Lee  
**Link:** https://arxiv.org/abs/2302.10174  
**Implemented as:** UniversalFakeDetect  
**Repository reference:** `models/image/universalfakedetect/`

This paper studies fake image detection across unseen generative models. Instead of training only on a fixed generator family, it uses features from a large pretrained vision-language model to improve generalization. DeepForensics uses this as an active image detector through the UniversalFakeDetect integration.

**Role in DeepForensics:**

- Provides CLIP-based image feature extraction.
- Improves generalization to unseen image generators.
- Supports the ensemble by detecting broad fake-image patterns.

### 3. Combining EfficientNet and Vision Transformers for Video Deepfake Detection

**Authors:** Davide Coccomini, Nicola Messina, Claudio Gennaro, Fabrizio Falchi  
**Link:** https://arxiv.org/abs/2107.02612  
**Implemented as:** Cross-Efficient ViT  
**Repository reference:** `models/video/cross_efficient_vit/`

This paper combines EfficientNet feature extraction with Vision Transformer-based reasoning for video deepfake detection. It focuses on facial video manipulation and uses frame-level analysis with a voting-style inference procedure.

**Role in DeepForensics:**

- Provides video deepfake detection.
- Extracts visual patterns across sampled frames.
- Adds video-specific evidence to the overall architecture.

### 4. Learning Transferable Visual Models From Natural Language Supervision

**Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever  
**Link:** https://arxiv.org/abs/2103.00020  
**Used by:** CLIP-based UniversalFakeDetect

CLIP is not a deepfake detector by itself in this project, but it is an important foundation for UniversalFakeDetect. CLIP learns visual representations from image-text pairs and transfers well to downstream classification tasks. In DeepForensics, CLIP features help detect fake images across different generators.

**Role in DeepForensics:**

- Provides transferable visual features.
- Supports generalization beyond one fixed fake-image generator.
- Forms the backbone for the UniversalFakeDetect model.

## Optional or Extension Research References

### 5. AI-Synthesized Voice Detection Using Neural Vocoder Artifacts

**Authors:** Chengzhe Sun, Shan Jia, Shuwei Hou, Siwei Lyu  
**Link:** https://openaccess.thecvf.com/content/CVPR2023W/WMF/html/Sun_AI-Synthesized_Voice_Detection_Using_Neural_Vocoder_Artifacts_CVPRW_2023_paper.html  
**Repository reference:** `models/audio/vocoder_artifacts/`

This paper detects synthesized speech by identifying neural vocoder artifacts. The repository contains an audio model integration folder for this work, but audio detection is not part of the currently active frontend flow unless enabled and configured.

**Possible extension:**

- Add audio upload support.
- Detect AI-generated speech.
- Combine audio, image, and video signals into a multimodal forensic verdict.

### 6. DeepfakeBench-based Detectors

**Repository references:**

- `models/image/ucf_deepfake_detection/`
- `models/image/spsl_deepfake_detection/`

These folders integrate detectors through DeepfakeBench. They can be discussed as future extension modules if enabled in Docker Compose and the model registry.

**Possible extension:**

- Add more image detectors to improve ensemble robustness.
- Compare detector performance across multiple datasets.
- Evaluate whether extra detectors improve voting and stacking results.

## System Architecture Based on Research

DeepForensics is structured as a modular ensemble system:

1. **Input Layer**
   - User uploads an image or video from the frontend.

2. **API Gateway**
   - FastAPI receives the file.
   - The gateway identifies media type and dispatches requests to configured detector services.

3. **Detector Services**
   - NPR processes image artifacts.
   - UniversalFakeDetect processes CLIP-based image features.
   - Cross-Efficient ViT processes video frames.

4. **Fusion Layer**
   - The gateway combines detector results using voting, averaging, or stacking.
   - The final result includes fake probability, model votes, confidence, and inference metadata.

5. **Explainability Layer**
   - The frontend presents a verdict, evidence summary, model support, verification guidance, and Grad-CAM-style heatmap visualization.
   - The current heatmap is frontend-generated from the preview and ensemble score. True model-level Grad-CAM would require detector services to return gradient-based heatmap data.

## Research Gap Addressed

Many deepfake detectors produce only a binary real/fake result. That is difficult to trust in a forensic setting because users need to understand why a file was flagged. DeepForensics addresses this by combining multiple detectors and presenting the output with:

- model consensus,
- probability score,
- evidence categories,
- verification guidance,
- visual heatmap-style explanation,
- modular detector expansion.

This makes the project suitable as an applied research system rather than only a model demo.

## Suggested Final-Year Report Positioning

You can describe the project as:

> Explainable Forensic Framework for Deepfake Detection through Visual–Semantic Consistency Analysis is a research-backed system implemented as DeepForensics. It integrates multiple detectors into a Dockerized full-stack application, uses NPR and CLIP-based methods for image forensics, EfficientNet and Vision Transformer-based analysis for video forensics, and ensemble fusion to produce a final verdict. The frontend improves interpretability by presenting evidence summaries, model support, confidence scores, and Grad-CAM-style attention maps.

## What Is Original in This Project

The originality is not inventing a new neural network from scratch. The contribution is in system integration and explainability:

- combining multiple published detectors into one application,
- creating a common API gateway and response format,
- adding ensemble fusion across model outputs,
- building a professional forensic dashboard,
- generating user-facing evidence explanations,
- supporting future detector modules through a shared SDK pattern.

## Recommended Citation List

1. Tan, C., Liu, H., Zhao, Y., Wei, S., Gu, G., Liu, P., and Wei, Y. **Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection.** arXiv:2312.10461.
2. Ojha, U., Li, Y., and Lee, Y. J. **Towards Universal Fake Image Detectors that Generalize Across Generative Models.** arXiv:2302.10174.
3. Coccomini, D., Messina, N., Gennaro, C., and Falchi, F. **Combining EfficientNet and Vision Transformers for Video Deepfake Detection.** arXiv:2107.02612.
4. Radford, A., Kim, J. W., Hallacy, C., et al. **Learning Transferable Visual Models From Natural Language Supervision.** arXiv:2103.00020.
5. Sun, C., Jia, S., Hou, S., and Lyu, S. **AI-Synthesized Voice Detection Using Neural Vocoder Artifacts.** CVPR Workshops, 2023.

## Important Note

The README previously referenced NPR using `arXiv:2310.14036`, but that identifier points to an unrelated paper. The correct NPR paper for the integrated NPR detector is:

https://arxiv.org/abs/2312.10461
