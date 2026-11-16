# Vision Robustness Lab

A lightweight computer-vision experimentation project for measuring how image perturbations change model inputs and prediction confidence.

## Focus

- reproducible image perturbations
- blur, noise, brightness, and compression transforms
- experiment configuration
- image-quality statistics
- robustness-oriented evaluation

## Example

```python
from transforms import gaussian_noise, adjust_brightness

noisy = gaussian_noise(image, sigma=12, seed=42)
darker = adjust_brightness(image, factor=0.7)
```

The project is model-agnostic by design, so the transforms can be used with CNNs, vision transformers, or multimodal models.

## Why robustness matters

Computer-vision models often see inputs that differ from clean training data. Compression, lighting changes, blur, and sensor noise can all change model behavior. This project provides reusable tools for creating controlled perturbations and measuring their effects.
