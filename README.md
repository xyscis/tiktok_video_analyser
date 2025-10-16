# TikTok Personality Analyzer 🎭

A deep learning system that analyzes TikTok videos using Convolutional Neural Networks (CNN) to predict Big Five personality traits from visual content.

## Overview

This tool downloads TikTok videos, extracts frames, and uses a ResNet50-based CNN to predict personality traits across five dimensions:
- **Openness** - Creativity and openness to new experiences
- **Conscientiousness** - Organization and goal-orientation
- **Extraversion** - Social energy and outgoingness
- **Agreeableness** - Cooperation and empathy
- **Neuroticism** - Emotional sensitivity and stability

## Features

✅ Download videos from TikTok URLs or user profiles  
✅ Random video sampling for unbiased analysis  
✅ Automatic face detection and frame extraction  
✅ ResNet50 backbone with attention mechanism  
✅ Multi-head architecture for personality traits  
✅ Aggregate predictions across multiple videos  
✅ Human-readable personality interpretations  

## Installation

### Requirements
- Python 3.7+
- CUDA-compatible GPU (recommended)

### Install Dependencies

```bash
pip install torch torchvision opencv-python pillow numpy yt-dlp requests
```

## Quick Start

### 1. Basic Usage

```python
from tiktok_personality_analyzer import TikTokPersonalityAnalyzer

# Initialize analyzer
analyzer = TikTokPersonalityAnalyzer(
    model_path='best_personality_model.pth',  # Optional
    device='cuda'
)

# Analyze a user from 5 random videos
results = analyzer.analyze_user(
    username='username',
    max_videos=5,
    random_selection=True
)

# Display results
analyzer.print_results(results)
```

### 2. Run from Command Line

```bash
python tiktok_personality_analyzer.py
```

Then enter a TikTok username when prompted.

## Usage Examples

### Analyze Single Video

```python
# Analyze from URL
results = analyzer.analyze_video_url('https://www.tiktok.com/@user/video/123456789')
```

### Analyze User Profile

```python
# Analyze 10 random videos
results = analyzer.analyze_user(
    username='username',
    max_videos=10,
    random_selection=True
)
```

### Custom Frame Extraction

```python
from tiktok_personality_analyzer import VideoProcessor

processor = VideoProcessor()

# Extract frames using different methods
frames_uniform = processor.extract_frames('video.mp4', num_frames=16, method='uniform')
frames_keyframes = processor.extract_frames('video.mp4', num_frames=16, method='keyframes')
```

## Architecture

### Model Components

1. **Backbone**: ResNet50 (pretrained on ImageNet)
2. **Attention Mechanism**: Learns important features
3. **Feature Extractor**: Multi-layer perceptron with dropout
4. **Trait Heads**: Separate output layers for each personality trait

### Pipeline

```
Download Video → Extract Frames → Face Detection → 
Preprocessing → CNN Inference → Aggregate Results → Interpretation
```

## Output Format

```json
{
  "openness": 75.3,
  "conscientiousness": 62.1,
  "extraversion": 81.5,
  "agreeableness": 68.9,
  "neuroticism": 45.2,
  "interpretation": {
    "openness": "Highly creative, curious, and open to new experiences",
    "extraversion": "Highly social, energetic, and outgoing",
    "dominant_trait": "Extraversion"
  },
  "videos_analyzed": 5,
  "random_selection": true,
  "timestamp": "2025-10-16 14:30:22"
}
```

## Configuration

### VideoProcessor Options

```python
processor = VideoProcessor(target_size=(224, 224))
```

### Model Parameters

```python
model = PersonalityCNN(
    num_traits=5,
    pretrained=True
)
```

## Training Your Own Model

To train on your own dataset:

1. Prepare labeled video data with personality scores
2. Extract frames and preprocess
3. Fine-tune the model:

```python
model = PersonalityCNN(pretrained=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    for frames, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

## Limitations

⚠️ **Important Disclaimers:**
- Personality prediction from video is inherently uncertain
- Model requires fine-tuning on labeled personality data
- Results should not be used for high-stakes decisions
- Face detection may fail with poor lighting or angles
- Accuracy depends on video quality and content

## Troubleshooting

### Common Issues

**Video download fails:**
```bash
# Update yt-dlp
pip install --upgrade yt-dlp
```

**CUDA out of memory:**
```python
# Use CPU instead
analyzer = TikTokPersonalityAnalyzer(device='cpu')
```

**No faces detected:**
- Ensure videos contain clear face shots
- Try disabling face detection: `focus_on_faces=False`

## Dependencies

- **torch**: Deep learning framework
- **torchvision**: Pre-trained models and transforms
- **opencv-python**: Video processing and face detection
- **yt-dlp**: TikTok video downloading
- **PIL**: Image processing
- **numpy**: Numerical operations

## License

This project is for educational and research purposes only. Respect TikTok's Terms of Service and user privacy when downloading and analyzing content.

## Contributing

Contributions welcome! Areas for improvement:
- Better personality prediction models
- Multi-modal analysis (audio, text, visual)
- Temporal modeling (LSTM/Transformer)
- Larger training datasets
- Cross-platform support

## Citation

If you use this code in your research, please cite:

```bibtex
@software{tiktok_personality_analyzer,
  title={TikTok Personality Analyzer},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/tiktok-personality-analyzer}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Disclaimer**: This tool is for research and educational purposes only. Always obtain proper consent before analyzing individuals' content.