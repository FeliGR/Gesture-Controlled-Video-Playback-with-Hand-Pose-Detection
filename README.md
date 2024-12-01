# AI Hand Gesture Video Controller

A web application that uses TensorFlow.js and MediaPipe Hands to control video playback through hand gestures.

## Features

### 🎮 Gesture Controls
- **Fist Gesture (✊)**: 
  - Pause and hide video playback
  - Detection threshold: 5 consecutive frames

- **Time Control (👋)**:
  - Open palm facing left
  - Move hand right: Advance video by 5 seconds
  - Move hand left: Rewind video by 5 seconds
  - Movement threshold: 20 pixels

- **Volume Control (👆)**:
  - Requires two hands:
    - Left hand: Open palm
    - Right hand: Pointing gesture
  - Move right hand up/down to adjust volume
  - Volume change step: 0.5
  - Update delay: 100ms

### 🎥 Video Display
- Main video player with NASA Curiosity landing footage
- Real-time webcam feed with hand tracking visualization
- Cyberpunk-themed UI with NASA color scheme

## Technical Details

### Dependencies
html
- TensorFlow.js
- MediaPipe Hands
- Hand Pose Detection model

### Key Components
- **Hand Detection**:
  - Uses MediaPipe Hands model
  - Supports tracking up to 2 hands
  - Full model type for accurate detection

- **Gesture Recognition**:
  - Real-time landmark detection
  - 3D hand pose estimation
  - Angle calculation for finger positions
  - Palm orientation detection

### UI Features
- Responsive design with breakpoints at 1024px and 768px
- NASA-inspired color scheme
- Real-time hand tracking visualization
- Instruction cards for gesture controls
- Status indicator with pulse animation

## Requirements
- Modern web browser with WebGL support
- Webcam access
- JavaScript enabled

## How to Use
1. Allow webcam access when prompted
2. Follow the gesture instructions shown in the cards
3. Use hand gestures to control video playback
4. Monitor real-time hand tracking in the webcam feed
