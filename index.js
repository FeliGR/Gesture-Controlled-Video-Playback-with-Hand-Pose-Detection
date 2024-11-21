document.addEventListener("DOMContentLoaded", function() {
  const webcamEl = document.querySelector("#webcam");
  const canvas = document.querySelector("#canvas");
  const outputMessageEl = document.querySelector("#output-message");

  function initTFJS() {
    if (typeof tf === "undefined") {
      throw new Error("TensorFlow.js not loaded");
    }
  }

  async function app() {
    const model = handPoseDetection.SupportedModels.MediaPipeHands;
    const detectorConfig = {
      runtime: 'mediapipe',
      modelType: 'full',
      maxHands: 2,
      solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands'
    };
    const detector = await handPoseDetection.createDetector(model, detectorConfig);

    const webcam = await tf.data.webcam(webcamEl, {
      resizeWidth: 252,
      resizeHeight: 252
    });

    const camerabbox = webcamEl.getBoundingClientRect();
    canvas.style.top = camerabbox.y + "px";
    canvas.style.left = camerabbox.x + "px";

    const context = canvas.getContext("2d");

    while (true) {
      const img = await webcam.capture();
      const hands = await detector.estimateHands(img, { flipHorizontal: true });

      context.clearRect(0, 0, canvas.width, canvas.height);

      if (hands.length > 0) {
        hands.forEach(hand => {
          const landmarks = hand.keypoints;
          drawHandLandmarks(context, landmarks);

          if (isFist(hand)) {
            const video = document.querySelector("#drone-video");
            if (video) {
              video.pause();
              video.style.display = "none";
            } else {
              console.error("Video element not found");
            }
          }
        });
      }

      img.dispose();
      await tf.nextFrame();
    }
  }

  function drawHandLandmarks(context, landmarks) {
    const color = 'red';
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],       // Pulgar
      [0, 5], [5, 6], [6, 7], [7, 8],       // Índice
      [0, 9], [9, 10], [10, 11], [11, 12],  // Medio
      [0, 13], [13, 14], [14, 15], [15, 16], // Anular
      [0, 17], [17, 18], [18, 19], [19, 20]  // Meñique
    ];

    connections.forEach(([start, end]) => {
      const startPoint = landmarks[start];
      const endPoint = landmarks[end];

      context.beginPath();
      context.moveTo(startPoint.x, startPoint.y);
      context.lineTo(endPoint.x, endPoint.y);
      context.lineWidth = 2;
      context.strokeStyle = 'blue';
      context.stroke();
    });

    landmarks.forEach((landmark) => {
      drawCircle(context, landmark.x, landmark.y, 3, color);
    });
  }

  function drawCircle(context, cx, cy, radius, color) {
    context.beginPath();
    context.arc(cx, cy, radius, 0, 2 * Math.PI, false);
    context.fillStyle = "red";
    context.fill();
    context.lineWidth = 1;
    context.strokeStyle = color;
    context.stroke();
  }

  function isFingerCurled(landmarks, tipIndex, pipIndex, mcpIndex) {
    const tipY = landmarks[tipIndex].y;
    const pipY = landmarks[pipIndex].y;
    const mcpY = landmarks[mcpIndex].y;
    return tipY > pipY && pipY > mcpY;
  }

  function isFist(hand) {
    const landmarks = hand.keypoints;

    const fingers = [
      { tip: 8, pip: 6, mcp: 5 },    // Índice
      { tip: 12, pip: 10, mcp: 9 },  // Medio
      { tip: 16, pip: 14, mcp: 13 }, // Anular
      { tip: 20, pip: 18, mcp: 17 }  // Meñique
    ];

    for (let i = 0; i < fingers.length; i++) {
      const { tip, pip, mcp } = fingers[i];
      if (!isFingerCurled(landmarks, tip, pip, mcp)) {
        return false;
      }
    }
    
    return hand.handedness === 'Right';
  }

  (async function initApp() {
    try {
      initTFJS();
      await app();
    } catch (error) {
      console.error(error);
      if (outputMessageEl) {
        outputMessageEl.innerText = error.message;
      }
    }
  }());
});
