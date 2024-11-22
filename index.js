document.addEventListener("DOMContentLoaded", () => {
  const webcamEl = document.querySelector("#webcam");
  const canvas = document.querySelector("#canvas");
  const outputMessageEl = document.querySelector("#output-message");

  // Initialize the application
  (async function initApp() {
    try {
      if (typeof tf === "undefined") {
        throw new Error("TensorFlow.js is not loaded");
      }
      await app();
    } catch (error) {
      console.error(error);
      if (outputMessageEl) outputMessageEl.innerText = error.message;
    }
  })();

  // Main application function
  async function app() {
    // Configure the hand pose detection model
    const model = handPoseDetection.SupportedModels.MediaPipeHands;
    const detectorConfig = {
      runtime: "mediapipe",
      modelType: "full",
      maxHands: 2,
      solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands",
    };
    const detector = await handPoseDetection.createDetector(model, detectorConfig);

    // Initialize the webcam feed
    const webcam = await tf.data.webcam(webcamEl, {
      resizeWidth: 252,
      resizeHeight: 252,
    });

    // Adjust canvas position to overlay the webcam video
    const { x, y } = webcamEl.getBoundingClientRect();
    Object.assign(canvas.style, { top: y + "px", left: x + "px" });
    const context = canvas.getContext("2d");
    canvas.width = webcamEl.width;
    canvas.height = webcamEl.height;

    let fistFrames = 0;
    const FIST_DETECTION_THRESHOLD = 5; // Threshold of consecutive frames

    // Start the detection loop
    while (true) {
      const img = await webcam.capture();
      const hands = await detector.estimateHands(img, { flipHorizontal: true });
      context.clearRect(0, 0, canvas.width, canvas.height);

      if (hands.length > 0) {
        hands.forEach((hand) => {
          const landmarks = hand.keypoints;
          drawHandLandmarks(context, landmarks);

          if (isFist(hand)) {
            fistFrames++;
            if (fistFrames >= FIST_DETECTION_THRESHOLD) {
              const video = document.querySelector("#drone-video");
              if (video) {
                video.pause();
                video.style.display = "none";
                console.log("Fist detected: video paused and hidden.");
              } else {
                console.error("Video element not found.");
              }
            }
          } else {
            fistFrames = 0; // Reset counter if fist is not detected
          }
        });
      } else {
        fistFrames = 0; // Reset if no hands are detected
      }

      img.dispose();
      await tf.nextFrame();
    }
  }

  /**
   * Draws hand landmarks and connections on the canvas.
   * @param {CanvasRenderingContext2D} context - The drawing context.
   * @param {Array} landmarks - The hand landmarks to draw.
   */
  function drawHandLandmarks(context, landmarks) {
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],     // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8],     // Index finger
      [0, 9], [9, 10], [10, 11], [11, 12],// Middle finger
      [0, 13], [13, 14], [14, 15], [15, 16],// Ring finger
      [0, 17], [17, 18], [18, 19], [19, 20],// Pinky finger
    ];
    context.lineWidth = 2;
    context.strokeStyle = "blue";

    // Draw connections
    connections.forEach(([start, end]) => {
      const startPoint = landmarks[start];
      const endPoint = landmarks[end];
      context.beginPath();
      context.moveTo(startPoint.x, startPoint.y);
      context.lineTo(endPoint.x, endPoint.y);
      context.stroke();
    });

    // Draw landmarks
    landmarks.forEach(({ x, y }) => {
      drawCircle(context, x, y, 3, "red");
    });
  }

  /**
   * Draws a circle on the canvas.
   * @param {CanvasRenderingContext2D} context - The drawing context.
   * @param {number} cx - The x-coordinate of the center.
   * @param {number} cy - The y-coordinate of the center.
   * @param {number} radius - The radius of the circle.
   * @param {string} color - The color of the circle.
   */
  function drawCircle(context, cx, cy, radius, color) {
    context.beginPath();
    context.arc(cx, cy, radius, 0, 2 * Math.PI);
    context.fillStyle = color;
    context.fill();
    context.strokeStyle = color;
    context.stroke();
  }

  /**
   * Calculates the angle between three points.
   * @param {Object} a - The first point.
   * @param {Object} b - The vertex point.
   * @param {Object} c - The third point.
   * @returns {number} - The angle in degrees.
   */
  function calculateAngle(a, b, c) {
    const ab = { x: a.x - b.x, y: a.y - b.y, z: (a.z || 0) - (b.z || 0) };
    const cb = { x: c.x - b.x, y: c.y - b.y, z: (c.z || 0) - (b.z || 0) };
    const dotProduct = ab.x * cb.x + ab.y * cb.y + ab.z * cb.z;
    const magnitudeAB = Math.hypot(ab.x, ab.y, ab.z);
    const magnitudeCB = Math.hypot(cb.x, cb.y, cb.z);
    const angleRad = Math.acos(dotProduct / (magnitudeAB * magnitudeCB));
    return (angleRad * 180) / Math.PI;
  }

  /**
   * Determines if a finger is curled.
   * @param {Array} landmarks - The hand landmarks.
   * @param {number} tip - Index of the fingertip landmark.
   * @param {number} pip - Index of the PIP joint landmark.
   * @param {number} mcp - Index of the MCP joint landmark.
   * @returns {boolean} - True if the finger is curled.
   */
  function isFingerCurled(landmarks, tip, pip, mcp) {
    return calculateAngle(landmarks[tip], landmarks[pip], landmarks[mcp]) < 65;
  }

  /**
   * Determines if the thumb is curled.
   * @param {Array} landmarks - The hand landmarks.
   * @returns {boolean} - True if the thumb is curled.
   */
  function isThumbCurled(landmarks) {
    const wrist = landmarks[0];
    const mcp = landmarks[2];
    const tip = landmarks[4];
    const vectorWristToMCP = {
      x: mcp.x - wrist.x,
      y: mcp.y - wrist.y,
      z: (mcp.z || 0) - (wrist.z || 0),
    };
    const vectorWristToTip = {
      x: tip.x - wrist.x,
      y: tip.y - wrist.y,
      z: (tip.z || 0) - (wrist.z || 0),
    };
    const dotProduct =
      vectorWristToMCP.x * vectorWristToTip.x +
      vectorWristToMCP.y * vectorWristToTip.y +
      vectorWristToMCP.z * vectorWristToTip.z;
    const magnitudeMCP = Math.hypot(
      vectorWristToMCP.x,
      vectorWristToMCP.y,
      vectorWristToMCP.z
    );
    const magnitudeTip = Math.hypot(
      vectorWristToTip.x,
      vectorWristToTip.y,
      vectorWristToTip.z
    );
    const angleDeg =
      (Math.acos(dotProduct / (magnitudeMCP * magnitudeTip)) * 180) / Math.PI;
    return angleDeg < 200; // Adjust if necessary
  }

  /**
   * Checks if the fingers are close together.
   * @param {Array} landmarks - The hand landmarks.
   * @returns {boolean} - True if fingers are close together.
   */
  function areFingersClose(landmarks) {
    const tips = [8, 12, 16, 20];
    let totalDist = 0;
    let pairs = 0;

    for (let i = 0; i < tips.length; i++) {
      for (let j = i + 1; j < tips.length; j++) {
        const dx = landmarks[tips[i]].x - landmarks[tips[j]].x;
        const dy = landmarks[tips[i]].y - landmarks[tips[j]].y;
        const dz = (landmarks[tips[i]].z || 0) - (landmarks[tips[j]].z || 0);
        totalDist += Math.hypot(dx, dy, dz);
        pairs++;
      }
    }
    return totalDist / pairs < 30; // Adjust if necessary
  }

  /**
   * Determines if a fist gesture is detected.
   * @param {Object} hand - The hand object from hand pose detection.
   * @returns {boolean} - True if a fist is detected.
   */
  function isFist(hand) {
    const landmarks = hand.keypoints3D || hand.keypoints;
    if (!landmarks) return false;

    const fingers = [
      { tip: 8, pip: 6, mcp: 5 },   // Index finger
      { tip: 12, pip: 10, mcp: 9 }, // Middle finger
      { tip: 16, pip: 14, mcp: 13 },// Ring finger
      { tip: 20, pip: 18, mcp: 17 },// Pinky finger
    ];

    for (const { tip, pip, mcp } of fingers) {
      if (!isFingerCurled(landmarks, tip, pip, mcp)) return false;
    }

    return isThumbCurled(landmarks) && areFingersClose(landmarks);
  }
});
