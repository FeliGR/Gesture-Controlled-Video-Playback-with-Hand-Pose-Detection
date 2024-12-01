document.addEventListener("DOMContentLoaded", () => {
    const webcamEl = document.querySelector("#webcam");
    const canvas = document.querySelector("#canvas");
    const context = canvas.getContext("2d");

    async function initApp() {
        try {
            if (typeof tf === "undefined") {
                throw new Error("TensorFlow.js is not loaded");
            }
            await app();
        } catch (error) {
            console.error(error);
        }
    }

    async function app() {
        const model = handPoseDetection.SupportedModels.MediaPipeHands;
        const detectorConfig = {
            runtime: "mediapipe",
            modelType: "full",
            maxHands: 2,
            solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands",
        };

        const detector = await handPoseDetection.createDetector(model, detectorConfig);
        const webcam = await tf.data.webcam(webcamEl);

        // Set canvas dimensions
        canvas.width = webcamEl.videoWidth || 640;
        canvas.height = webcamEl.videoHeight || 480;

        let fistFrames = 0;
        const FIST_DETECTION_THRESHOLD = 5;

        // Variables para el Punto 1
        let previousPositions = [];
        const MAX_BUFFER_SIZE = 5;
        const MOVEMENT_THRESHOLD = 20;

        // New variables for volume control
        let previousYPositions = [];
        let lastVolumeUpdate = 0;
        const VOLUME_UPDATE_DELAY = 100;
        const VOLUME_CHANGE_STEP = 0.5;

        while (true) {
            const img = await webcam.capture();
            const hands = await detector.estimateHands(img, { flipHorizontal: true });
            context.clearRect(0, 0, canvas.width, canvas.height);

            if (hands.length === 2) {
                // Handle two hands case (volume control)
                const leftHand = hands.find(h => h.handedness === 'Left') || hands[0];
                const rightHand = hands.find(h => h.handedness === 'Right') || hands[1];

                drawHandLandmarks(context, leftHand.keypoints);
                drawHandLandmarks(context, rightHand.keypoints);

                if (isHandOpen(leftHand) && isPointingLeft(rightHand.keypoints)) {
                    const rightIndex = rightHand.keypoints[8];
                    previousYPositions.push(rightIndex.y);

                    if (previousYPositions.length > MAX_BUFFER_SIZE) {
                        previousYPositions.shift();
                    }

                    if (previousYPositions.length === MAX_BUFFER_SIZE) {
                        const now = Date.now();
                        if (now - lastVolumeUpdate > VOLUME_UPDATE_DELAY) {
                            const deltaY = previousYPositions[0] - previousYPositions[previousYPositions.length - 1];

                            if (Math.abs(deltaY) > MOVEMENT_THRESHOLD) {
                                const video = document.querySelector("#drone-video");
                                if (video) {
                                    const volumeChange = deltaY > 0 ? VOLUME_CHANGE_STEP : -VOLUME_CHANGE_STEP;
                                    video.volume = Math.max(0, Math.min(1, video.volume + volumeChange));
                                    console.log(`Volume: ${video.volume.toFixed(2)}`);
                                    lastVolumeUpdate = now;
                                    previousYPositions = [];
                                }
                            }
                        }
                    }
                } else {
                    previousYPositions = [];
                }

            } else if (hands.length === 1) {
                // Handle single hand cases (existing functionality)
                const hand = hands[0];
                const landmarks = hand.keypoints;
                drawHandLandmarks(context, landmarks);

                if (isFist(hand)) {
                    fistFrames++;
                    if (fistFrames >= FIST_DETECTION_THRESHOLD) {
                        const video = document.querySelector("#drone-video");
                        if (video) {
                            video.pause();
                            video.style.display = "none";
                        }
                    }
                } else {
                    fistFrames = 0;
                }

                if (isHandOpen(hand) && isPalmFacingLeft(hand)) {
                    const wrist = hand.keypoints[0];
                    previousPositions.push(wrist.x);

                    if (previousPositions.length > MAX_BUFFER_SIZE) {
                        previousPositions.shift();
                    }

                    if (previousPositions.length === MAX_BUFFER_SIZE) {
                        const deltaX = previousPositions[previousPositions.length - 1] - previousPositions[0];

                        if (Math.abs(deltaX) > MOVEMENT_THRESHOLD) {
                            const video = document.querySelector("#drone-video");
                            if (video) {
                                if (deltaX > 0) {
                                    video.currentTime += 5;
                                    console.log("Avanzar video");
                                } else {
                                    video.currentTime -= 5;
                                    console.log("Retroceder video");
                                }
                            }
                            previousPositions = [];
                        }
                    }
                } else {
                    previousPositions = [];
                }
            } else {
                fistFrames = 0;
                previousPositions = [];
                previousYPositions = [];
            }

            img.dispose();
            await tf.nextFrame();
        }
    }

    // Add new function for pointing gesture
    function isPointingLeft(landmarks) {
        // Index extended, others curled
        if (!isFingerExtended(landmarks, 8, 6, 5)) return false;
        
        const otherFingers = [
            { tip: 12, pip: 10, mcp: 9 },  // Middle
            { tip: 16, pip: 14, mcp: 13 }, // Ring
            { tip: 20, pip: 18, mcp: 17 }  // Pinky
        ];
        
        for (const { tip, pip, mcp } of otherFingers) {
            if (!isFingerCurled(landmarks, tip, pip, mcp)) return false;
        }
        
        return isThumbCurled(landmarks);
    }


    function drawHandLandmarks(context, landmarks) {
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],     // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],     // Index
            [5, 9], [9, 10], [10, 11], [11, 12], // Middle
            [9, 13], [13, 14], [14, 15], [15, 16], // Ring
            [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
            [0, 5], [5, 9], [9, 13], [13, 17], [17, 0] // Palm
        ];

        // Draw connections
        context.lineWidth = 2;
        context.strokeStyle = "blue";

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

    function drawCircle(context, cx, cy, radius, color) {
        context.beginPath();
        context.arc(cx, cy, radius, 0, 2 * Math.PI);
        context.fillStyle = color;
        context.fill();
        context.strokeStyle = color;
        context.stroke();
    }

    function calculateAngle(a, b, c) {
        const ab = { x: a.x - b.x, y: a.y - b.y };
        const cb = { x: c.x - b.x, y: c.y - b.y };
        const dotProduct = ab.x * cb.x + ab.y * cb.y;
        const magnitudeAB = Math.hypot(ab.x, ab.y);
        const magnitudeCB = Math.hypot(cb.x, cb.y);
        const angleRad = Math.acos(dotProduct / (magnitudeAB * magnitudeCB));
        return (angleRad * 180) / Math.PI;
    }

    function isFingerCurled(landmarks, tip, pip, mcp) {
        return calculateAngle(landmarks[tip], landmarks[pip], landmarks[mcp]) < 70;
    }

    function isFingerExtended(landmarks, tip, pip, mcp) {
        return calculateAngle(landmarks[tip], landmarks[pip], landmarks[mcp]) > 160;
    }

    function isThumbExtended(landmarks) {
        const tip = landmarks[4];
        const mcp = landmarks[2];
        const wrist = landmarks[0];
        const angle = calculateAngle(tip, mcp, wrist);
        return angle > 150;
    }

    function areFingersExtended(landmarks) {
        const fingers = [
            { tip: 8, pip: 6, mcp: 5 },   // Index
            { tip: 12, pip: 10, mcp: 9 }, // Middle
            { tip: 16, pip: 14, mcp: 13 }, // Ring
            { tip: 20, pip: 18, mcp: 17 }  // Pinky
        ];

        for (const { tip, pip, mcp } of fingers) {
            if (!isFingerExtended(landmarks, tip, pip, mcp)) return false;
        }
        return true;
    }

    function isFist(hand) {
        const landmarks = hand.keypoints;
        if (!landmarks) return false;

        const fingers = [
            { tip: 8, pip: 6, mcp: 5 },   // Index
            { tip: 12, pip: 10, mcp: 9 }, // Middle
            { tip: 16, pip: 14, mcp: 13 }, // Ring
            { tip: 20, pip: 18, mcp: 17 }  // Pinky
        ];

        for (const { tip, pip, mcp } of fingers) {
            if (!isFingerCurled(landmarks, tip, pip, mcp)) return false;
        }

        return isThumbCurled(landmarks) && areFingersClose(landmarks)
    }

    function isThumbCurled(landmarks) {
        const wrist = landmarks[0];
        const mcp = landmarks[2];
        const tip = landmarks[4];

        const vectorWristToMCP = {
            x: mcp.x - wrist.x,
            y: mcp.y - wrist.y,
            z: (mcp.z || 0) - (wrist.z || 0)
        };

        const vectorWristToTip = {
            x: tip.x - wrist.x,
            y: tip.y - wrist.y,
            z: (tip.z || 0) - (wrist.z || 0)
        };

        const dotProduct = vectorWristToMCP.x * vectorWristToTip.x +
            vectorWristToMCP.y * vectorWristToTip.y +
            vectorWristToMCP.z * vectorWristToTip.z;

        const magnitudeMCP = Math.hypot(vectorWristToMCP.x, vectorWristToMCP.y, vectorWristToMCP.z);
        const magnitudeTip = Math.hypot(vectorWristToTip.x, vectorWristToTip.y, vectorWristToTip.z);

        const angleDeg = (Math.acos(dotProduct / (magnitudeMCP * magnitudeTip)) * 180) / Math.PI;
        return angleDeg < 210;
    }

    function areFingersClose(landmarks) {
        const tips = [8, 12, 16, 20];
        let totalDist = 0;
        let pairs = 0;

        for (let i = 0; i < tips.length; i++) {
            for (let j = i + 1; j < tips.length; j++) {
                const dx = landmarks[tips[i]].x - landmarks[tips[j]].x;
                const dy = landmarks[tips[i]].y - landmarks[tips[j]].y;
                totalDist += Math.hypot(dx, dy);
                pairs++;
            }
        }
        return totalDist / pairs < 40;
    }

    function isHandOpen(hand) {
        const landmarks = hand.keypoints;
        if (!landmarks) return false;

        return areFingersExtended(landmarks) && isThumbExtended(landmarks);
    }

    function isPalmFacingLeft(hand) {
        const landmarks = hand.keypoints3D;
        if (!landmarks || landmarks.length === 0) return false;

        const wrist = landmarks[0];
        const indexMCP = landmarks[5];
        const pinkyMCP = landmarks[17];

        const vector1 = {
            x: indexMCP.x - wrist.x,
            y: indexMCP.y - wrist.y,
            z: indexMCP.z - wrist.z
        };

        const vector2 = {
            x: pinkyMCP.x - wrist.x,
            y: pinkyMCP.y - wrist.y,
            z: pinkyMCP.z - wrist.z
        };

        // Calcular el vector normal (producto cruzado de vector1 y vector2)
        const normal = {
            x: vector1.y * vector2.z - vector1.z * vector2.y,
            y: vector1.z * vector2.x - vector1.x * vector2.z,
            z: vector1.x * vector2.y - vector1.y * vector2.x
        };

        // Normalizar el vector normal
        const magnitude = Math.hypot(normal.x, normal.y, normal.z);
        normal.x /= magnitude;
        normal.y /= magnitude;
        normal.z /= magnitude;

        // Verificar si la componente x está cerca de 1 (palma hacia la izquierda)
        return normal.x > 0.5;
    }

    initApp();
});