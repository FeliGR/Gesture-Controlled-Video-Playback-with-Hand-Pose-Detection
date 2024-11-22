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
        canvas.width = webcamEl.videoWidth || 300;
        canvas.height = webcamEl.videoHeight || 300;

        let fistFrames = 0;
        const FIST_DETECTION_THRESHOLD = 5;

        while (true) {
            const img = await webcam.capture();
            const hands = await detector.estimateHands(img, { flipHorizontal: true });

            // Clear previous drawings
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
                            }
                        }
                    } else {
                        fistFrames = 0;
                    }
                });
            } else {
                fistFrames = 0;
            }

            img.dispose();
            await tf.nextFrame();
        }
    }

    function drawHandLandmarks(context, landmarks) {
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],     // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],     // Index
            [0, 9], [9, 10], [10, 11], [11, 12], // Middle
            [0, 13], [13, 14], [14, 15], [15, 16], // Ring
            [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
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
        const ab = { x: a.x - b.x, y: a.y - b.y, z: (a.z || 0) - (b.z || 0) };
        const cb = { x: c.x - b.x, y: c.y - b.y, z: (c.z || 0) - (b.z || 0) };
        const dotProduct = ab.x * cb.x + ab.y * cb.y + ab.z * cb.z;
        const magnitudeAB = Math.hypot(ab.x, ab.y, ab.z);
        const magnitudeCB = Math.hypot(cb.x, cb.y, cb.z);
        const angleRad = Math.acos(dotProduct / (magnitudeAB * magnitudeCB));
        return (angleRad * 180) / Math.PI;
    }

    function isFingerCurled(landmarks, tip, pip, mcp) {
        return calculateAngle(landmarks[tip], landmarks[pip], landmarks[mcp]) < 70;
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

    function isFist(hand) {
        const landmarks = hand.keypoints3D || hand.keypoints;
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

        return isThumbCurled(landmarks) && areFingersClose(landmarks);
    }

    initApp();
});