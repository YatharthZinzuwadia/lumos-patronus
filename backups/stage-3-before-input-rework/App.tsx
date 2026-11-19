import { useEffect, useRef, useState } from "react";

import { FilesetResolver, GestureRecognizer } from "@mediapipe/tasks-vision";

import { GestureStabilizer, type Landmark } from "./hand/gestureDetection";

import {
  createHandVfx,
  type GestureInput,
  type HandPoint,
} from "./vfx/HandVfx";

import "./App.css";

function averageLandmarks(landmarks: Landmark[], indices: number[]): HandPoint {
  let x = 0;
  let y = 0;

  for (const index of indices) {
    x += landmarks[index].x;
    y += landmarks[index].y;
  }

  return {
    x: x / indices.length,
    y: y / indices.length,
  };
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const vfxHostRef = useRef<HTMLDivElement>(null);

  const recognizerRef = useRef<GestureRecognizer | null>(null);

  const vfxRef = useRef<ReturnType<typeof createHandVfx> | null>(null);

  const streamRef = useRef<MediaStream | null>(null);

  const frameRef = useRef<number | null>(null);

  const gestureSmootherRef = useRef(new GestureStabilizer());

  const [running, setRunning] = useState(false);

  const [status, setStatus] = useState("Camera stopped");

  const [gestureLabel, setGestureLabel] = useState("NO HAND");

  async function startCamera() {
    try {
      setStatus("Loading gesture recognition model...");

      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
      );

      let recognizer: GestureRecognizer;

      // Attempt GPU acceleration first.
      try {
        recognizer = await GestureRecognizer.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/models/gesture_recognizer.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.55,
          minTrackingConfidence: 0.55,
          cannedGesturesClassifierOptions: {
            scoreThreshold: 0.35,
            maxResults: 1,
          },
        });
      } catch (gpuError) {
        console.warn(
          "GPU gesture recognition unavailable. Falling back to CPU.",
          gpuError,
        );

        recognizer = await GestureRecognizer.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/models/gesture_recognizer.task",
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.55,
          minTrackingConfidence: 0.55,
          cannedGesturesClassifierOptions: {
            scoreThreshold: 0.35,
            maxResults: 1,
          },
        });
      }

      recognizerRef.current = recognizer;

      setStatus("Opening camera...");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });

      streamRef.current = stream;

      const video = videoRef.current;
      const host = vfxHostRef.current;

      if (!video || !host) {
        throw new Error("Camera or graphics container not found.");
      }

      video.srcObject = stream;

      await video.play();

      // Start the Three.js engine.
      const vfx = createHandVfx(host);

      vfxRef.current = vfx;

      gestureSmootherRef.current.reset();

      let lastVideoTime = -1;
      let lastGestureLabel = "";
      let lastStatus = "";

      setRunning(true);

      setStatus("Camera active. Show your hands.");

      function render() {
        if (
          video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
          video.currentTime !== lastVideoTime
        ) {
          lastVideoTime = video.currentTime;

          const timestamp = performance.now();

          // Detect hands and classify gestures.
          const result = recognizer.recognizeForVideo(video, timestamp);

          // Stabilize the classifications over time.
          const stableHands = gestureSmootherRef.current.update(
            result,
            timestamp,
          );

          // Create the diagnostic label.
          const label =
            stableHands.length === 0
              ? "NO HAND"
              : stableHands.length >= 2
                ? `${stableHands.length} HANDS · ${stableHands
                    .map((hand) => `${hand.side}: ${hand.gesture}`)
                    .join(" / ")}`
                : `${stableHands[0].side} · ${stableHands[0].gesture}`;

          if (label !== lastGestureLabel) {
            lastGestureLabel = label;
            setGestureLabel(label);
          }

          // Convert detected gestures into visual controls.
          const inputs: GestureInput[] = [];

          for (const hand of stableHands) {
            const landmarks = result.landmarks[hand.index];

            if (!landmarks || landmarks.length < 21) {
              continue;
            }

            let point: HandPoint;

            switch (hand.gesture) {
              case "PINCH":
                // Grab position = midpoint between thumb and index.
                point = averageLandmarks(landmarks, [4, 8]);
                break;

              case "OPEN PALM":
              case "FIST":
                // Generate/collapse around the palm center.
                point = averageLandmarks(landmarks, [0, 5, 9, 13, 17]);
                break;

              default:
                // Point and neutral control use the index fingertip.
                point = {
                  x: landmarks[8].x,
                  y: landmarks[8].y,
                };
            }

            inputs.push({
              side: hand.side,
              gesture: hand.gesture,
              point,
            });
          }

          // Send one or two hands to the graphics engine.
          vfx.updateHands(inputs);

          const nextStatus =
            stableHands.length > 0
              ? "Gesture tracking active."
              : "Camera active. Show your hands.";

          if (nextStatus !== lastStatus) {
            lastStatus = nextStatus;
            setStatus(nextStatus);
          }
        }

        frameRef.current = requestAnimationFrame(render);
      }

      render();
    } catch (error) {
      console.error("Startup error:", error);

      stopCamera();

      setStatus(
        error instanceof Error ? error.message : "Unable to start camera.",
      );
    }
  }

  function stopCamera() {
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => {
      track.stop();
    });

    streamRef.current = null;

    recognizerRef.current?.close();
    recognizerRef.current = null;

    vfxRef.current?.dispose();
    vfxRef.current = null;

    gestureSmootherRef.current.reset();

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    setRunning(false);
    setStatus("Camera stopped");
    setGestureLabel("NO HAND");
  }

  useEffect(() => {
    return () => {
      if (frameRef.current !== null) {
        cancelAnimationFrame(frameRef.current);
      }

      streamRef.current?.getTracks().forEach((track) => {
        track.stop();
      });

      recognizerRef.current?.close();
      vfxRef.current?.dispose();

      gestureSmootherRef.current.reset();
    };
  }, []);

  return (
    <main className="app">
      <header>
        <p className="eyebrow">LUMOS / PATRONUS</p>
        <h1>MAGIC IRL</h1>
        <p className="subtitle">Move your index finger and paint with light.</p>
      </header>

      <section className="stage">
        <video ref={videoRef} muted playsInline />

        <div ref={vfxHostRef} className="vfx-layer" />

        <div className="gesture-readout">
          <span>LIVE GESTURE INPUT</span>
          <strong>{gestureLabel}</strong>
        </div>
      </section>

      <section className="controls">
        <p className="status">{status}</p>

        {!running ? (
          <button onClick={startCamera}>Start Camera</button>
        ) : (
          <button onClick={stopCamera}>Stop Camera</button>
        )}
      </section>
    </main>
  );
}

export default App;
