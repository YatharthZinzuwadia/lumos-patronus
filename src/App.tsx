import { useEffect, useRef, useState } from "react";

import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

import { HandInputProcessor } from "./hand/gestureDetection";

import { createHandVfx } from "./vfx/HandVfx";

import "./App.css";

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const vfxHostRef = useRef<HTMLDivElement>(null);

  const landmarkerRef = useRef<HandLandmarker | null>(null);

  const vfxRef = useRef<ReturnType<typeof createHandVfx> | null>(null);

  const processorRef = useRef(new HandInputProcessor());

  const streamRef = useRef<MediaStream | null>(null);

  const frameRef = useRef<number | null>(null);

  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Camera stopped");
  const [gestureLabel, setGestureLabel] = useState("NO HAND");

  async function startCamera() {
    try {
      setStatus("Loading hand tracking...");

      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
      );

      let landmarker: HandLandmarker;

      // GPU first, CPU fallback.
      try {
        landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/models/hand_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.55,
          minTrackingConfidence: 0.55,
        });
      } catch (gpuError) {
        console.warn("GPU unavailable. Using CPU.", gpuError);

        landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/models/hand_landmarker.task",
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.55,
          minTrackingConfidence: 0.55,
        });
      }

      landmarkerRef.current = landmarker;

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

      // Preserve the existing 3D engine.
      vfxRef.current = createHandVfx(host);

      processorRef.current.reset();

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

          const now = performance.now();

          const result = landmarker.detectForVideo(video, now);

          const detectedHands = processorRef.current.update(
            result.landmarks,
            result.worldLandmarks,
            result.handednesses,
            now,
          );

          /**
           * Sort by visible screen position.
           *
           * The camera and VFX layers are mirrored with CSS,
           * but the image passed into MediaPipe is not.
           *
           * Therefore, visible screen X = 1 - landmark X.
           */
          const sortedHands = [...detectedHands].sort(
            (a, b) => 1 - a.palmCenter.x - (1 - b.palmCenter.x),
          );

          // Diagnostic display with useful calibration values.
          const label =
            sortedHands.length === 0
              ? "NO HAND"
              : sortedHands
                  .map((hand) => {
                    const held = hand.stale ? " · HOLD" : "";

                    return (
                      `${hand.side}: ${hand.gesture}` +
                      ` [O ${hand.openness.toFixed(1)}` +
                      ` / P ${hand.pinchRatio.toFixed(2)}]` +
                      held
                    );
                  })
                  .join(" | ");

          if (label !== lastGestureLabel) {
            lastGestureLabel = label;
            setGestureLabel(label);
          }

          /**
           * Single hand:
           *  - Pinch follows thumb/index midpoint.
           *  - Open/Fist follows palm center.
           *  - Point/Neutral follows index fingertip.
           *
           * Two hands:
           *  - Always follow palm centers.
           *  - Prevents gesture-dependent position jumps.
           */
          const twoHands = sortedHands.length >= 2;

          const visualHands = sortedHands.map((hand) => {
            let point = hand.indexTip;

            if (twoHands) {
              point = hand.palmCenter;
            } else if (hand.gesture === "PINCH") {
              point = hand.pinchPoint;
            } else if (
              hand.gesture === "OPEN PALM" ||
              hand.gesture === "FIST"
            ) {
              point = hand.palmCenter;
            }

            return {
              side: hand.side,
              gesture: hand.gesture,
              point,
              openness: hand.openness,
              pinchStrength: hand.pinchStrength,
              pinchActive: hand.pinchActive,
            };
          });

          vfxRef.current?.updateHands(visualHands);

          const nextStatus = detectedHands.some((hand) => !hand.stale)
            ? "Hand tracking active."
            : detectedHands.length > 0
              ? "Holding recent hand position."
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

    landmarkerRef.current?.close();
    landmarkerRef.current = null;

    vfxRef.current?.dispose();
    vfxRef.current = null;

    processorRef.current.reset();

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

      landmarkerRef.current?.close();
      vfxRef.current?.dispose();

      processorRef.current.reset();
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
