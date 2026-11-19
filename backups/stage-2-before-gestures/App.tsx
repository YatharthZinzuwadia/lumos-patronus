import { useEffect, useRef, useState } from "react";

import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

import { createHandVfx } from "./vfx/HandVfx";
import { detectGestures } from "./hand/gestureDetection";

import "./App.css";

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const vfxHostRef = useRef<HTMLDivElement>(null);

  const detectorRef = useRef<HandLandmarker | null>(null);
  const vfxRef = useRef<ReturnType<typeof createHandVfx> | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const frameRef = useRef<number | null>(null);

  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Camera stopped");
  const [gestureLabel, setGestureLabel] = useState("NO HAND");

  async function startCamera() {
    try {
      setStatus("Loading hand-tracking model...");

      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
      );

      let detector: HandLandmarker;

      // GPU acceleration with CPU fallback.
      try {
        detector = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/models/hand_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
        });
      } catch {
        detector = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/models/hand_landmarker.task",
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
        });
      }

      detectorRef.current = detector;

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
        throw new Error("Camera or graphics element not found.");
      }

      video.srcObject = stream;
      await video.play();

      const vfx = createHandVfx(host);
      vfxRef.current = vfx;

      let lastVideoTime = -1;
      let lastGestureLabel = "";
      let hasDetectedHand = false;

      setRunning(true);
      setStatus("Camera active. Show your hands.");

      function render() {
        if (
          video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
          video.currentTime !== lastVideoTime
        ) {
          lastVideoTime = video.currentTime;

          const result = detector.detectForVideo(video, performance.now());

          // Recognize every detected hand.
          const detected = detectGestures(
            result.landmarks,
            result.worldLandmarks,
            result.handednesses,
          );

          // Update React only when the recognized gesture changes.
          if (detected.label !== lastGestureLabel) {
            lastGestureLabel = detected.label;
            setGestureLabel(detected.label);
          }

          // For now, the existing 3D effect follows the first hand.
          // We'll connect both hands to visual interactions next.
          const hand = result.landmarks[0];

          if (hand) {
            vfx.updateHand({
              x: hand[8].x,
              y: hand[8].y,
            });

            if (!hasDetectedHand) {
              setStatus("Hand tracking active!");
              hasDetectedHand = true;
            }
          } else {
            vfx.updateHand(null);

            if (hasDetectedHand) {
              setStatus("Camera active. Show your hands.");
              hasDetectedHand = false;
            }
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

    detectorRef.current?.close();
    detectorRef.current = null;

    vfxRef.current?.dispose();
    vfxRef.current = null;

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

      detectorRef.current?.close();
      vfxRef.current?.dispose();
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
