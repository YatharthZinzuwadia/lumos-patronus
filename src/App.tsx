import { useEffect, useRef, useState } from "react";

import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

import { HandInputProcessor, type HandControl } from "./hand/gestureDetection";

import {
  createHandVfxStudio,
  type GestureInput,
  type HandPoint,
} from "./vfx/HandVfxStudio";

import "./App.css";

function getVisualPoint(hand: HandControl, twoHands: boolean): HandPoint {
  if (twoHands) {
    return hand.palmCenter;
  }

  if (hand.gesture === "PINCH") {
    return hand.pinchPoint;
  }

  if (hand.gesture === "OPEN PALM" || hand.gesture === "FIST") {
    return hand.palmCenter;
  }

  return hand.indexTip;
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const vfxHostRef = useRef<HTMLDivElement>(null);

  const landmarkerRef = useRef<HandLandmarker | null>(null);

  const vfxRef = useRef<ReturnType<typeof createHandVfxStudio> | null>(null);

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
        console.warn(
          "GPU initialization failed. Falling back to CPU.",
          gpuError,
        );

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
        throw new Error("Camera element or graphics container was not found.");
      }

      video.srcObject = stream;
      await video.play();

      vfxRef.current = createHandVfxStudio(host, video);

      processorRef.current.reset();

      let lastVideoTime = -1;
      let lastGestureLabel = "";
      let lastStatus = "";
      let lastHudUpdate = 0;

      setRunning(true);
      setGestureLabel("NO HAND");
      setStatus("Camera active. Show your hands.");

      function render() {
        if (
          video &&
          video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
          video.currentTime !== lastVideoTime
        ) {
          lastVideoTime = video.currentTime;

          const now = performance.now();

          const result = landmarker.detectForVideo(video, now);

          const detectedHands = processorRef.current.update(
            result.landmarks ?? [],
            result.worldLandmarks ?? undefined,
            result.handednesses,
            now,
          );

          // Sort according to mirrored screen position.
          const sortedHands = [...detectedHands].sort(
            (a, b) => 1 - a.palmCenter.x - (1 - b.palmCenter.x),
          );

          const twoHands = sortedHands.length >= 2;

          // Limit React HUD updates to avoid unnecessary rendering.
          const label =
            sortedHands.length === 0
              ? "NO HAND"
              : sortedHands
                  .map((hand) => {
                    const held = hand.stale ? " · HOLD" : "";

                    return (
                      `ID ${hand.id} ${hand.side}: ${hand.gesture}` +
                      ` [O ${hand.openness.toFixed(2)}` +
                      ` / P ${hand.pinchRatio.toFixed(2)}]` +
                      held
                    );
                  })
                  .join(" | ");

          if (label !== lastGestureLabel && now - lastHudUpdate >= 100) {
            lastGestureLabel = label;
            lastHudUpdate = now;
            setGestureLabel(label);
          }

          const visualHands: GestureInput[] = sortedHands.map((hand) => {
            const currentLandmarks = hand.stale
              ? undefined
              : result.landmarks?.[hand.sourceIndex];

            const fingertips = currentLandmarks
              ? [4, 8, 12, 16, 20].map((index) => ({
                  x: currentLandmarks[index].x,
                  y: currentLandmarks[index].y,
                }))
              : [];

            return {
              id: hand.id,
              side: hand.side,
              gesture: hand.gesture,
              point: getVisualPoint(hand, twoHands),

              openness: hand.openness,
              pinchStrength: hand.pinchStrength,
              pinchActive: hand.pinchActive,

              fingertips,
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
      // eslint-disable-next-line react-hooks/exhaustive-deps
      processorRef.current.reset();
    };
  }, []);

  return (
    <main className="app">
      <header>
        <p className="eyebrow">LUMOS / PATRONUS</p>
        <h1>MAGIC IRL</h1>
        <p className="subtitle">
          Heres a small lil surprise for my cute lil wizard
        </p>
      </header>

      <section className="stage">
        <video ref={videoRef} muted playsInline />

        <div ref={vfxHostRef} className="vfx-layer" />

        {/* <div className="gesture-readout">
          <span>LIVE GESTURE INPUT</span>
          <strong>{gestureLabel}</strong>
        </div> */}
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
