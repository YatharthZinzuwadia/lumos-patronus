import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import "./App.css";

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const detectorRef = useRef<HandLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameRef = useRef<number | null>(null);

  const trailRef = useRef<{ x: number; y: number }[]>([]);

  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Camera stopped");

  async function startCamera() {
    try {
      setStatus("Loading MediaPipe model...");

      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
      );

      let detector: HandLandmarker;

      // Try GPU acceleration first, then CPU if unavailable.
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

      setStatus("Opening webcam...");

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
      const canvas = canvasRef.current;

      if (!video || !canvas) {
        throw new Error("Video or canvas element not found.");
      }

      video.srcObject = stream;
      await video.play();

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext("2d");

      if (!ctx) {
        throw new Error("Could not create canvas context.");
      }

      trailRef.current = [];

      let lastVideoTime = -1;
      let hasDetectedHand = false;

      setRunning(true);
      setStatus("Show your hand to the camera.");

      function render() {
        if (
          video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
          video.currentTime !== lastVideoTime
        ) {
          lastVideoTime = video.currentTime;

          const result = detector.detectForVideo(video, performance.now());

          const hand = result.landmarks?.[0];

          if (hand) {
            // Landmark 8 = index fingertip.
            const fingertip = hand[8];

            trailRef.current.push({
              x: fingertip.x * canvas.width,
              y: fingertip.y * canvas.height,
            });

            if (trailRef.current.length > 45) {
              trailRef.current.shift();
            }

            if (!hasDetectedHand) {
              setStatus("Tracking active!");
              hasDetectedHand = true;
            }
          } else {
            trailRef.current.shift();
          }
        }

        // Fade previous frames to create a trailing effect.
        ctx.globalCompositeOperation = "destination-out";
        ctx.fillStyle = "rgba(0, 0, 0, 0.10)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.globalCompositeOperation = "lighter";

        const points = trailRef.current;

        if (points.length > 1) {
          ctx.beginPath();
          ctx.moveTo(points[0].x, points[0].y);

          for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x, points[i].y);
          }

          // Outer glow.
          ctx.lineWidth = 12;
          ctx.lineCap = "round";
          ctx.lineJoin = "round";
          ctx.strokeStyle = "rgba(0, 190, 255, 0.85)";
          ctx.shadowColor = "#00c8ff";
          ctx.shadowBlur = 28;
          ctx.stroke();

          // Bright inner line.
          ctx.shadowBlur = 5;
          ctx.lineWidth = 2.5;
          ctx.strokeStyle = "#ffffff";
          ctx.stroke();

          ctx.shadowBlur = 0;
        }

        // Draw a glowing dot at the latest fingertip position.
        if (points.length > 0) {
          const last = points[points.length - 1];

          ctx.beginPath();
          ctx.arc(last.x, last.y, 7, 0, Math.PI * 2);
          ctx.fillStyle = "#ffffff";
          ctx.shadowColor = "#00c8ff";
          ctx.shadowBlur = 30;
          ctx.fill();
          ctx.shadowBlur = 0;
        }

        ctx.globalCompositeOperation = "source-over";

        frameRef.current = requestAnimationFrame(render);
      }

      render();
    } catch (error) {
      console.error(error);
      setStatus(
        error instanceof Error ? error.message : "Something went wrong.",
      );
    }
  }

  function stopCamera() {
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    detectorRef.current?.close();
    detectorRef.current = null;

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");

    if (canvas && ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    trailRef.current = [];

    setRunning(false);
    setStatus("Camera stopped");
  }

  useEffect(() => {
    return () => {
      if (frameRef.current !== null) {
        cancelAnimationFrame(frameRef.current);
      }

      streamRef.current?.getTracks().forEach((track) => track.stop());
      detectorRef.current?.close();
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
        <canvas ref={canvasRef} />
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
