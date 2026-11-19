import * as THREE from "three";
import type { GestureName } from "../hand/gestureDetection";

export type HandPoint = {
  x: number;
  y: number;
};

export type GestureInput = {
  side: string;
  gesture: GestureName;
  point: HandPoint;

  openness: number;
  pinchStrength: number;
  pinchActive: boolean;
};

const WORLD_HEIGHT = 10;
const MAX_TRAIL_POINTS = 55;
const PARTICLE_COUNT = 420;

// function clamp(value: number, min: number, max: number) {
//   return Math.max(min, Math.min(max, value));
// }

function createGlowTexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");

  canvas.width = 128;
  canvas.height = 128;

  const ctx = canvas.getContext("2d");

  if (!ctx) {
    throw new Error("Unable to create glow texture.");
  }

  const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);

  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.15, "rgba(100,240,255,0.9)");
  gradient.addColorStop(0.4, "rgba(0,180,255,0.35)");
  gradient.addColorStop(1, "rgba(0,150,255,0)");

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 128, 128);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;

  return texture;
}

export function createHandVfx(host: HTMLDivElement) {
  const scene = new THREE.Scene();

  const renderer = new THREE.WebGLRenderer({
    alpha: true,
    antialias: true,
    powerPreference: "high-performance",
  });

  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  renderer.setClearColor(0x000000, 0);

  host.appendChild(renderer.domElement);

  const camera = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100);

  camera.position.set(0, 0, 10);

  function resize() {
    const width = Math.max(host.clientWidth, 1);
    const height = Math.max(host.clientHeight, 1);

    const aspect = width / height;

    camera.left = -(WORLD_HEIGHT * aspect) / 2;
    camera.right = (WORLD_HEIGHT * aspect) / 2;
    camera.top = WORLD_HEIGHT / 2;
    camera.bottom = -WORLD_HEIGHT / 2;

    camera.updateProjectionMatrix();

    renderer.setSize(width, height, false);
  }

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(host);

  resize();

  // Main 3D object.
  const handGroup = new THREE.Group();
  scene.add(handGroup);

  const glowTexture = createGlowTexture();

  const glow = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: glowTexture,
      color: 0x00caff,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );

  glow.scale.set(2, 2, 1);
  glow.position.z = -0.3;

  handGroup.add(glow);

  const halo = new THREE.Mesh(
    new THREE.SphereGeometry(0.43, 32, 32),
    new THREE.MeshBasicMaterial({
      color: 0x00bfff,
      transparent: true,
      opacity: 0.13,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );

  handGroup.add(halo);

  const core = new THREE.Mesh(
    new THREE.SphereGeometry(0.13, 24, 24),
    new THREE.MeshBasicMaterial({
      color: 0xcaffff,
    }),
  );

  handGroup.add(core);

  const wireframe = new THREE.Mesh(
    new THREE.IcosahedronGeometry(0.29, 1),
    new THREE.MeshBasicMaterial({
      color: 0x65edff,
      wireframe: true,
      transparent: true,
      opacity: 0.9,
    }),
  );

  handGroup.add(wireframe);

  const ring1 = new THREE.Mesh(
    new THREE.TorusGeometry(0.43, 0.012, 8, 80),
    new THREE.MeshBasicMaterial({
      color: 0x16dfff,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );

  const ring2 = new THREE.Mesh(
    new THREE.TorusGeometry(0.52, 0.009, 8, 80),
    new THREE.MeshBasicMaterial({
      color: 0x8ffaff,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );

  ring1.rotation.x = 0.7;
  ring2.rotation.y = 1.1;

  handGroup.add(ring1, ring2);

  // Particle system.
  const particlePositions = new Float32Array(PARTICLE_COUNT * 3);

  const particleGeometry = new THREE.BufferGeometry();

  particleGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(particlePositions, 3),
  );

  const particleMaterial = new THREE.PointsMaterial({
    color: 0x74ecff,
    size: 4,
    sizeAttenuation: false,
    transparent: true,
    opacity: 0.9,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  const particles = new THREE.Points(particleGeometry, particleMaterial);

  handGroup.add(particles);

  // Fingertip/palm movement trail.
  const trailPositions = new Float32Array(MAX_TRAIL_POINTS * 3);

  const trailGeometry = new THREE.BufferGeometry();

  trailGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(trailPositions, 3).setUsage(
      THREE.DynamicDrawUsage,
    ),
  );

  trailGeometry.setDrawRange(0, 0);

  const trail = new THREE.Line(
    trailGeometry,
    new THREE.LineBasicMaterial({
      color: 0x32dfff,
      transparent: true,
      opacity: 0.7,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );

  trail.frustumCulled = false;
  scene.add(trail);

  const history: THREE.Vector3[] = [];

  let hasHand = false;

  let targetSpread = 1;
  let currentSpread = 1;

  let targetScale = 1;
  let currentScale = 1;

  let targetRotation = 0;

  // Receives stabilized hand gestures from MediaPipe.

  function updateHands(hands: GestureInput[]) {
    if (hands.length === 0) {
      handGroup.visible = false;
      trail.visible = false;

      history.length = 0;
      trailGeometry.setDrawRange(0, 0);

      hasHand = false;

      targetSpread = 1;
      targetScale = 1;
      targetRotation = 0;

      return;
    }

    let point: HandPoint;

    if (hands.length >= 2) {
      // Two-hand interaction.
      // App.tsx already sorts hands by visible screen position.
      const first = hands[0];
      const second = hands[1];

      point = {
        x: (first.point.x + second.point.x) / 2,
        y: (first.point.y + second.point.y) / 2,
      };

      const aspect = host.clientWidth / Math.max(host.clientHeight, 1);

      const dx = (second.point.x - first.point.x) * aspect;
      const dy = second.point.y - first.point.y;

      const separation = Math.hypot(dx, dy);

      const openness = (first.openness + second.openness) / 2;

      const pinchStrength = (first.pinchStrength + second.pinchStrength) / 2;

      // Hand separation is the primary scale control.
      targetSpread = Math.max(
        0.55,
        Math.min(
          2.4,
          0.55 + separation * 1.35 + openness * 0.35 - pinchStrength * 0.12,
        ),
      );

      targetScale = Math.max(0.85, Math.min(1.35, 0.85 + separation * 0.25));

      targetRotation = Math.atan2(dy, dx);
    } else {
      // Single-hand interaction.
      const hand = hands[0];

      point = hand.point;

      // Continuous hand openness controls particle spread.
      targetSpread = 0.12 + hand.openness * 1.85;

      targetScale = 0.65 + hand.openness * 0.42;

      // Pinching pulls the visual structure inward.
      if (hand.pinchActive) {
        targetSpread = Math.min(
          targetSpread,
          0.22 + (1 - hand.pinchStrength) * 0.12,
        );

        targetScale = 0.85;
      }

      targetRotation = 0;
    }

    const width =
      WORLD_HEIGHT * (host.clientWidth / Math.max(host.clientHeight, 1));

    const x = (point.x - 0.5) * width;
    const y = (0.5 - point.y) * WORLD_HEIGHT;

    const targetPosition = new THREE.Vector3(x, y, 0);

    if (!hasHand) {
      handGroup.position.copy(targetPosition);
    } else {
      handGroup.position.lerp(targetPosition, 0.35);
    }

    hasHand = true;
    handGroup.visible = true;

    history.push(new THREE.Vector3(x, y, 0.05));

    if (history.length > MAX_TRAIL_POINTS) {
      history.shift();
    }

    const attribute = trailGeometry.getAttribute(
      "position",
    ) as THREE.BufferAttribute;

    history.forEach((position, index) => {
      attribute.setXYZ(index, position.x, position.y, position.z);
    });

    attribute.needsUpdate = true;

    trailGeometry.setDrawRange(0, history.length);
    trail.visible = true;
  }

  renderer.setAnimationLoop((time) => {
    const t = time * 0.001;

    wireframe.rotation.x = t * 0.5;
    wireframe.rotation.y = t * 0.8;

    ring1.rotation.z = t * 0.8;
    ring2.rotation.x = t * 0.5;
    ring2.rotation.z = -t * 0.4;

    // Smooth transitions rather than instant visual jumps.
    currentSpread += (targetSpread - currentSpread) * 0.12;

    currentScale += (targetScale - currentScale) * 0.12;

    const rotationDifference = Math.atan2(
      Math.sin(targetRotation - handGroup.rotation.z),
      Math.cos(targetRotation - handGroup.rotation.z),
    );

    handGroup.rotation.z += rotationDifference * 0.12;

    const pulse = 1 + Math.sin(t * 2.5) * 0.055;

    handGroup.scale.setScalar(currentScale * pulse);

    glow.scale.setScalar(2 * pulse);

    // Animate particles according to the active gesture.
    const particleAttribute = particleGeometry.getAttribute(
      "position",
    ) as THREE.BufferAttribute;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const angle = i * 2.399 + t * 0.75;

      const baseRadius =
        0.2 + (i % 8) * 0.035 + Math.sin(t * 2 + i * 0.1) * 0.035;

      const radius = baseRadius * currentSpread;

      const px = Math.cos(angle) * radius;
      const py = Math.sin(angle * 1.15) * radius * 0.75;
      const pz = Math.sin(angle * 0.7 + t) * 0.18;

      particleAttribute.setXYZ(i, px, py, pz);
    }

    particleAttribute.needsUpdate = true;

    renderer.render(scene, camera);
  });

  function dispose() {
    renderer.setAnimationLoop(null);

    resizeObserver.disconnect();

    scene.traverse((object) => {
      const item = object as THREE.Object3D & {
        geometry?: THREE.BufferGeometry;
        material?: THREE.Material | THREE.Material[];
      };

      item.geometry?.dispose();

      if (Array.isArray(item.material)) {
        item.material.forEach((material) => {
          material.dispose();
        });
      } else {
        item.material?.dispose();
      }
    });

    glowTexture.dispose();

    renderer.dispose();
    renderer.domElement.remove();
  }

  return {
    updateHands,
    dispose,
  };
}
