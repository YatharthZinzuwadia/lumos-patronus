import * as THREE from "three";

import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

import type { GestureName } from "../hand/gestureDetection";

export type HandPoint = {
  x: number;
  y: number;
};

export type GestureInput = {
  id: number;
  side: string;
  gesture: GestureName;
  point: HandPoint;
  openness: number;
  pinchStrength: number;
  pinchActive: boolean;
  fingertips: HandPoint[];
};

const WORLD_HEIGHT = 10;

const MAX_HANDS = 2;
const FINGERTIPS_PER_HAND = 5;
const MAX_TRAILS = MAX_HANDS * FINGERTIPS_PER_HAND;

const TRAIL_LENGTH = 28;
const MAX_PETALS = 650;

const BLOOM_LAYER = 1;

const TIP_COLORS = [0x7df9ff, 0x9be7ff, 0xc6a7ff, 0xffa7db, 0xffffff];

const PETAL_COLORS = ["#8be9ff", "#b5d8ff", "#d5b7ff", "#ffb9df", "#ffffff"];

const clamp = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value));

function markForBloom(object: THREE.Object3D) {
  // Retain the normal rendering layer and add the bloom layer.
  object.layers.enable(BLOOM_LAYER);
}

function createGlowTexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");

  canvas.width = 128;
  canvas.height = 128;

  const ctx = canvas.getContext("2d");

  if (!ctx) {
    throw new Error("Could not create glow texture.");
  }

  const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);

  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.15, "rgba(125,249,255,0.9)");
  gradient.addColorStop(0.4, "rgba(0,160,255,0.25)");
  gradient.addColorStop(1, "rgba(0,100,255,0)");

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 128, 128);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;

  return texture;
}

function createPetalGeometry(): THREE.BufferGeometry {
  const geometry = new THREE.PlaneGeometry(1, 1.35, 6, 8);

  const position = geometry.attributes.position;

  for (let i = 0; i < position.count; i++) {
    const x = position.getX(i);
    const y = position.getY(i);

    const v = (y + 0.675) / 1.35;

    const width = Math.sin(Math.pow(v, 0.68) * Math.PI);

    const newX = x * width;
    const newY = (v - 0.15) * 1.25;
    const newZ = Math.sin(v * Math.PI) * 0.18 - x * x * 0.25;

    position.setXYZ(i, newX, newY, newZ);
  }

  geometry.computeVertexNormals();

  return geometry;
}

const BACKGROUND_VERTEX_SHADER = `
  varying vec2 vUv;

  void main() {
    vUv = uv;

    gl_Position =
      projectionMatrix *
      modelViewMatrix *
      vec4(position, 1.0);
  }
`;

const BACKGROUND_FRAGMENT_SHADER = `
  uniform sampler2D uVideo;
  uniform float uTime;
  uniform float uDistortStrength;

  uniform float uViewAspect;
  uniform vec2 uVideoScale;

  uniform vec2 uPoints[10];
  uniform vec2 uVelocities[10];
  uniform float uActive[10];

  varying vec2 vUv;

  float hash(vec2 p) {
    return fract(
      sin(dot(p, vec2(127.1, 311.7))) * 43758.5453
    );
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(
      mix(
        hash(i),
        hash(i + vec2(1.0, 0.0)),
        u.x
      ),
      mix(
        hash(i + vec2(0.0, 1.0)),
        hash(i + vec2(1.0, 1.0)),
        u.x
      ),
      u.y
    );
  }

  void main() {
    vec2 totalDisplacement = vec2(0.0);
    float aura = 0.0;

    for (int i = 0; i < 10; i++) {
      if (uActive[i] > 0.5) {
        vec2 difference = vUv - uPoints[i];

        difference.x *= uViewAspect;

        float distanceToFinger = length(difference);

        float falloff = exp(
          -distanceToFinger * distanceToFinger / 0.018
        );

        vec2 direction = normalize(
          difference + vec2(0.00001)
        );

        float ripple = sin(
          distanceToFinger * 135.0 -
          uTime * 6.0 +
          noise(vUv * 12.0) * 2.0
        );

        vec2 velocity = uVelocities[i];
        velocity.x *= uViewAspect;

        totalDisplacement += (
          direction * ripple * 0.0025 -
          velocity * 0.0008
        ) * falloff * uDistortStrength;

        aura += falloff * 0.12;
      }
    }

    // Center-crop the video without changing its exposure.
    vec2 sourceUv =
      (vUv - 0.5) * uVideoScale + 0.5;

    vec2 warpedUv = clamp(
      sourceUv + totalDisplacement,
      vec2(0.002),
      vec2(0.998)
    );

    // Very subtle chromatic separation around the fingertips.
    float split = min(aura * 0.0015, 0.003);

    float red = texture2D(
      uVideo,
      clamp(warpedUv + vec2(split, 0.0), 0.002, 0.998)
    ).r;

    float green = texture2D(
      uVideo,
      warpedUv
    ).g;

    float blue = texture2D(
      uVideo,
      clamp(warpedUv - vec2(split, 0.0), 0.002, 0.998)
    ).b;

    vec3 color = vec3(red, green, blue);

    // Reduced from the previous shader to avoid brightening
    // the actual camera footage around the hands.
    color += vec3(0.004, 0.012, 0.018) * min(aura, 1.0);

    gl_FragColor = vec4(color, 1.0);
  }
`;

const COMPOSITE_VERTEX_SHADER = `
  varying vec2 vUv;

  void main() {
    vUv = uv;

    gl_Position =
      projectionMatrix *
      modelViewMatrix *
      vec4(position, 1.0);
  }
`;

const COMPOSITE_FRAGMENT_SHADER = `
  uniform sampler2D tDiffuse;
  uniform sampler2D bloomTexture;

  varying vec2 vUv;

  void main() {
    vec4 base = texture2D(tDiffuse, vUv);
    vec3 bloom = texture2D(bloomTexture, vUv).rgb;

    // Screen blend adds glow without altering the camera image.
    vec3 combined =
      1.0 - (1.0 - base.rgb) * (1.0 - bloom * 0.55);

    gl_FragColor = vec4(combined, base.a);
  }
`;

export function createHandVfxStudio(
  host: HTMLDivElement,
  video: HTMLVideoElement,
) {
  const scene = new THREE.Scene();

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: "high-performance",
  });

  // IMPORTANT:
  // Do not apply ACES exposure or global tone mapping to the webcam.
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));

  host.appendChild(renderer.domElement);

  const camera = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100);

  camera.position.set(0, 0, 10);

  // ---------------------------------------------------------
  // SELECTIVE BLOOM PIPELINE
  // ---------------------------------------------------------

  const bloomLayer = new THREE.Layers();
  bloomLayer.set(BLOOM_LAYER);

  // Store visibility so camera/background objects can be
  // temporarily excluded from the bloom-only render.
  const hiddenObjects = new Map<string, boolean>();

  function darkenNonBloomed(object: THREE.Object3D) {
    const candidate = object as THREE.Object3D & {
      material?: THREE.Material | THREE.Material[];
    };

    if (candidate.material && !bloomLayer.test(object.layers)) {
      hiddenObjects.set(object.uuid, object.visible);
      object.visible = false;
    }
  }

  function restoreVisibility(object: THREE.Object3D) {
    if (hiddenObjects.has(object.uuid)) {
      object.visible = hiddenObjects.get(object.uuid)!;
      hiddenObjects.delete(object.uuid);
    }
  }

  // Pipeline A: only bloom-enabled objects.
  const bloomComposer = new EffectComposer(renderer);

  bloomComposer.renderToScreen = false;

  bloomComposer.addPass(new RenderPass(scene, camera));

  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(1, 1),
    0.45, // Reduced glow strength
    0.4, // Smaller bloom radius
    0.9, // Only very bright VFX pixels bloom
  );

  bloomComposer.addPass(bloomPass);

  // Pipeline B: complete original scene plus selected bloom.
  const finalComposer = new EffectComposer(renderer);

  finalComposer.addPass(new RenderPass(scene, camera));

  const compositeMaterial = new THREE.ShaderMaterial({
    uniforms: {
      tDiffuse: { value: null },
      bloomTexture: { value: null },
    },
    vertexShader: COMPOSITE_VERTEX_SHADER,
    fragmentShader: COMPOSITE_FRAGMENT_SHADER,
    toneMapped: false,
  });

  const compositePass = new ShaderPass(compositeMaterial, "tDiffuse");

  finalComposer.addPass(compositePass);
  finalComposer.addPass(new OutputPass());

  // ---------------------------------------------------------
  // VIDEO BACKGROUND + GLSL DISTORTION
  // ---------------------------------------------------------

  const videoTexture = new THREE.VideoTexture(video);

  videoTexture.colorSpace = THREE.SRGBColorSpace;
  videoTexture.minFilter = THREE.LinearFilter;
  videoTexture.magFilter = THREE.LinearFilter;

  const videoUniforms = {
    uVideo: { value: videoTexture },
    uTime: { value: 0 },
    uDistortStrength: { value: 1.0 },
    uViewAspect: {
      value: host.clientWidth / Math.max(host.clientHeight, 1),
    },
    uVideoScale: {
      value: new THREE.Vector2(1, 1),
    },
    uPoints: {
      value: Array.from(
        { length: MAX_TRAILS },
        () => new THREE.Vector2(0.5, 0.5),
      ),
    },
    uVelocities: {
      value: Array.from({ length: MAX_TRAILS }, () => new THREE.Vector2(0, 0)),
    },
    uActive: {
      value: new Float32Array(MAX_TRAILS),
    },
  };

  const backgroundMaterial = new THREE.ShaderMaterial({
    uniforms: videoUniforms,
    vertexShader: BACKGROUND_VERTEX_SHADER,
    fragmentShader: BACKGROUND_FRAGMENT_SHADER,
    depthWrite: false,
    depthTest: false,
    toneMapped: false,
  });

  const background = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    backgroundMaterial,
  );

  // The camera background is intentionally NOT bloom-enabled.
  background.position.z = -4;
  scene.add(background);

  // ---------------------------------------------------------
  // RESPONSIVE WORLD SIZE
  // ---------------------------------------------------------

  let worldWidth =
    WORLD_HEIGHT * (host.clientWidth / Math.max(host.clientHeight, 1));

  function resize() {
    const width = Math.max(host.clientWidth, 1);
    const height = Math.max(host.clientHeight, 1);

    const aspect = width / height;

    worldWidth = WORLD_HEIGHT * aspect;

    camera.left = -worldWidth / 2;
    camera.right = worldWidth / 2;
    camera.top = WORLD_HEIGHT / 2;
    camera.bottom = -WORLD_HEIGHT / 2;

    camera.updateProjectionMatrix();

    background.scale.set(worldWidth, WORLD_HEIGHT, 1);

    renderer.setSize(width, height, false);

    bloomComposer.setSize(width, height);
    finalComposer.setSize(width, height);

    bloomPass.resolution.set(width, height);

    videoUniforms.uViewAspect.value = aspect;

    const videoAspect = video.videoWidth / Math.max(video.videoHeight, 1);

    const scale = new THREE.Vector2(1, 1);

    if (videoAspect > aspect) {
      scale.x = aspect / videoAspect;
    } else {
      scale.y = videoAspect / aspect;
    }

    videoUniforms.uVideoScale.value.copy(scale);
  }

  // ---------------------------------------------------------
  // MAIN ENERGY OBJECT
  // ---------------------------------------------------------

  const energyGroup = new THREE.Group();

  scene.add(energyGroup);

  const glowTexture = createGlowTexture();

  const glowMaterial = new THREE.SpriteMaterial({
    map: glowTexture,
    color: new THREE.Color(0x58eaff).multiplyScalar(2.0),
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });

  const glow = new THREE.Sprite(glowMaterial);

  glow.scale.set(2.2, 2.2, 1);

  markForBloom(glow);
  energyGroup.add(glow);

  const coreMaterial = new THREE.MeshBasicMaterial({
    color: new THREE.Color(0xe7ffff).multiplyScalar(2.0),
    toneMapped: false,
  });

  const core = new THREE.Mesh(
    new THREE.SphereGeometry(0.14, 32, 32),
    coreMaterial,
  );

  markForBloom(core);
  energyGroup.add(core);

  const wireframeMaterial = new THREE.MeshBasicMaterial({
    color: new THREE.Color(0x6cecff).multiplyScalar(1.8),
    wireframe: true,
    transparent: true,
    opacity: 0.95,
    toneMapped: false,
  });

  const wireframe = new THREE.Mesh(
    new THREE.IcosahedronGeometry(0.36, 1),
    wireframeMaterial,
  );

  markForBloom(wireframe);
  energyGroup.add(wireframe);

  const ring1 = new THREE.Mesh(
    new THREE.TorusGeometry(0.55, 0.012, 8, 96),
    new THREE.MeshBasicMaterial({
      color: new THREE.Color(0x35dbff).multiplyScalar(1.8),
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      toneMapped: false,
    }),
  );

  const ring2 = new THREE.Mesh(
    new THREE.TorusGeometry(0.7, 0.009, 8, 96),
    new THREE.MeshBasicMaterial({
      color: new THREE.Color(0xb5f5ff).multiplyScalar(1.7),
      transparent: true,
      opacity: 0.65,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      toneMapped: false,
    }),
  );

  ring1.rotation.x = 0.7;
  ring2.rotation.y = 1.1;

  markForBloom(ring1);
  markForBloom(ring2);

  energyGroup.add(ring1, ring2);

  // Orbital particle cloud.
  const ORB_PARTICLES = 220;

  const orbPositions = new Float32Array(ORB_PARTICLES * 3);

  for (let i = 0; i < ORB_PARTICLES; i++) {
    const angle = i * 2.399;
    const radius = 0.45 + Math.random() * 0.42;

    orbPositions[i * 3] = Math.cos(angle) * radius;

    orbPositions[i * 3 + 1] = Math.sin(angle * 1.15) * radius;

    orbPositions[i * 3 + 2] = (Math.random() - 0.5) * 0.5;
  }

  const orbGeometry = new THREE.BufferGeometry();

  orbGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(orbPositions, 3),
  );

  const orbMaterial = new THREE.PointsMaterial({
    color: new THREE.Color(0x91f5ff).multiplyScalar(2.0),
    size: 2.5,
    sizeAttenuation: false,
    transparent: true,
    opacity: 0.9,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });

  const orbParticles = new THREE.Points(orbGeometry, orbMaterial);

  markForBloom(orbParticles);
  energyGroup.add(orbParticles);

  // ---------------------------------------------------------
  // TEN INDEPENDENT FINGERTIP TRAILS
  // ---------------------------------------------------------

  type Trail = {
    line: THREE.Line;
    history: THREE.Vector3[];
    active: boolean;
  };

  const trails: Trail[] = [];

  for (let i = 0; i < MAX_TRAILS; i++) {
    const geometry = new THREE.BufferGeometry();

    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(TRAIL_LENGTH * 3), 3).setUsage(
        THREE.DynamicDrawUsage,
      ),
    );

    geometry.setDrawRange(0, 0);

    const material = new THREE.LineBasicMaterial({
      color: new THREE.Color(
        TIP_COLORS[i % FINGERTIPS_PER_HAND],
      ).multiplyScalar(1.8),
      transparent: true,
      opacity: 0.88,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      toneMapped: false,
    });

    const line = new THREE.Line(geometry, material);

    line.frustumCulled = false;
    line.visible = false;

    markForBloom(line);
    scene.add(line);

    trails.push({
      line,
      history: [],
      active: false,
    });
  }

  const lastUv: (THREE.Vector2 | null)[] = Array(MAX_TRAILS).fill(null);

  const lastSampleTime = new Float64Array(MAX_TRAILS);
  const lastSpawnTime = new Float64Array(MAX_TRAILS);

  // Keep each detected hand connected to the same five trail slots.
  const handSlots = new Map<number, number>();

  function getHandSlot(hand: GestureInput): number {
    const existing = handSlots.get(hand.id);

    if (existing !== undefined) {
      return existing;
    }

    const preferred = hand.side === "LEFT" ? 0 : hand.side === "RIGHT" ? 1 : -1;

    const occupied = new Set(handSlots.values());

    let slot = preferred >= 0 && !occupied.has(preferred) ? preferred : -1;

    if (slot < 0) {
      slot = !occupied.has(0) ? 0 : 1;
    }

    handSlots.set(hand.id, slot);

    return slot;
  }

  // ---------------------------------------------------------
  // INSTANCED PETALS
  // ---------------------------------------------------------

  const petalGeometry = createPetalGeometry();

  const petalMaterial = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.94,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });

  const petalMesh = new THREE.InstancedMesh(
    petalGeometry,
    petalMaterial,
    MAX_PETALS,
  );

  petalMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

  petalMesh.frustumCulled = false;

  markForBloom(petalMesh);
  scene.add(petalMesh);

  const petalPositions = new Float32Array(MAX_PETALS * 3);

  const petalVelocities = new Float32Array(MAX_PETALS * 3);

  const petalRotations = new Float32Array(MAX_PETALS * 3);

  const petalRotationSpeeds = new Float32Array(MAX_PETALS * 3);

  const petalLifetimes = new Float32Array(MAX_PETALS);
  const petalScales = new Float32Array(MAX_PETALS);

  const dummy = new THREE.Object3D();
  const tempColor = new THREE.Color();

  for (let i = 0; i < MAX_PETALS; i++) {
    dummy.position.set(0, 0, -100);
    dummy.scale.set(0, 0, 0);
    dummy.updateMatrix();

    petalMesh.setMatrixAt(i, dummy.matrix);
    petalMesh.setColorAt(i, tempColor.set("#ffffff"));
  }

  petalMesh.instanceMatrix.needsUpdate = true;

  if (petalMesh.instanceColor) {
    petalMesh.instanceColor.needsUpdate = true;
  }

  let petalCursor = 0;
  let petalColorsDirty = false;

  function spawnPetal(
    x: number,
    y: number,
    vx: number,
    vy: number,
    burst = false,
  ) {
    const index = petalCursor;
    const i3 = index * 3;

    const angle = Math.random() * Math.PI * 2;
    const force = burst ? 1.25 : 0.55;

    petalPositions[i3] = x;
    petalPositions[i3 + 1] = y;
    petalPositions[i3 + 2] = 0.2 + Math.random() * 0.3;

    petalVelocities[i3] = vx * 0.1 + Math.cos(angle) * force;

    petalVelocities[i3 + 1] = vy * 0.1 + Math.sin(angle) * force;

    petalVelocities[i3 + 2] = (Math.random() - 0.5) * 0.15;

    petalRotations[i3] = Math.random() * Math.PI * 2;
    petalRotations[i3 + 1] = Math.random() * Math.PI * 2;
    petalRotations[i3 + 2] = Math.random() * Math.PI * 2;

    petalRotationSpeeds[i3] = (Math.random() - 0.5) * 3;

    petalRotationSpeeds[i3 + 1] = (Math.random() - 0.5) * 3;

    petalRotationSpeeds[i3 + 2] = (Math.random() - 0.5) * 2;

    petalLifetimes[index] = 1;

    petalScales[index] = (burst ? 0.17 : 0.11) * (0.65 + Math.random() * 0.8);

    const color = PETAL_COLORS[Math.floor(Math.random() * PETAL_COLORS.length)];

    petalMesh.setColorAt(index, tempColor.set(color).multiplyScalar(2.0));

    petalColorsDirty = true;

    petalCursor = (petalCursor + 1) % MAX_PETALS;
  }

  // ---------------------------------------------------------
  // INPUT STATE
  // ---------------------------------------------------------

  let targetX = 0;
  let targetY = 0;

  let targetSpread = 1;
  let currentSpread = 1;

  let targetScale = 1;
  let currentScale = 1;

  let targetRotation = 0;

  let previousTime = performance.now();

  function toWorld(point: HandPoint): THREE.Vector3 {
    return new THREE.Vector3(
      (point.x - 0.5) * worldWidth,
      (0.5 - point.y) * WORLD_HEIGHT,
      0,
    );
  }

  function updateHands(hands: GestureInput[]) {
    const now = performance.now();

    videoUniforms.uActive.value.fill(0);

    trails.forEach((trail) => {
      trail.active = false;
    });

    const activeIds = new Set(hands.map((hand) => hand.id));

    for (const id of handSlots.keys()) {
      if (!activeIds.has(id)) {
        handSlots.delete(id);
      }
    }

    if (hands.length === 0) {
      energyGroup.visible = false;

      targetSpread = 1;
      targetScale = 1;
      targetRotation = 0;

      return;
    }

    const activeHands = hands.slice(0, MAX_HANDS);

    let center: HandPoint;

    if (activeHands.length === 2) {
      const first = activeHands[0];
      const second = activeHands[1];

      center = {
        x: (first.point.x + second.point.x) / 2,
        y: (first.point.y + second.point.y) / 2,
      };

      const dx =
        (second.point.x - first.point.x) *
        (host.clientWidth / Math.max(host.clientHeight, 1));

      const dy = second.point.y - first.point.y;

      const separation = Math.hypot(dx, dy);

      const averageOpenness = (first.openness + second.openness) / 2;

      targetSpread = clamp(
        0.55 + separation * 1.35 + averageOpenness * 0.35,
        0.55,
        2.4,
      );

      targetScale = clamp(0.85 + separation * 0.25, 0.85, 1.35);

      targetRotation = Math.atan2(dy, dx);
    } else {
      const hand = activeHands[0];

      center = hand.point;

      targetSpread = 0.12 + hand.openness * 1.85;
      targetScale = 0.65 + hand.openness * 0.42;

      if (hand.pinchActive) {
        targetSpread = Math.min(
          targetSpread,
          0.22 + (1 - hand.pinchStrength) * 0.12,
        );

        targetScale = 0.85;
      }

      targetRotation = 0;
    }

    targetX = (center.x - 0.5) * worldWidth;
    targetY = (0.5 - center.y) * WORLD_HEIGHT;

    // Update each fingertip independently.
    activeHands.forEach((hand) => {
      const handSlot = getHandSlot(hand);

      hand.fingertips
        .slice(0, FINGERTIPS_PER_HAND)
        .forEach((finger, fingerIndex) => {
          const slot = handSlot * FINGERTIPS_PER_HAND + fingerIndex;

          const world = toWorld(finger);

          const uv = new THREE.Vector2(finger.x, 1 - finger.y);

          const previous = lastUv[slot];

          const dt = lastSampleTime[slot]
            ? Math.max(0.001, (now - lastSampleTime[slot]) / 1000)
            : 1 / 60;

          let vx = 0;
          let vy = 0;

          if (previous) {
            vx = (uv.x - previous.x) / dt;
            vy = (uv.y - previous.y) / dt;
          }

          lastUv[slot] = uv.clone();
          lastSampleTime[slot] = now;

          videoUniforms.uPoints.value[slot].copy(uv);

          videoUniforms.uVelocities.value[slot].set(
            clamp(vx, -2, 2),
            clamp(vy, -2, 2),
          );

          videoUniforms.uActive.value[slot] = 1;

          const trail = trails[slot];

          trail.active = true;

          trail.history.push(new THREE.Vector3(world.x, world.y, 0.35));

          if (trail.history.length > TRAIL_LENGTH) {
            trail.history.shift();
          }

          const speed = Math.hypot(vx * worldWidth, vy * WORLD_HEIGHT);

          const spawnInterval = hand.pinchActive ? 55 : 90;

          if (
            (speed > 0.5 || hand.pinchActive) &&
            now - lastSpawnTime[slot] > spawnInterval
          ) {
            spawnPetal(
              world.x,
              world.y,
              vx * worldWidth,
              -vy * WORLD_HEIGHT,
              hand.pinchActive,
            );

            lastSpawnTime[slot] = now;
          }
        });
    });

    energyGroup.visible = true;
  }

  // ---------------------------------------------------------
  // RESPONSIVE LAYOUT
  // ---------------------------------------------------------

  const resizeObserver = new ResizeObserver(resize);

  resizeObserver.observe(host);

  video.addEventListener("loadedmetadata", resize);

  resize();

  // ---------------------------------------------------------
  // ANIMATION + SELECTIVE BLOOM COMPOSITING
  // ---------------------------------------------------------

  renderer.setAnimationLoop(() => {
    const now = performance.now();

    const dt = Math.min((now - previousTime) / 1000, 0.05);

    previousTime = now;

    const time = now * 0.001;

    videoUniforms.uTime.value = time;

    const smoothing = 1 - Math.exp(-dt * 9);

    energyGroup.position.x += (targetX - energyGroup.position.x) * smoothing;

    energyGroup.position.y += (targetY - energyGroup.position.y) * smoothing;

    currentSpread += (targetSpread - currentSpread) * smoothing;

    currentScale += (targetScale - currentScale) * smoothing;

    const rotationDifference = Math.atan2(
      Math.sin(targetRotation - energyGroup.rotation.z),
      Math.cos(targetRotation - energyGroup.rotation.z),
    );

    energyGroup.rotation.z += rotationDifference * smoothing;

    const pulse = 1 + Math.sin(time * 2.4) * 0.045;

    energyGroup.scale.setScalar(currentScale * pulse);
    glow.scale.setScalar(2.2 * pulse);

    wireframe.rotation.x = time * 0.45;
    wireframe.rotation.y = time * 0.7;

    ring1.rotation.z = time * 0.8;
    ring2.rotation.x = time * 0.55;
    ring2.rotation.z = -time * 0.4;

    orbParticles.scale.setScalar(currentSpread);
    orbParticles.rotation.y = time * 0.25;

    // Update the ten fingertip trails.
    trails.forEach((trail) => {
      if (!trail.active && trail.history.length > 0) {
        trail.history.shift();
      }

      const position = trail.line.geometry.getAttribute(
        "position",
      ) as THREE.BufferAttribute;

      trail.history.forEach((point, index) => {
        position.setXYZ(index, point.x, point.y, point.z);
      });

      position.needsUpdate = true;

      trail.line.geometry.setDrawRange(0, trail.history.length);

      trail.line.visible = trail.history.length >= 2;

      const material = trail.line.material as THREE.LineBasicMaterial;

      material.opacity = trail.active ? 0.9 : 0.55;
    });

    // Animate and recycle instanced petals.
    let hasActivePetals = false;

    for (let i = 0; i < MAX_PETALS; i++) {
      if (petalLifetimes[i] <= 0) continue;

      const i3 = i * 3;

      petalLifetimes[i] -= dt * 0.48;

      if (petalLifetimes[i] <= 0) {
        dummy.position.set(0, 0, -100);
        dummy.scale.set(0, 0, 0);
        dummy.updateMatrix();

        petalMesh.setMatrixAt(i, dummy.matrix);

        hasActivePetals = true;
        continue;
      }

      hasActivePetals = true;

      petalVelocities[i3] *= Math.pow(0.985, dt * 60);
      petalVelocities[i3 + 1] *= Math.pow(0.985, dt * 60);

      petalVelocities[i3 + 1] -= 0.8 * dt;

      petalPositions[i3] += petalVelocities[i3] * dt;
      petalPositions[i3 + 1] += petalVelocities[i3 + 1] * dt;
      petalPositions[i3 + 2] += petalVelocities[i3 + 2] * dt;

      petalRotations[i3] += petalRotationSpeeds[i3] * dt;
      petalRotations[i3 + 1] += petalRotationSpeeds[i3 + 1] * dt;
      petalRotations[i3 + 2] += petalRotationSpeeds[i3 + 2] * dt;

      const life = clamp(petalLifetimes[i], 0, 1);

      const fadeIn = clamp((1 - life) * 8, 0, 1);
      const fadeOut = clamp(life * 3, 0, 1);

      const scale = petalScales[i] * fadeIn * fadeOut;

      dummy.position.set(
        petalPositions[i3],
        petalPositions[i3 + 1],
        petalPositions[i3 + 2],
      );

      dummy.rotation.set(
        petalRotations[i3],
        petalRotations[i3 + 1],
        petalRotations[i3 + 2],
      );

      dummy.scale.set(scale, scale, scale);

      dummy.updateMatrix();

      petalMesh.setMatrixAt(i, dummy.matrix);
    }

    if (hasActivePetals) {
      petalMesh.instanceMatrix.needsUpdate = true;
    }

    if (petalColorsDirty && petalMesh.instanceColor) {
      petalMesh.instanceColor.needsUpdate = true;
      petalColorsDirty = false;
    }

    // PASS 1: Render only the VFX objects for bloom.
    // The natural camera background is temporarily hidden.
    scene.traverse(darkenNonBloomed);

    bloomComposer.render();

    // Restore the original scene before rendering the final image.
    scene.traverse(restoreVisibility);

    // Transfer the VFX-only bloom texture into the final compositor.
    compositeMaterial.uniforms.bloomTexture.value =
      bloomComposer.readBuffer.texture;

    // PASS 2: Render the natural camera image and composite VFX glow.
    finalComposer.render();
  });

  // ---------------------------------------------------------
  // CLEANUP
  // ---------------------------------------------------------

  function dispose() {
    renderer.setAnimationLoop(null);

    resizeObserver.disconnect();

    video.removeEventListener("loadedmetadata", resize);

    bloomComposer.dispose();
    finalComposer.dispose();

    scene.traverse((object) => {
      const item = object as THREE.Object3D & {
        geometry?: THREE.BufferGeometry;
        material?: THREE.Material | THREE.Material[];
      };

      item.geometry?.dispose();

      if (Array.isArray(item.material)) {
        item.material.forEach((material) => material.dispose());
      } else {
        item.material?.dispose();
      }
    });

    videoTexture.dispose();
    glowTexture.dispose();

    backgroundMaterial.dispose();
    compositeMaterial.dispose();

    renderer.dispose();

    renderer.domElement.remove();
  }

  return {
    updateHands,
    dispose,
  };
}
