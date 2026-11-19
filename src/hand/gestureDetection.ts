export type GestureName = "PINCH" | "OPEN PALM" | "FIST" | "POINT" | "NEUTRAL";

export type Landmark = {
  x: number;
  y: number;
  z?: number;
};

export type Point2D = {
  x: number;
  y: number;
};

type HandednessCategory = {
  categoryName?: string;
  score?: number;
};

type HandDetection = {
  image: Landmark[];
  shape: Landmark[];
  side: string;
  palm: Point2D;
};

export type HandControl = {
  id: number;
  sourceIndex: number;
  side: string;
  gesture: GestureName;

  palmCenter: Point2D;
  indexTip: Point2D;
  pinchPoint: Point2D;

  openness: number;
  pinchRatio: number;
  pinchStrength: number;
  pinchActive: boolean;

  stale: boolean;
};

type HandTrack = {
  id: number;
  sourceIndex: number;
  side: string;

  lastSeen: number;
  lastUpdate: number;

  palmCenter: Point2D;
  indexTip: Point2D;
  pinchPoint: Point2D;

  openness: number;
  pinchRatio: number;

  pinchActive: boolean;
  pinchCandidate: boolean | null;
  pinchCandidateSince: number;

  gesture: GestureName;
  gestureCandidate: GestureName;
  gestureCandidateSince: number;
};

const clamp = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value));

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

function lerpPoint(a: Point2D, b: Point2D, t: number): Point2D {
  return {
    x: lerp(a.x, b.x, t),
    y: lerp(a.y, b.y, t),
  };
}

function averagePoint(landmarks: Landmark[], indices: number[]): Point2D {
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

function distance3D(a: Landmark, b: Landmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0));
}

function distance2D(a: Point2D, b: Point2D): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function jointAngle(a: Landmark, b: Landmark, c: Landmark): number {
  const ba = {
    x: a.x - b.x,
    y: a.y - b.y,
    z: (a.z ?? 0) - (b.z ?? 0),
  };

  const bc = {
    x: c.x - b.x,
    y: c.y - b.y,
    z: (c.z ?? 0) - (b.z ?? 0),
  };

  const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;

  const lengthA = Math.hypot(ba.x, ba.y, ba.z);
  const lengthB = Math.hypot(bc.x, bc.y, bc.z);

  if (lengthA * lengthB < 0.000001) return 0;

  return Math.acos(clamp(dot / (lengthA * lengthB), -1, 1)) * (180 / Math.PI);
}

/**
 * Scale-independent openness using 3D hand landmarks.
 *
 * Palm reference: wrist (0) to middle MCP (9).
 * Five fingertip distances are normalized against that reference.
 */
function calculateOpenness(hand: Landmark[]): number {
  const palmLength = distance3D(hand[0], hand[9]) || 0.0001;

  const tips = [4, 8, 12, 16, 20];

  let total = 0;

  for (const tip of tips) {
    total += distance3D(hand[tip], hand[9]);
  }

  const ratio = total / tips.length / palmLength;

  return clamp((ratio - 0.4) / 0.55, 0, 1);
}

/**
 * Normalized thumb-index distance.
 * Smaller values indicate a tighter pinch.
 */
function calculatePinchRatio(hand: Landmark[]): number {
  const palmLength = distance3D(hand[0], hand[9]) || 0.0001;

  return distance3D(hand[4], hand[8]) / palmLength;
}

function isPointGesture(hand: Landmark[]): boolean {
  const index = jointAngle(hand[5], hand[6], hand[8]);
  const middle = jointAngle(hand[9], hand[10], hand[12]);
  const ring = jointAngle(hand[13], hand[14], hand[16]);
  const pinky = jointAngle(hand[17], hand[18], hand[20]);

  return index > 150 && middle < 145 && ring < 145 && pinky < 145;
}

export class HandInputProcessor {
  private tracks = new Map<number, HandTrack>();
  private nextId = 1;

  private readonly lostGraceMs = 180;

  reset() {
    this.tracks.clear();
    this.nextId = 1;
  }

  update(
    imageLandmarks: Landmark[][],
    worldLandmarks: Landmark[][] | undefined,
    handednesses: HandednessCategory[][] | undefined,
    now: number,
  ): HandControl[] {
    // Remove expired tracks, but retain brief detection dropouts.
    for (const [id, track] of this.tracks) {
      if (now - track.lastSeen > this.lostGraceMs) {
        this.tracks.delete(id);
      }
    }

    const detections: HandDetection[] = [];

    imageLandmarks.forEach((image, index) => {
      if (image.length < 21) return;

      const world = worldLandmarks?.[index];

      const shape = world && world.length >= 21 ? world : image;

      const category = handednesses?.[index]?.[0];

      const side = category?.categoryName?.toUpperCase() ?? "UNKNOWN";

      detections.push({
        image,
        shape,
        side,
        palm: averagePoint(image, [0, 5, 9, 13, 17]),
      });
    });

    // Associate current detections with previously tracked hands.
    const pairs: {
      detectionIndex: number;
      trackId: number;
      cost: number;
    }[] = [];

    const activeTracks = [...this.tracks.values()];

    detections.forEach((detection, detectionIndex) => {
      for (const track of activeTracks) {
        const positionDistance = distance2D(detection.palm, track.palmCenter);

        const sidePenalty =
          detection.side !== "UNKNOWN" &&
          track.side !== "UNKNOWN" &&
          detection.side !== track.side
            ? 0.1
            : 0;

        pairs.push({
          detectionIndex,
          trackId: track.id,
          cost: positionDistance + sidePenalty,
        });
      }
    });

    pairs.sort((a, b) => a.cost - b.cost);

    const assignedDetections = new Set<number>();
    const assignedTracks = new Set<number>();
    const assignments = new Map<number, HandTrack>();

    for (const pair of pairs) {
      if (
        pair.cost > 0.45 ||
        assignedDetections.has(pair.detectionIndex) ||
        assignedTracks.has(pair.trackId)
      ) {
        continue;
      }

      const track = this.tracks.get(pair.trackId);

      if (!track) continue;

      assignments.set(pair.detectionIndex, track);
      assignedDetections.add(pair.detectionIndex);
      assignedTracks.add(pair.trackId);
    }

    // Create new tracks or update existing ones.
    detections.forEach((detection, index) => {
      let track = assignments.get(index);

      const openness = calculateOpenness(detection.shape);
      const pinchRatio = calculatePinchRatio(detection.shape);

      const pinchPoint = averagePoint(detection.image, [4, 8]);

      const indexTip = {
        x: detection.image[8].x,
        y: detection.image[8].y,
      };

      if (!track) {
        track = {
          id: this.nextId++,
          sourceIndex: index,
          side: detection.side,

          lastSeen: now,
          lastUpdate: now,

          palmCenter: detection.palm,
          indexTip,
          pinchPoint,

          openness,
          pinchRatio,

          pinchActive: false,
          pinchCandidate: null,
          pinchCandidateSince: now,

          gesture: "NEUTRAL",
          gestureCandidate: "NEUTRAL",
          gestureCandidateSince: now,
        };

        this.tracks.set(track.id, track);
      } else {
        this.updateTrack(
          track,
          index,
          detection,
          openness,
          pinchRatio,
          indexTip,
          pinchPoint,
          now,
        );
      }
    });

    return [...this.tracks.values()]
      .filter((track) => now - track.lastSeen <= this.lostGraceMs)
      .map((track) => ({
        id: track.id,
        sourceIndex: track.sourceIndex,
        side: track.side,
        gesture: track.gesture,

        palmCenter: track.palmCenter,
        indexTip: track.indexTip,
        pinchPoint: track.pinchPoint,

        openness: track.openness,
        pinchRatio: track.pinchRatio,

        pinchStrength: clamp(1 - (track.pinchRatio - 0.15) / 0.5, 0, 1),

        pinchActive: track.pinchActive,

        stale: now - track.lastSeen > 0,
      }));
  }

  private updateTrack(
    track: HandTrack,
    sourceIndex: number,
    detection: HandDetection,
    rawOpenness: number,
    rawPinchRatio: number,
    indexTip: Point2D,
    pinchPoint: Point2D,
    now: number,
  ) {
    const dt = clamp(now - track.lastUpdate, 0, 100);

    // Delta-time-based smoothing.
    const positionAlpha = 1 - Math.exp(-dt / 45);
    const metricAlpha = 1 - Math.exp(-dt / 80);

    track.palmCenter = lerpPoint(
      track.palmCenter,
      detection.palm,
      positionAlpha,
    );

    track.indexTip = lerpPoint(track.indexTip, indexTip, positionAlpha);

    track.pinchPoint = lerpPoint(track.pinchPoint, pinchPoint, positionAlpha);

    track.openness = lerp(track.openness, rawOpenness, metricAlpha);

    track.pinchRatio = lerp(track.pinchRatio, rawPinchRatio, metricAlpha);

    track.sourceIndex = sourceIndex;
    track.side = detection.side;
    track.lastSeen = now;
    track.lastUpdate = now;

    this.updatePinch(track, now);

    const isPoint = isPointGesture(detection.shape);

    let proposed: GestureName = "NEUTRAL";

    if (track.pinchActive) {
      proposed = "PINCH";
    } else if (isPoint) {
      proposed = "POINT";
    } else {
      const openThreshold = track.gesture === "OPEN PALM" ? 0.56 : 0.68;

      const fistThreshold = track.gesture === "FIST" ? 0.43 : 0.3;

      if (track.openness > openThreshold) {
        proposed = "OPEN PALM";
      } else if (track.openness < fistThreshold) {
        proposed = "FIST";
      }
    }

    this.updateGesture(track, proposed, now);
  }

  private updatePinch(track: HandTrack, now: number) {
    // TODO(LUMOS-INPUT): Revisit pinch transitions.
    // Current hysteresis works, but activation and release still
    // need calibration against different hands and fast movements.
    const desired = track.pinchActive
      ? track.pinchRatio < 0.36
      : track.pinchRatio < 0.23;

    if (desired === track.pinchActive) {
      track.pinchCandidate = null;
      return;
    }

    if (track.pinchCandidate !== desired) {
      track.pinchCandidate = desired;
      track.pinchCandidateSince = now;
      return;
    }

    if (now - track.pinchCandidateSince >= 70) {
      track.pinchActive = desired;
      track.pinchCandidate = null;
    }
  }

  private updateGesture(track: HandTrack, proposed: GestureName, now: number) {
    if (proposed === track.gesture) {
      track.gestureCandidate = proposed;
      track.gestureCandidateSince = now;
      return;
    }

    if (proposed !== track.gestureCandidate) {
      track.gestureCandidate = proposed;
      track.gestureCandidateSince = now;
      return;
    }

    if (now - track.gestureCandidateSince >= 90) {
      track.gesture = proposed;
      track.gestureCandidate = proposed;
      track.gestureCandidateSince = now;
    }
  }
}
