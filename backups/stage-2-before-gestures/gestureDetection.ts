export type Landmark3D = {
  x: number;
  y: number;
  z?: number;
};

export type GestureName = "PINCH" | "OPEN PALM" | "FIST" | "POINT" | "NEUTRAL";

export type DetectedHand = {
  side: string;
  gesture: GestureName;
  pinchRatio: number;
  extendedFingers: number;
};

type HandednessCategory = {
  categoryName?: string;
};

function distance(a: Landmark3D, b: Landmark3D): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = (a.z ?? 0) - (b.z ?? 0);

  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// Calculates the angle formed by three joints.
function jointAngle(a: Landmark3D, b: Landmark3D, c: Landmark3D): number {
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

  const magnitudeA = Math.sqrt(ba.x ** 2 + ba.y ** 2 + ba.z ** 2);

  const magnitudeB = Math.sqrt(bc.x ** 2 + bc.y ** 2 + bc.z ** 2);

  const denominator = magnitudeA * magnitudeB;

  if (denominator < 0.000001) return 0;

  const cosine = Math.max(-1, Math.min(1, dot / denominator));

  return Math.acos(cosine) * (180 / Math.PI);
}

export function detectGestures(
  landmarks: Landmark3D[][],
  worldLandmarks: Landmark3D[][] | undefined,
  handednesses: HandednessCategory[][] | undefined,
): {
  hands: DetectedHand[];
  label: string;
} {
  const hands: DetectedHand[] = landmarks.map((imageHand, index) => {
    // Use actual 3D hand coordinates when available.
    const worldHand = worldLandmarks?.[index];

    const hand = worldHand && worldHand.length >= 21 ? worldHand : imageHand;

    const side =
      handednesses?.[index]?.[0]?.categoryName?.toUpperCase() ??
      `HAND ${index + 1}`;

    // Four main finger extension tests.
    const fingerAngles = [
      jointAngle(hand[5], hand[6], hand[8]),
      jointAngle(hand[9], hand[10], hand[12]),
      jointAngle(hand[13], hand[14], hand[16]),
      jointAngle(hand[17], hand[18], hand[20]),
    ];

    // An angle near 180 degrees indicates an extended finger.
    const extended = fingerAngles.map((angle) => angle > 150);

    const extendedFingers = extended.filter(Boolean).length;

    // Pinch detection: thumb tip versus index fingertip.
    // Normalize against palm width so hand distance from camera
    // has less influence on the threshold.
    const palmWidth = distance(hand[5], hand[17]);

    const pinchDistance = distance(hand[4], hand[8]);

    const pinchRatio = pinchDistance / Math.max(palmWidth, 0.0001);

    let gesture: GestureName = "NEUTRAL";

    if (pinchRatio < 0.3) {
      gesture = "PINCH";
    } else if (extendedFingers === 4) {
      gesture = "OPEN PALM";
    } else if (extended[0] && extendedFingers === 1) {
      gesture = "POINT";
    } else if (extendedFingers === 0) {
      gesture = "FIST";
    }

    return {
      side,
      gesture,
      pinchRatio,
      extendedFingers,
    };
  });

  let label = "NO HAND";

  if (hands.length === 1) {
    label = `${hands[0].side} · ${hands[0].gesture}`;
  } else if (hands.length >= 2) {
    label = `${hands.length} HANDS · ${hands
      .map((hand) => `${hand.side}: ${hand.gesture}`)
      .join(" / ")}`;
  }

  return {
    hands,
    label,
  };
}
