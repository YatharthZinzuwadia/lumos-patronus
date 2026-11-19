export type GestureName =
  | "PINCH"
  | "OPEN PALM"
  | "FIST"
  | "POINT"
  | "VICTORY"
  | "NEUTRAL";

export type Landmark = {
  x: number;
  y: number;
  z?: number;
};

type Category = {
  categoryName?: string;
  score?: number;
};

export type GestureFrame = {
  landmarks: Landmark[][];
  gestures: Category[][];
  handednesses: Category[][];
};

export type StableHand = {
  index: number;
  side: string;
  gesture: GestureName;
  confidence: number;
  pinchRatio: number;
};

type GestureTrack = {
  stable: GestureName;
  candidate: GestureName;
  candidateSince: number;
  history: GestureName[];
};

function distance(a: Landmark, b: Landmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function mapModelGesture(name: string): GestureName {
  switch (name) {
    case "Open_Palm":
      return "OPEN PALM";

    case "Closed_Fist":
      return "FIST";

    case "Pointing_Up":
      return "POINT";

    case "Victory":
      return "VICTORY";

    default:
      return "NEUTRAL";
  }
}

export class GestureStabilizer {
  private tracks = new Map<string, GestureTrack>();

  // How many recent observations we retain.
  private readonly historySize = 7;

  // Minimum number of matching observations.
  private readonly requiredVotes = 4;

  // Minimum time a new gesture must remain the leading candidate.
  private readonly transitionDelay = 120;

  reset() {
    this.tracks.clear();
  }

  update(result: GestureFrame, now: number): StableHand[] {
    const landmarks = result.landmarks ?? [];

    if (landmarks.length === 0) {
      this.reset();
      return [];
    }

    const seen = new Set<string>();

    const output = landmarks.map((hand, index) => {
      const side =
        result.handednesses?.[index]?.[0]?.categoryName?.toUpperCase() ??
        `HAND ${index + 1}`;

      seen.add(side);

      const category = result.gestures?.[index]?.[0];

      const confidence = category?.score ?? 0;

      // Pinch geometry:
      // Thumb tip = 4
      // Index fingertip = 8
      // Index MCP = 5
      // Pinky MCP = 17
      const palmWidth = distance(hand[5], hand[17]);

      const pinchDistance = distance(hand[4], hand[8]);

      const pinchRatio = pinchDistance / Math.max(palmWidth, 0.0001);

      let track = this.tracks.get(side);

      if (!track) {
        track = {
          stable: "NEUTRAL",
          candidate: "NEUTRAL",
          candidateSince: now,
          history: [],
        };

        this.tracks.set(side, track);
      }

      // Hysteresis: once pinching, require a wider separation
      // before releasing the pinch state.
      const alreadyPinching =
        track.stable === "PINCH" || track.candidate === "PINCH";

      const pinchThreshold = alreadyPinching ? 0.38 : 0.27;

      let rawGesture: GestureName;

      if (pinchRatio < pinchThreshold) {
        rawGesture = "PINCH";
      } else if (confidence >= 0.35) {
        rawGesture = mapModelGesture(category?.categoryName ?? "None");
      } else {
        rawGesture = "NEUTRAL";
      }

      // Keep a rolling history of observations.
      track.history.push(rawGesture);

      if (track.history.length > this.historySize) {
        track.history.shift();
      }

      // Count votes for each gesture.
      const votes = new Map<GestureName, number>();

      for (const gesture of track.history) {
        votes.set(gesture, (votes.get(gesture) ?? 0) + 1);
      }

      // Prefer the existing stable gesture if there's a tie.
      let winner = track.stable;
      let winnerCount = votes.get(winner) ?? 0;

      for (const [gesture, count] of votes.entries()) {
        if (count > winnerCount) {
          winner = gesture;
          winnerCount = count;
        }
      }

      if (winner === track.stable) {
        track.candidate = track.stable;
        track.candidateSince = now;
      } else {
        if (winner !== track.candidate) {
          track.candidate = winner;
          track.candidateSince = now;
        }

        const heldLongEnough =
          now - track.candidateSince >= this.transitionDelay;

        const enoughVotes = winnerCount >= this.requiredVotes;

        if (heldLongEnough && enoughVotes) {
          track.stable = winner;
          track.candidate = winner;
          track.candidateSince = now;
        }
      }

      return {
        index,
        side,
        gesture: track.stable,
        confidence,
        pinchRatio,
      };
    });

    // Remove tracks for hands that are no longer visible.
    for (const side of this.tracks.keys()) {
      if (!seen.has(side)) {
        this.tracks.delete(side);
      }
    }

    return output;
  }
}
