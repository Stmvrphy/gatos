import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---- meme mapping -----------------------------------------------------
// Each gesture maps to one or more meme images. When a gesture has more
// than one image, one is picked at random each time the gesture is newly
// (re)triggered, so repeated gestures don't always show the same frame.
const GESTURE_MEMES = {
  crossFingers: ["memes/BLIZZARD_OZZY.webp"],
  twoHandsCoverFace: ["memes/BOOTLEG_DYLAN.jpg"],
  heroesBowie: ["memes/HEROES_BOWIE.jpg"],
  queenShoulders: ["memes/QUEEN2_QUEEN.jpeg"],
  madonnaTrueBlue: ["memes/MADONNA_TRUE_BLUE.jpeg"],
  theBendsRadiohead: ["memes/THEBENDS_RADIOHEAD.jpeg"],
  default: ["memes/rick_cover.jpg"],
};

// how many consecutive frames a gesture must hold before we switch to it
const STABLE_FRAMES_REQUIRED = 5;
// if no hand / no gesture is seen for this long, fall back to default
const DEFAULT_FALLBACK_MS = 600;
// how long we trust a stale face box after the face detector loses the face
const FACE_STALE_MS = 1200;

const video = document.getElementById("video");
const memeImg = document.getElementById("memeImg");
const placeholder = document.getElementById("placeholder");
const debugHud = document.getElementById("debugHud");

let handLandmarker, faceLandmarker;
let lastVideoTime = -1;
let currentGesture = "";
let candidateGesture = "default";
let candidateStreak = 0;
let lastNonDefaultAt = performance.now();
let lastFace = null; // { mouthCenter, faceWidth, mouthOpen, yawDeg, pitchDeg, t }
let lastFaceSeenThisFrame = false;
let lastYawDebug = 0;
let lastPitchDebug = 0;

async function init() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
  });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  applyGesture("default");
  requestAnimationFrame(loop);
}

// ---- 3D-aware geometry helpers -----------------------------------------
// Using z (depth) as well as x/y makes these tests far more robust to hand
// rotation, foreshortening, and motion blur than a plain 2D/wrist-distance
// check would be.
function vec(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z || 0) - (a.z || 0) };
}
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}
function angleDeg(v1, v2) {
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const m1 = Math.hypot(v1.x, v1.y, v1.z);
  const m2 = Math.hypot(v2.x, v2.y, v2.z);
  if (m1 < 1e-9 || m2 < 1e-9) return 180;
  return (Math.acos(Math.min(1, Math.max(-1, dot / (m1 * m2)))) * 180) / Math.PI;
}

// a finger is "extended" if its two segments (mcp->pip, pip->tip) point in
// roughly the same direction; "curled" if it folds back sharply.
function fingerExtended(lm, mcp, pip, tip) {
  const angle = angleDeg(vec(lm[mcp], lm[pip]), vec(lm[pip], lm[tip]));
  return angle < 45;
}

// extract the head's left/right turn angle (yaw, degrees) from MediaPipe's
// facial transformation matrix - its own estimate of head pose, far more
// robust than trying to infer turn from landmark distances.
function headPoseAnglesFromMatrix(matrixData) {
  const r00 = matrixData[0];
  const r10 = matrixData[4];
  const r20 = matrixData[8];
  const r21 = matrixData[9];
  const r22 = matrixData[10];
  const sy = Math.hypot(r00, r10);
  if (sy < 1e-6) return { yaw: 0, pitch: 0 };
  const yaw = (Math.atan2(-r20, sy) * 180) / Math.PI;
  const pitch = (Math.atan2(r21, r22) * 180) / Math.PI;
  return { yaw, pitch };
}

function classifyHand(lm) {
  const handScale = dist(lm[0], lm[9]) || 1e-6; // wrist -> middle mcp

  const indexUp = fingerExtended(lm, 5, 6, 8);
  const middleUp = fingerExtended(lm, 9, 10, 12);
  const ringUp = fingerExtended(lm, 13, 14, 16);
  const pinkyUp = fingerExtended(lm, 17, 18, 20);

  // thumb + pinky spread apart from each other = shaka/rock-on shape.
  // tucked thumb sits close to the pinky-side of the palm; an abducted
  // thumb sticks straight out and this distance grows a lot.
  const thumbPinkySpread = dist(lm[4], lm[17]) / handScale;
  const thumbOut = thumbPinkySpread > 1.05;

  const curledCount = [indexUp, middleUp, ringUp, pinkyUp].filter((v) => !v).length;

  return {
    indexUp,
    middleUp,
    ringUp,
    pinkyUp,
    thumbOut,
    curledCount,
    handScale,
    indexTip: lm[8],
    indexPip: lm[6],
    indexMcp: lm[5],
    wrist: lm[0],
    palmCenter: lm[9],
  };
}

function updateFace(faceResult) {
  const now = performance.now();
  const sawFace = !!(faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0);

  if (sawFace) {
    const f = faceResult.faceLandmarks[0];
    const upperLip = f[13];
    const lowerLip = f[14];
    const rightCheek = f[234];
    const leftCheek = f[454];
    const mouthCenter = {
      x: (upperLip.x + lowerLip.x) / 2,
      y: (upperLip.y + lowerLip.y) / 2,
      z: ((upperLip.z || 0) + (lowerLip.z || 0)) / 2,
    };
    const faceWidth = dist(rightCheek, leftCheek);
    // how open the mouth is right now - normalized so it doesn't depend on
    // distance from the camera.
    const mouthOpen = dist(upperLip, lowerLip) / faceWidth;

    let yawDeg = 0;
    let pitchDeg = 0;
    if (faceResult.facialTransformationMatrixes && faceResult.facialTransformationMatrixes.length > 0) {
      const angles = headPoseAnglesFromMatrix(faceResult.facialTransformationMatrixes[0].data);
      yawDeg = angles.yaw;
      pitchDeg = angles.pitch;
    }

    lastFace = { mouthCenter, faceWidth, mouthOpen, yawDeg, pitchDeg, t: now };
    lastYawDebug = yawDeg;
    lastPitchDebug = pitchDeg;
  }
  lastFaceSeenThisFrame = sawFace;
}

// a hand is "pointing" if only the index finger is extended (thumb can be
// either way) - the shape both hands make in the finger-tips-touching pose.
function isPointing(h) {
  return h.indexUp && !h.middleUp && !h.ringUp && !h.pinkyUp;
}

function isCrossGesture(h0, h1) {
  if (!isPointing(h0) || !isPointing(h1)) return false;
  const avgScale = (h0.handScale + h1.handScale) / 2;

  const v0 = vec(h0.indexMcp, h0.indexTip);
  const v1 = vec(h1.indexMcp, h1.indexTip);

  const ang = angleDeg(v0, v1);
  const isPerpendicular = ang >= 50 && ang <= 130;

  const v0Horiz = Math.abs(v0.x) > Math.abs(v0.y) * 0.8;
  const v0Vert = Math.abs(v0.y) > Math.abs(v0.x) * 0.8;
  const v1Horiz = Math.abs(v1.x) > Math.abs(v1.y) * 0.8;
  const v1Vert = Math.abs(v1.y) > Math.abs(v1.x) * 0.8;

  const oneHOneV = (v0Horiz && v1Vert) || (v0Vert && v1Horiz);

  const mid0 = {
    x: (h0.indexMcp.x + h0.indexTip.x) / 2,
    y: (h0.indexMcp.y + h0.indexTip.y) / 2,
    z: ((h0.indexMcp.z || 0) + (h0.indexTip.z || 0)) / 2,
  };
  const mid1 = {
    x: (h1.indexMcp.x + h1.indexTip.x) / 2,
    y: (h1.indexMcp.y + h1.indexTip.y) / 2,
    z: ((h1.indexMcp.z || 0) + (h1.indexTip.z || 0)) / 2,
  };

  const dMid = dist(mid0, mid1) / avgScale;
  const dTip0Mid1 = dist(h0.indexTip, mid1) / avgScale;
  const dTip1Mid0 = dist(h1.indexTip, mid0) / avgScale;

  const isClose = Math.min(dMid, dTip0Mid1, dTip1Mid0) < 1.5;

  return isPerpendicular && oneHOneV && isClose;
}

function isBowieHeroesGesture(hands, lastFace) {
  if (hands.length !== 2) return false;
  if (hands.some((h) => h.curledCount >= 3)) return false;

  const hRaised = hands[0].palmCenter.y < hands[1].palmCenter.y ? hands[0] : hands[1];
  const hLower = hands[0].palmCenter.y < hands[1].palmCenter.y ? hands[1] : hands[0];

  const yDiff = hLower.palmCenter.y - hRaised.palmCenter.y;
  if (yDiff < 0.08) return false;

  if (lastFace) {
    const { mouthCenter, faceWidth } = lastFace;

    const lowerUnderFace = hLower.palmCenter.y > mouthCenter.y - faceWidth * 0.3;
    const lowerNearX = Math.abs(hLower.palmCenter.x - mouthCenter.x) / faceWidth < 1.8;
    const lowerDist = dist(hLower.palmCenter, mouthCenter) / faceWidth < 3.0;

    const raisedAtHeadHeight =
      hRaised.palmCenter.y > mouthCenter.y - faceWidth * 2.0 &&
      hRaised.palmCenter.y < mouthCenter.y + faceWidth * 0.7;
    const raisedAtSide = Math.abs(hRaised.palmCenter.x - mouthCenter.x) / faceWidth > 0.5;
    const raisedDist = dist(hRaised.palmCenter, mouthCenter) / faceWidth < 3.0;

    return (
      lowerUnderFace &&
      lowerNearX &&
      lowerDist &&
      raisedAtHeadHeight &&
      raisedAtSide &&
      raisedDist
    );
  }
  const xDiff = Math.abs(hRaised.palmCenter.x - hLower.palmCenter.x);
  return yDiff > 0.10 && xDiff > 0.10;
}

function isQueenShouldersGesture(hands, lastFace) {
  if (hands.length !== 2) return false;
  if (hands.some((h) => h.curledCount >= 3)) return false;

  const yDiff = Math.abs(hands[0].palmCenter.y - hands[1].palmCenter.y);
  if (yDiff > 0.18) return false;

  const hLeft = hands[0].palmCenter.x < hands[1].palmCenter.x ? hands[0] : hands[1];
  const hRight = hands[0].palmCenter.x < hands[1].palmCenter.x ? hands[1] : hands[0];
  const xSpread = hRight.palmCenter.x - hLeft.palmCenter.x;

  if (xSpread < 0.12 || xSpread > 0.48) return false;

  if (lastFace) {
    const { mouthCenter, faceWidth } = lastFace;
    const bothBelowMouth = hands.every(
      (h) => h.palmCenter.y > mouthCenter.y + faceWidth * 0.15
    );
    const bothAboveWaist = hands.every(
      (h) => h.palmCenter.y < mouthCenter.y + faceWidth * 3.2
    );
    const leftOnLeft = hLeft.palmCenter.x < mouthCenter.x + faceWidth * 0.2;
    const rightOnRight = hRight.palmCenter.x > mouthCenter.x - faceWidth * 0.2;

    return bothBelowMouth && bothAboveWaist && leftOnLeft && rightOnRight;
  }
  const avgY = (hands[0].palmCenter.y + hands[1].palmCenter.y) / 2;
  return avgY > 0.35;
}

function decideGesture(handResult) {
  const now = performance.now();
  const faceIsFresh = !!lastFace && now - lastFace.t < FACE_STALE_MS;

  if (!handResult.landmarks || handResult.landmarks.length === 0) {
    if (faceIsFresh) {
      if (Math.abs(lastFace.pitchDeg) > 10.0) {
        if (Math.abs(lastFace.yawDeg) > 18.0) {
          return "madonnaTrueBlue";
        }
        return "theBendsRadiohead";
      }
    }
    return "default";
  }

  const hands = handResult.landmarks.map(classifyHand);

  if (hands.length === 2) {
    if (isCrossGesture(hands[0], hands[1])) {
      return "crossFingers";
    }

    if (isBowieHeroesGesture(hands, faceIsFresh ? lastFace : null)) {
      return "heroesBowie";
    }

    if (isQueenShouldersGesture(hands, faceIsFresh ? lastFace : null)) {
      return "queenShoulders";
    }

    if (faceIsFresh) {
      const { mouthCenter, faceWidth } = lastFace;
      const coverThreshold = !lastFaceSeenThisFrame ? 1.4 : 1.2;
      const bothCovering = hands.every(
        (h) => dist(h.palmCenter, mouthCenter) / faceWidth < coverThreshold
      );
      if (bothCovering) {
        return "twoHandsCoverFace";
      }
    }
  }

  if (faceIsFresh) {
    if (Math.abs(lastFace.pitchDeg) > 10.0) {
      if (Math.abs(lastFace.yawDeg) > 18.0) {
        return "madonnaTrueBlue";
      }
      return "theBendsRadiohead";
    }
  }

  return "default";
}

function pickImage(gesture) {
  const images = GESTURE_MEMES[gesture];
  return images ? images[Math.floor(Math.random() * images.length)] : "";
}

function applyGesture(gesture) {
  if (gesture === currentGesture) return;
  currentGesture = gesture;
  if (gesture in GESTURE_MEMES) {
    memeImg.src = pickImage(gesture);
    memeImg.style.display = "block";
    if (placeholder) placeholder.style.display = "none";
  } else {
    memeImg.src = "";
    memeImg.style.display = "none";
    if (placeholder) placeholder.style.display = "block";
  }
}

function loop() {
  const now = performance.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const ts = performance.now();

    const handResult = handLandmarker.detectForVideo(video, ts);
    const faceResult = faceLandmarker.detectForVideo(video, ts);
    updateFace(faceResult);

    const gesture = decideGesture(handResult);

    // debounce
    if (gesture === candidateGesture) {
      candidateStreak++;
    } else {
      candidateGesture = gesture;
      candidateStreak = 1;
    }

    if (candidateStreak >= STABLE_FRAMES_REQUIRED) {
      applyGesture(gesture);
    }

    if (gesture !== "default") lastNonDefaultAt = now;
    if (now - lastNonDefaultAt > DEFAULT_FALLBACK_MS && currentGesture !== "default") {
      applyGesture("default");
    }

    updateDebugHud();
  }
  requestAnimationFrame(loop);
}

function updateDebugHud() {
  if (!debugHud) return;
  debugHud.textContent =
    `Gesto: ${currentGesture}\n` +
    `Yaw: ${lastYawDebug >= 0 ? "+" : ""}${lastYawDebug.toFixed(1)}°\n` +
    `Pitch: ${lastPitchDebug >= 0 ? "+" : ""}${lastPitchDebug.toFixed(1)}°`;
}

init().catch((err) => console.error(err));
