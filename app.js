// Global variables for optical flow
let video = document.getElementById('video');
let canvasOutput = document.getElementById('canvasOutput');
let ctx = canvasOutput.getContext('2d');
let cap; // OpenCV VideoCapture
let prevGray = null;
let prevPoints = null;
const maxCorners = 100;
const qualityLevel = 0.3;
const minDistance = 7;
const blockSize = 7;
const winSize = new cv.Size(15, 15);
const maxLevel = 2;
const criteria = new cv.TermCriteria(cv.TermCriteria_EPS | cv.TermCriteria_COUNT, 10, 0.03);

// Variables for inertial integration
let lastAccelTime = null;
let displacementAccel = 0; // in meters
let accelVector = { x: 0, y: 0, z: 0 };

// Variables for optical flow displacement
let displacementFlow = 0; // in “world” units (meters)
const pixelToMeter = 0.001; // example conversion factor (to be calibrated)

// Complementary filter parameter (0 ≤ alpha ≤ 1)
// alpha: weight for the inertial integration; (1-alpha): weight for optical flow
const alpha = 0.6;

// Final fused displacement estimate
let fusedDisplacement = 0;

// Wait for OpenCV.js to load
function onOpenCvReady() {
  console.log('OpenCV.js is ready.');
  startCamera();
}

// Start the camera using getUserMedia (MDN getUserMedia API)
// :contentReference[oaicite:0]{index=0}
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    video.play();
    // Once the video is playing, start processing frames
    video.addEventListener('playing', () => {
      cap = new cv.VideoCapture(video);
      processVideo();
    });
  } catch (err) {
    console.error('Error accessing camera:', err);
  }
}

// Process video frames to compute optical flow displacement
function processVideo() {
  let frame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  let gray = new cv.Mat();
  // Read frame from video
  cap.read(frame);
  cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY);

  // If no previous frame, initialize keypoints
  if (!prevGray) {
    prevGray = gray.clone();
    prevPoints = new cv.Mat();
    cv.goodFeaturesToTrack(prevGray, prevPoints, maxCorners, qualityLevel, minDistance, new cv.Mat(), blockSize);
  } else {
    // Calculate optical flow using Lucas-Kanade method
    let nextPoints = new cv.Mat();
    let status = new cv.Mat();
    let err = new cv.Mat();
    cv.calcOpticalFlowPyrLK(prevGray, gray, prevPoints, nextPoints, status, err, winSize, maxLevel, criteria);

    // Compute average displacement in pixels for the points that were successfully tracked
    let dx = 0, dy = 0, count = 0;
    for (let i = 0; i < status.rows; i++) {
      if (status.data[i] === 1) {
        let prevPt = { x: prevPoints.data32F[i*2], y: prevPoints.data32F[i*2+1] };
        let nextPt = { x: nextPoints.data32F[i*2], y: nextPoints.data32F[i*2+1] };
        dx += nextPt.x - prevPt.x;
        dy += nextPt.y - prevPt.y;
        count++;
      }
    }
    if (count > 0) {
      let avgDx = dx / count;
      let avgDy = dy / count;
      // Euclidean displacement in pixels
      let flowPixels = Math.sqrt(avgDx*avgDx + avgDy*avgDy);
      // Convert pixel displacement to meters (using an assumed conversion factor)
      displacementFlow = flowPixels * pixelToMeter;
      document.getElementById('flowInfo').textContent = `Optical Flow: ${avgDx.toFixed(2)}, ${avgDy.toFixed(2)} (px)`;
    }

    // Prepare for next iteration
    prevGray.delete();
    prevGray = gray.clone();
    prevPoints.delete();
    // Re-detect keypoints periodically (here every frame for simplicity)
    prevPoints = new cv.Mat();
    cv.goodFeaturesToTrack(prevGray, prevPoints, maxCorners, qualityLevel, minDistance, new cv.Mat(), blockSize);
    nextPoints.delete();
    status.delete();
    err.delete();
  }
  // Draw the frame (optional visualization)
  cv.imshow('canvasOutput', frame);
  frame.delete();

  // Fuse with the inertial displacement and update the final estimate
  fuseDisplacements();

  // Schedule next frame processing
  requestAnimationFrame(processVideo);
}

// Complementary filter: combine inertial displacement and optical flow displacement
function fuseDisplacements() {
  // Simple weighted average
  fusedDisplacement = alpha * displacementAccel + (1 - alpha) * displacementFlow;
  document.getElementById('displacement').textContent = `Estimated Displacement: ${fusedDisplacement.toFixed(3)} m`;
}

// DeviceMotion event handler for inertial sensor readings (MDN DeviceMotion API)
// :contentReference[oaicite:1]{index=1}
window.addEventListener('devicemotion', (event) => {
  const now = performance.now();
  if (lastAccelTime === null) {
    lastAccelTime = now;
    return;
  }
  const dt = (now - lastAccelTime) / 1000; // convert ms to seconds
  lastAccelTime = now;
  
  // Use acceleration without gravity if available; otherwise, use accelerationIncludingGravity
  let ax = event.acceleration && event.acceleration.x !== null ? event.acceleration.x : event.accelerationIncludingGravity.x;
  let ay = event.acceleration && event.acceleration.y !== null ? event.acceleration.y : event.accelerationIncludingGravity.y;
  let az = event.acceleration && event.acceleration.z !== null ? event.acceleration.z : event.accelerationIncludingGravity.z;
  
  // For this demo, we simply assume that the device is held horizontally,
  // so most motion is along one axis (e.g., the x-axis). In a real application,
  // you’d remove the gravity component using device orientation.
  accelVector.x = ax;
  accelVector.y = ay;
  accelVector.z = az;
  
  // Simple integration: displacement = previous displacement + 0.5 * acceleration * dt^2
  // (Note: In practice, you would integrate velocity and apply drift corrections.)
  displacementAccel += 0.5 * ax * dt * dt;
  
  document.getElementById('accelInfo').textContent = `Acceleration: ${ax.toFixed(2)}, ${ay.toFixed(2)}, ${az.toFixed(2)} (m/s²)`;
});
