(() => {
  "use strict";

  const validKeys = new Set(["w", "a", "s", "d", "i", "j", "k", "l"]);
  const held = new Set();
  const frameQueue = [];
  const canvas = document.getElementById("world");
  const context = canvas.getContext("2d", { alpha: false });
  const statusNode = document.getElementById("status");
  const logNode = document.getElementById("log");
  const connectButton = document.getElementById("connect");
  const stopButton = document.getElementById("stop");
  const promptButton = document.getElementById("update-prompt");

  let socket = null;
  let eventId = 1;
  let incomingFrames = 0;
  let incomingMime = "image/jpeg";
  let playbackTimer = null;
  let pingTimer = null;
  let drawing = false;
  let droppedFrames = 0;

  function setStatus(text, kind = "idle") {
    statusNode.textContent = text;
    statusNode.className = `status ${kind}`;
  }

  function log(message) {
    const timestamp = new Date().toLocaleTimeString();
    logNode.textContent += `[${timestamp}] ${message}\n`;
    logNode.scrollTop = logNode.scrollHeight;
  }

  function updateKeyButtons() {
    document.querySelectorAll("[data-key]").forEach((button) => {
      button.classList.toggle("active", held.has(button.dataset.key));
    });
  }

  function sendControl(extra = {}) {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    const payload = {
      type: "session.control",
      event_id: eventId++,
      actions: [...held],
      client_ts_ms: Date.now(),
      ...extra,
    };
    socket.send(JSON.stringify(payload));
  }

  function setHeld(key, active) {
    if (!validKeys.has(key)) return;
    const changed = active ? !held.has(key) : held.has(key);
    if (!changed) return;
    if (active) held.add(key);
    else held.delete(key);
    updateKeyButtons();
    sendControl();
  }

  async function drawNextFrame() {
    if (drawing || frameQueue.length === 0) return;
    drawing = true;
    const blob = frameQueue.shift();
    document.getElementById("buffer").textContent = String(frameQueue.length);
    try {
      const bitmap = await createImageBitmap(blob);
      if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
      }
      context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      bitmap.close();
    } catch (error) {
      log(`Frame decode failed: ${error.message}`);
    } finally {
      drawing = false;
    }
  }

  function startPlayback(fps) {
    clearInterval(playbackTimer);
    playbackTimer = setInterval(drawNextFrame, 1000 / Math.max(1, fps));
  }

  function stopConnection() {
    clearInterval(playbackTimer);
    clearInterval(pingTimer);
    playbackTimer = null;
    pingTimer = null;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "session.stop" }));
      socket.close();
    }
    socket = null;
    held.clear();
    frameQueue.length = 0;
    updateKeyButtons();
    connectButton.disabled = false;
    stopButton.disabled = true;
    promptButton.disabled = true;
    setStatus("Disconnected");
  }

  function readImageReference() {
    const input = document.getElementById("image");
    const file = input.files && input.files[0];
    if (!file) throw new Error("Choose an initial image first.");
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error || new Error("Failed to read image."));
      reader.readAsDataURL(file);
    });
  }

  async function connect() {
    if (socket) stopConnection();
    try {
      const imageReference = await readImageReference();
      const url = document.getElementById("url").value.trim();
      const prompt = document.getElementById("prompt").value.trim();
      const quality = Number(document.getElementById("quality").value);
      const pixelFormat = document.getElementById("pixel-format").value;
      if (!url || !prompt) throw new Error("WebSocket URL and prompt are required.");

      eventId = 1;
      incomingFrames = 0;
      droppedFrames = 0;
      frameQueue.length = 0;
      logNode.textContent = "";
      setStatus("Connecting...");
      connectButton.disabled = true;
      socket = new WebSocket(url);
      socket.binaryType = "arraybuffer";
      socket.onopen = () => {
        socket.send(JSON.stringify({
          type: "session.start",
          prompt,
          image_reference: imageReference,
          width: 832,
          height: 480,
          fps: 16,
          seed: 42,
          num_inference_steps: 4,
          flow_shift: 5.0,
          pixel_format: pixelFormat,
          pixel_quality: quality,
          initial_actions: [...held],
        }));
        pingTimer = setInterval(() => {
          if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: "session.ping" }));
          }
        }, 20_000);
      };
      socket.onmessage = (event) => {
        if (typeof event.data === "string") {
          const message = JSON.parse(event.data);
          if (message.type === "session.started") {
            setStatus("Live", "connected");
            stopButton.disabled = false;
            promptButton.disabled = false;
            incomingMime = message.mime_type || "image/jpeg";
            startPlayback(message.fps || 16);
          } else if (message.type === "video.chunk") {
            incomingFrames = message.frame_count;
            incomingMime = message.mime_type || incomingMime;
            document.getElementById("chunk").textContent = String(message.chunk_index);
            document.getElementById("latency").textContent = `${Number(message.latency_ms).toFixed(0)} ms`;
            document.getElementById("applied").textContent = (message.applied_event_ids || []).join(",") || "-";
          } else if (message.type === "session.control.queued") {
            log(`Control ${message.event_id}: ${message.status}`);
          } else if (message.type === "session.done") {
            log(`Session done after ${message.chunks} chunks.`);
            stopConnection();
          } else if (message.type === "session.pong") {
            // Keepalive acknowledgement; no UI update needed.
          } else if (message.type === "error") {
            setStatus("Error", "error");
            log(`ERROR ${message.code || ""}: ${message.message}`);
          } else {
            log(JSON.stringify(message));
          }
          return;
        }

        if (incomingFrames <= 0) {
          log("Received an unexpected binary frame without video.chunk metadata.");
          return;
        }
        incomingFrames -= 1;
        frameQueue.push(new Blob([event.data], { type: incomingMime }));
        if (frameQueue.length > 48) {
          frameQueue.splice(0, frameQueue.length - 48);
          droppedFrames += 1;
          log(`Dropped stale frames to preserve interactivity (drops=${droppedFrames}).`);
        }
        document.getElementById("buffer").textContent = String(frameQueue.length);
      };
      socket.onerror = () => setStatus("WebSocket error", "error");
      socket.onclose = () => {
        if (socket) stopConnection();
      };
    } catch (error) {
      setStatus("Error", "error");
      connectButton.disabled = false;
      log(error.message);
    }
  }

  window.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
    const key = event.key.toLowerCase();
    if (validKeys.has(key)) {
      event.preventDefault();
      setHeld(key, true);
    }
  });
  window.addEventListener("keyup", (event) => {
    const key = event.key.toLowerCase();
    if (validKeys.has(key)) {
      event.preventDefault();
      setHeld(key, false);
    }
  });
  window.addEventListener("blur", () => {
    if (held.size) {
      held.clear();
      updateKeyButtons();
      sendControl();
    }
  });

  document.querySelectorAll("[data-key]").forEach((button) => {
    const key = button.dataset.key;
    button.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      button.setPointerCapture(event.pointerId);
      setHeld(key, true);
    });
    const release = (event) => {
      event.preventDefault();
      setHeld(key, false);
    };
    button.addEventListener("pointerup", release);
    button.addEventListener("pointercancel", release);
  });

  connectButton.addEventListener("click", connect);
  stopButton.addEventListener("click", stopConnection);
  promptButton.addEventListener("click", () => {
    const prompt = document.getElementById("prompt").value.trim();
    if (prompt) sendControl({ prompt });
  });
})();
