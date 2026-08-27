# LingBot-World realtime WebSocket

This example exposes a persistent LingBot AR-Diffusion session through the
public `WS /v1/realtime/world` endpoint. The browser sends held WASD/IJKL key
state and receives JPEG/WebP pixel frames as binary WebSocket messages.

## Start the model server

The bundled deployment uses two GPUs, Ulysses SP=2, stateful spatial VAE
parallel decode, and eager execution (required for current LingBot quality):

```bash
vllm serve robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
  --omni \
  --stage-configs-path vllm_omni/deploy/lingbot_world_realtime.yaml \
  --host 0.0.0.0 \
  --port 8000
```

For a local checkpoint, replace the model ID with its directory. Device IDs and
parallel degrees can be edited in the deploy YAML.

## Start the Web UI

```bash
python -m http.server 8080 \
  --directory examples/online_serving/lingbot_world_realtime
```

Open <http://127.0.0.1:8080>, choose the initial image, then click **Connect**.
Use the keyboard or on-screen buttons:

- `W` / `S`: move forward / backward
- `A` / `D`: strafe left / right
- `I` / `K`: pitch up / down
- `J` / `L`: yaw left / right

## Protocol

Start a session with one JSON text message:

```json
{
  "type": "session.start",
  "prompt": "A stable first-person world.",
  "image_reference": "data:image/jpeg;base64,...",
  "width": 832,
  "height": 480,
  "fps": 16,
  "pixel_format": "jpeg",
  "pixel_quality": 85,
  "initial_actions": []
}
```

Update held key state whenever it changes:

```json
{
  "type": "session.control",
  "event_id": 1,
  "actions": ["w", "l"],
  "client_ts_ms": 1730000000000
}
```

The server replies with `video.chunk` JSON metadata followed immediately by
`frame_count` binary JPEG/WebP frames. `session.control.queued` acknowledges
accepted controls, and `session.done` closes a stopped or bounded session.

Only one realtime world session is admitted per server process. AR-Diffusion
currently requires a single stage replica and session-affine worker state.
