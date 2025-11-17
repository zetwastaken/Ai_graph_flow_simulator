"""FastAPI server exposing REST + WebSocket interfaces."""

from __future__ import annotations

import asyncio
from asyncio import QueueEmpty
from typing import Any, Dict

from fastapi import Depends, FastAPI, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ..auth.security import JWTAuthenticator
from ..config import SimulationConfig
from ..simulator import FlowSimulator
from ..streaming.streaming_simulator import StreamingSimulator

app = FastAPI(title="AI Flow Simulator API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _init_components():
    config = SimulationConfig()
    simulator = FlowSimulator(config)
    simulator.setup()
    simulator.run()
    streamer = StreamingSimulator(simulator)
    streamer.start_background(real_time=config.real_time_mode)
    auth = JWTAuthenticator(config.auth_public_key, config.jwt_audience)
    return simulator, streamer, auth


def configure_runtime(simulator: FlowSimulator, streamer: StreamingSimulator, auth: JWTAuthenticator):
    """Inject shared runtime objects when the API runs inside another process."""
    app.state.simulator = simulator
    app.state.streamer = streamer
    app.state.auth = auth
    app.state.external_dependencies = True


@app.on_event("startup")
async def startup_event():
    if getattr(app.state, "external_dependencies", False):
        return
    simulator, streamer, auth = _init_components()
    app.state.simulator = simulator
    app.state.streamer = streamer
    app.state.auth = auth


def require_auth(authorization: str = Header(default="")):
    token = authorization.replace("Bearer ", "")
    auth: JWTAuthenticator = app.state.auth
    auth.verify(token)


@app.get("/status")
async def status(_: Any = Depends(require_auth)):
    simulator: FlowSimulator = app.state.simulator
    report = simulator.generate_report()
    return {
        "simulation": report["simulation_info"],
        "topology": report["topology_info"],
        "flows": report["flow_statistics"],
        "edges": report["edge_statistics"],
        "anomalies": report["anomaly_statistics"],
    }


@app.get("/snapshot")
async def snapshot(limit: int = 100, _: Any = Depends(require_auth)):
    streamer: StreamingSimulator = app.state.streamer
    return streamer.snapshot(limit)


@app.websocket("/ws/stream")
async def stream_endpoint(websocket: WebSocket):
    token = websocket.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        require_auth(token)
    except Exception:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    streamer: StreamingSimulator = app.state.streamer
    queue: asyncio.Queue = asyncio.Queue(maxsize=streamer.config.stream_buffer_size)
    loop = asyncio.get_event_loop()

    def listener(payload: Dict):
        def _push():
            if queue.full():
                try:
                    queue.get_nowait()
                except QueueEmpty:
                    pass
            queue.put_nowait(payload)

        loop.call_soon_threadsafe(_push)

    streamer.register_listener(listener)
    try:
        # Immediately push the latest payload if available
        latest = streamer.latest_payload()
        if latest:
            await websocket.send_json(latest)
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    finally:
        if listener in streamer.listeners:
            streamer.listeners.remove(listener)
