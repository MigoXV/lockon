from __future__ import annotations

from concurrent import futures

import grpc
import typer

from lockon.protos.gym_env import gym_env_pb2_grpc
from lockon.protos.gym_v2 import gym_env_pb2_grpc as gym_v2_pb2_grpc

app = typer.Typer(no_args_is_help=True)


@app.command("turret-server")
def turret_server(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        envvar="LOCKON_TURRET_HOST",
        help="Host/interface to bind the turret gRPC server to.",
    ),
    port: int = typer.Option(
        50051,
        "--port",
        envvar="LOCKON_TURRET_PORT",
        help="Port to bind the turret gRPC server to.",
    ),
    camera_width: int = typer.Option(
        640,
        "--camera-width",
        envvar="LOCKON_TURRET_CAMERA_WIDTH",
        help="Rendered camera frame width in pixels.",
    ),
    camera_height: int = typer.Option(
        480,
        "--camera-height",
        envvar="LOCKON_TURRET_CAMERA_HEIGHT",
        help="Rendered camera frame height in pixels.",
    ),
    camera_fovy: float = typer.Option(
        60.0,
        "--camera-fovy",
        envvar="LOCKON_TURRET_CAMERA_FOVY",
        help="Rendered camera vertical field of view in degrees.",
    ),
    show_mujoco_viewer: bool = typer.Option(
        False,
        "--show-mujoco-viewer",
        envvar="LOCKON_TURRET_SHOW_MUJOCO_VIEWER",
        help="Show a live MuJoCo viewer synced to the turret server state.",
    ),
) -> None:
    from lockon.servicers.turret import create_servicer as create_v1_servicer
    from lockon.servicers.turret_v2 import create_servicer as create_v2_servicer

    servicer = create_v1_servicer(
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fovy_deg=camera_fovy,
    )
    servicer_v2 = create_v2_servicer(
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fovy_deg=camera_fovy,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    gym_env_pb2_grpc.add_ArmEnvServicer_to_server(servicer, server)
    gym_v2_pb2_grpc.add_GymEnvServicer_to_server(servicer_v2, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    typer.echo(
        f"Turret gRPC server listening on {host}:{port} with GymEnv v1 and GymEnv v2 "
        f"(rgb, {camera_width}x{camera_height}, fovy={camera_fovy})"
    )
    if not show_mujoco_viewer:
        try:
            server.wait_for_termination()
        finally:
            servicer.close()
            servicer_v2.close()
        return

    import time

    import mujoco.viewer

    def _get_viewer_session() -> object | None:
        session = servicer.get_viewer_session()
        if session is not None and session.env is not None:
            return session
        session_v2 = servicer_v2.get_viewer_session()
        if session_v2 is not None and session_v2.env is not None:
            return session_v2
        return None

    try:
        while True:
            session = _get_viewer_session()
            if session is None or session.env is None:
                time.sleep(0.1)
                continue

            with mujoco.viewer.launch_passive(session.env.model, session.env.data) as viewer:
                while viewer.is_running():
                    active_session = _get_viewer_session()
                    if active_session is not session:
                        break
                    with session.lock:
                        viewer.sync()
                    time.sleep(session.env.dt)
            break
    finally:
        server.stop(grace=0)
        servicer.close()
        servicer_v2.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
