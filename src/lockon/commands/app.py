from __future__ import annotations

import typer

from lockon.vcodec import ObservationFormat

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
    observation_format: ObservationFormat = typer.Option(
        ObservationFormat.RGB,
        "--observation-format",
        envvar="LOCKON_TURRET_OBSERVATION_FORMAT",
        help="Observation transport format: rgb, jpeg, or h264.",
    ),
    jpeg_quality: int = typer.Option(
        80,
        "--jpeg-quality",
        envvar="LOCKON_TURRET_JPEG_QUALITY",
        help="JPEG quality used when observation format is jpeg.",
    ),
    h264_bitrate_kbps: int = typer.Option(
        4000,
        "--h264-bitrate-kbps",
        envvar="LOCKON_TURRET_H264_BITRATE_KBPS",
        help="Target H.264 bitrate in kbps.",
    ),
    h264_gop: int = typer.Option(
        30,
        "--h264-gop",
        envvar="LOCKON_TURRET_H264_GOP",
        help="H.264 GOP/keyframe interval in frames.",
    ),
    show_mujoco_viewer: bool = typer.Option(
        False,
        "--show-mujoco-viewer",
        envvar="LOCKON_TURRET_SHOW_MUJOCO_VIEWER",
        help="Show a live MuJoCo viewer synced to the turret server state.",
    ),
) -> None:
    from lockon.servicers.turret import create_servicer, serve

    servicer = create_servicer(
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fovy_deg=camera_fovy,
        observation_format=observation_format.value,
        jpeg_quality=jpeg_quality,
        h264_bitrate_kbps=h264_bitrate_kbps,
        h264_gop=h264_gop,
    )
    server = serve(
        host=host,
        port=port,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fovy_deg=camera_fovy,
        observation_format=observation_format.value,
        jpeg_quality=jpeg_quality,
        h264_bitrate_kbps=h264_bitrate_kbps,
        h264_gop=h264_gop,
        servicer=servicer,
    )
    typer.echo(
        f"Turret gRPC server listening on {host}:{port} "
        f"({observation_format.value}, {camera_width}x{camera_height}, fovy={camera_fovy})"
    )
    if not show_mujoco_viewer:
        try:
            server.wait_for_termination()
        finally:
            servicer.close()
        return

    import time

    import mujoco.viewer

    try:
        while True:
            session = servicer.get_viewer_session()
            if session is None or session.env is None:
                time.sleep(0.1)
                continue

            with mujoco.viewer.launch_passive(session.env.model, session.env.data) as viewer:
                while viewer.is_running():
                    active_session = servicer.get_viewer_session()
                    if active_session is not session:
                        break
                    with session.lock:
                        viewer.sync()
                    time.sleep(session.env.dt)
            break
    finally:
        server.stop(grace=0)
        servicer.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
