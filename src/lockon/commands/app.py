from __future__ import annotations

import typer

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
) -> None:
    from lockon.servicers.turret import serve

    server = serve(host=host, port=port)
    typer.echo(f"Turret gRPC server listening on {host}:{port}")
    server.wait_for_termination()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
