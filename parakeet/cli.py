from __future__ import annotations

import argparse
import logging

import trio

from parakeet.config import Config
from parakeet.server import ASRSocketServer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parakeet")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Start the streaming socket server.")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--status-port", type=int, default=None)
    serve.add_argument("--device", default="cpu")
    serve.add_argument("--model-size", choices=("small", "large"), default="small")
    serve.add_argument("--max-num-streams", type=int, default=100)
    serve.add_argument("--sample-rate", type=int, default=16000)
    serve.add_argument("--max-stream-seconds", type=int, default=300)
    serve.add_argument("--max-seq-len", type=int, default=5000)
    return parser


def _print_config(
    config: Config,
    host: str,
    port: int,
    status_port: int | None,
    device: str,
) -> None:
    print("Starting Parakeet streaming socket server")
    print(f"  host: {host}")
    print(f"  port: {port}")
    if status_port is not None:
        print(f"  status_port: {status_port}")
    print(f"  device: {device}")
    print(f"  model_size: {config.size}")
    print(f"  max_num_streams: {config.max_num_streams}")
    print(f"  sample_rate: {config.sample_rate}")
    print(f"  max_stream_seconds: {config.max_stream_seconds}")
    print(f"  max_seq_len: {config.max_seq_len}")


async def _serve_async(args: argparse.Namespace) -> None:
    config = Config(
        size=args.model_size,
        max_num_streams=args.max_num_streams,
        sample_rate=args.sample_rate,
        max_stream_seconds=args.max_stream_seconds,
        max_seq_len=args.max_seq_len,
    )
    _print_config(config, args.host, args.port, args.status_port, args.device)
    server = ASRSocketServer(
        config,
        host=args.host,
        port=args.port,
        status_port=args.status_port,
        device=args.device,
    )
    try:
        await server.serve()
    finally:
        server.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        trio.run(_serve_async, args)


if __name__ == "__main__":
    main()
