import argparse
import logging

import trio

from parakeet.config import Config
from parakeet.profiler import add_profile_args, run_profile
from parakeet.server import ASRSocketServer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parakeet")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Start the streaming socket server.")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument(
        "--port", type=int, default=8765, help="Port to listen for TCP clients."
    )
    serve.add_argument(
        "--ws-port",
        type=int,
        default=8000,
        help="Port to listen for WebSocket clients.",
    )
    serve.add_argument("--status-port", type=int, default=8050)
    serve.add_argument("--device", default="cuda")
    serve.add_argument(
        "--model-size", choices=("small", "large"), default="small", help="Model size"
    )
    serve.add_argument(
        "--dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
        help="Model compute dtype.",
    )
    serve.add_argument(
        "--shape-log",
        default=None,
        help="Optional path to write unique tensor shape observations.",
    )
    serve.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="Enable CUDA graphs for encoder (fixed batch sizes).",
    )
    serve.add_argument(
        "--cuda-graphs-batch-sizes",
        default="1,2,4,8",
        help="Comma-separated batch sizes for CUDA graphs (e.g. 1,2,4).",
    )
    serve.add_argument("--max-num-streams", type=int, default=100)
    serve.add_argument("--sample-rate", type=int, default=16000)
    serve.add_argument("--max-stream-seconds", type=int, default=300)
    serve.add_argument("--pre-encode-batch-size", type=int, default=32)
    serve.add_argument("--encode-batch-size", type=int, default=32)
    serve.add_argument("--decode-batch-size", type=int, default=32)
    serve.add_argument(
        "--paged-kv-cache",
        action="store_true",
        help="Enable paged KV cache for encoder attention.",
    )
    serve.add_argument("--paged-kv-page-size", type=int, default=16)
    serve.add_argument("--paged-kv-max-pages", type=int, default=0)

    profile = subparsers.add_parser(
        "profile", help="Run a short profiling pass and export a Chrome trace."
    )
    add_profile_args(profile)
    return parser


def _print_config(
    config: Config,
    host: str,
    port: int,
    ws_port: int | None,
    status_port: int | None,
    device: str,
) -> None:
    print("Starting Parakeet streaming socket server")
    print(f"  host: {host}")
    print(f"  port: {port}")
    print(f"  ws_port: {ws_port}")
    print(f"  status_port: {status_port}")
    print(f"  device: {device}")
    print(f"  model_size: {config.size}")
    print(f"  dtype: {config.dtype}")
    if config.shape_log_path:
        print(f"  shape_log: {config.shape_log_path}")
    if config.cuda_graphs:
        print(f"  cuda_graphs: {config.cuda_graphs}")
        print(f"  cuda_graph_batch_sizes: {config.cuda_graph_batch_sizes}")
    print(f"  max_num_streams: {config.max_num_streams}")
    print(f"  sample_rate: {config.sample_rate}")
    print(f"  max_stream_seconds: {config.max_stream_seconds}")
    print(f"  pre_encode_batch_size: {config.pre_encode_batch_size}")
    print(f"  encode_batch_size: {config.encode_batch_size}")
    print(f"  decode_batch_size: {config.decode_batch_size}")
    print(f"  paged_kv_cache: {config.paged_kv_cache}")
    print(f"  paged_kv_page_size: {config.paged_kv_page_size}")
    print(f"  paged_kv_max_pages: {config.paged_kv_max_pages}")


async def _serve_async(args: argparse.Namespace) -> None:
    config = Config(
        size=args.model_size,
        max_num_streams=args.max_num_streams,
        sample_rate=args.sample_rate,
        max_stream_seconds=args.max_stream_seconds,
        dtype=args.dtype,
        shape_log_path=args.shape_log,
        cuda_graphs=args.cuda_graphs,
        cuda_graph_batch_sizes=tuple(
            int(x) for x in args.cuda_graphs_batch_sizes.split(",") if x.strip()
        ),
        pre_encode_batch_size=args.pre_encode_batch_size,
        encode_batch_size=args.encode_batch_size,
        decode_batch_size=args.decode_batch_size,
        paged_kv_cache=args.paged_kv_cache,
        paged_kv_page_size=args.paged_kv_page_size,
        paged_kv_max_pages=(
            None if args.paged_kv_max_pages <= 0 else args.paged_kv_max_pages
        ),
    )
    _print_config(
        config, args.host, args.port, args.ws_port, args.status_port, args.device
    )
    server = ASRSocketServer(
        config,
        host=args.host,
        tcp_port=args.port,
        ws_port=args.ws_port,
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
    if args.command == "profile":
        run_profile(args)


if __name__ == "__main__":
    main()
