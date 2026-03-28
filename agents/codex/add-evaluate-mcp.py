try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python <= 3.10

import tomli_w

import argparse
import os



def parse_args(*args, **kwargs):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--k-server-bench-home", type=str, default=os.getenv("K_SERVER_BENCH_HOME", None), help="Path to the k-server-bench repository")
    parser.add_argument("--evaluator-home", type=str, default=os.getenv("K_SERVER_BENCH_EVALUATOR_HOME", "tools/evaluator"), help="Relative or absolute path to evaluator home")
    parser.add_argument("--evaluator-name", type=str, default=os.getenv("K_SERVER_BENCH_EVALUATOR_NAME", "evaluate.py"))
    parser.add_argument("--mcp-executable-name", type=str, default=os.getenv("K_SERVER_BENCH_MCP_EXECUTABLE_NAME", "evaluate_mcp_server.py"))

    parser.add_argument("--codex-home", type=str, default=os.getenv("CODEX_HOME", None), help="Path to the codex home")
    parser.add_argument("--startup-timeout-sec", type=int, default=20, help="Startup timeout in seconds")
    parser.add_argument("--tool-timeout-sec", type=int, default=3600, help="Tool timeout in seconds")
    parser.add_argument("--mcp-server-name", type=str, default="evaluate", help="Name of the MCP server")
    parser.add_argument("--mcp-server-command", type=str, default="python3", help="Command to run the MCP server")
    parser.add_argument(
        "--args",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Full MCP server 'args' list. Defaults to [<evaluate_mcp_server.py>]. "
            "If you override this, do not forget to include the evaluator MCP executable path yourself."
        ),
    )
    parser.add_argument("--create-config", action="store_true", help="Create a config.toml file if it does not exist")
    parser.add_argument("--override-mcp-config", action="store_true", help="Override the MCP server config if it already exists")
    parser.add_argument("--env", type=str, default=[], nargs="+", help="Additional environment variables to set for the MCP server. (e.g. --env K_SERVER_EVALUATE_METRICS_NAMES=... --env K_SERVER_EVALUATE_SEARCH_TIMEOUT=...)")
    
    return parser.parse_args(*args, **kwargs)

def main():
    args = parse_args()

    if args.codex_home is None:
        print("CODEX_HOME is not set")
        print("CODEX_HOME should point to the codex home")
        exit(1)

    if args.k_server_bench_home is None and not os.path.isabs(args.evaluator_home):
        print("K_SERVER_BENCH_HOME must be set when --evaluator-home is relative")
        print("K_SERVER_BENCH_HOME should point to the k-server-bench repository")
        exit(1)

    if os.path.isabs(args.evaluator_home):
        args.evaluator_home = os.path.abspath(args.evaluator_home)
    else:
        args.evaluator_home = os.path.abspath(os.path.join(args.k_server_bench_home, args.evaluator_home))
    args.codex_home = os.path.abspath(args.codex_home)
    mcp_executable_path = os.path.join(args.evaluator_home, args.mcp_executable_name)

    print(args)

    if not args.codex_home.endswith(".codex"):
        print("WARNING: CODEX_HOME usually ends with .codex")

    if not os.path.exists(os.path.join(args.codex_home, "config.toml")):
        print(f"WARNING: config.toml does not exist at {os.path.join(args.codex_home, 'config.toml')}")
        if args.create_config:
            print("Creating config.toml due --create-config=True")
            os.makedirs(args.codex_home, exist_ok=True)
            with open(os.path.join(args.codex_home, "config.toml"), "w") as f:
                pass
        else:
            print("Skipping due to --create-config=False")
            exit(1)

    with open(os.path.join(args.codex_home, "config.toml"), "rb") as f:
        config = tomllib.load(f)

    print("Current config:")
    print(config)
    print("")

    mcp_env = {}
    for env in args.env:
        try:
            key, value = env.split("=")
            mcp_env[key] = value
        except ValueError:
            print(f"WARNING: Invalid environment variable: {env}")
            print("Skipping...")
            continue

    evaluate_mcp_server = {
        "command": args.mcp_server_command,
        "startup_timeout_sec": args.startup_timeout_sec,
        "tool_timeout_sec": args.tool_timeout_sec,
        "args": args.args if args.args is not None else [mcp_executable_path],
        "env": mcp_env
    }

    if config.get("mcp_servers", {}).get(args.mcp_server_name, None) is not None and not args.override_mcp_config:
        print(f"WARNING: MCP server '{args.mcp_server_name}' already exists")
        print("Exiting...")
        exit(1)

    if "mcp_servers" not in config:
        config["mcp_servers"] = {}
    
    config["mcp_servers"][args.mcp_server_name] = evaluate_mcp_server

    print(f"Resulting '{args.mcp_server_name}' config:")
    print(evaluate_mcp_server)

    with open(os.path.join(args.codex_home, "config.toml"), "wb") as f:
        tomli_w.dump(config, f)


if __name__ == "__main__":
    main()
