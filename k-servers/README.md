# k-servers

Install this package from inside [`k-server-bench/k-servers`](/home/brillian/rl-k-servers/k-server-bench/k-servers), for example with:

```bash
pip install -e .
```

After installation, the top-level scripts in [`k-server-bench/scripts`](/home/brillian/rl-k-servers/k-server-bench/scripts) can import `kserverclean` directly.

## Known Issues

- `parallel_bfs_exploration` can show periodic throughput drops during long runs due to Python garbage collection (GC) pauses in the main process.
- Disabling GC (`--disable-gc-during-run`) can reduce pause spikes, but usually causes faster memory growth.
- For now, prefer leaving GC enabled when memory stability is more important than peak/instant throughput.
