# Why a Docker container needs two clocks

- `docker run` is only a client. The container is managed by the Docker daemon.
- Killing the client (what `subprocess.run(timeout=...)` does) leaves the container running. I saw it myself with `kill -9` and `docker ps`.
- `--rm` only deletes a container after it exits, so it doesn't help here.

**Two clocks**
1. **Inner:** `timeout -k 1 N python -` runs inside the container and ends the program by itself (exit code 124).
2. **Outer:** `subprocess.run(timeout=N+5)`, then `docker kill <name>` as a backup.

**Plain `subprocess` needs only one clock,** because killing the child kills the program.

**Rule:** a limit only works if it's enforced where the thing happens.

**Not checked yet:** exit code 124 from `timeout` in the image, and `docker kill`.