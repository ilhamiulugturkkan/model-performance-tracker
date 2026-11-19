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


# Finding a Threshold with Burs Experiment

**H0:** The real value behaves just like burst exp's data. If p < 0.05 we reject that and raise an alarm.

**Threshold (99.8% empirical):** the line the monitor actually uses. An alarm fires when the difference is above it.

**MDS:** a property of that threshold. It tells you how large a real drop must be for the threshold to catch it 80% of the time.

**r = 12 × (MDS12 / d)²** d is the wanted drop rate and r is the needed repeats to detect that drop rate. More repeats reduces the noise. Smaller d values needs more repeats. (In our system days = ceil(r/3)/2 )