# GPU Coordinator

Automatically manages vLLM service when other programs need GPU access.

## What it does

- Monitors for GPU-intensive processes (like RDB indexing)
- Temporarily stops vLLM to free GPU memory
- Restarts vLLM when GPU processes finish

## Installation

```bash
sudo cp gpu-coordinator /usr/local/bin/
sudo chmod +x /usr/local/bin/gpu-coordinator
sudo cp gpu-coordinator.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gpu-coordinator.service
sudo systemctl start gpu-coordinator.service
```

## Usage

Just run your GPU programs normally:

```bash
rdb search "how do i connect to wifi?" # vLLM pauses automatically 
```
vLLM automatically resumes when finished.
