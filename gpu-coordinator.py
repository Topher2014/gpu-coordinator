#!/usr/bin/env python3
# /usr/local/bin/gpu-coordinator
"""
GPU Coordinator - Automatically manages vLLM service when other GPU processes run.
Monitors for processes that need exclusive GPU access and pauses vLLM accordingly.
"""

import time
import subprocess
import psutil
import logging
import signal
import sys
from typing import Set

class GPUCoordinator:
    def __init__(self):
        # Configuration - edit these defaults as needed
        self.vllm_service = 'vllm.service'
        self.check_interval = 3  # seconds between checks
        self.grace_period = 8    # seconds before stopping vLLM
        self.log_level = 'INFO'
        
        # Processes that need exclusive GPU access
        self.gpu_intensive_processes = {
            'rdb',
            'python -m rdb',
            'embedding',
            'indexing',
            'trainer',
            'finetune',
        }
        
        self.vllm_was_running = False
        self.gpu_process_start_time = None
        self.logger = self._setup_logging()
        self.running = True
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('gpu-coordinator')
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, self.log_level))
        return logger
    
    def _is_process_gpu_intensive(self, proc: psutil.Process) -> bool:
        """Check if a process is GPU-intensive and needs exclusive access."""
        try:
            cmdline = ' '.join(proc.cmdline())
            
            # Check against known GPU-intensive process patterns
            for pattern in self.gpu_intensive_processes:
                if pattern in cmdline:
                    return True
            
            # Check for specific command patterns
            gpu_keywords = ['embed', 'index', 'build', 'train', 'finetune']
            if any(keyword in cmdline.lower() for keyword in gpu_keywords):
                return True
                
            return False
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    def _get_gpu_intensive_processes(self) -> Set[psutil.Process]:
        """Get all currently running GPU-intensive processes."""
        gpu_processes = set()
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if self._is_process_gpu_intensive(proc):
                    gpu_processes.add(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return gpu_processes
    
    def _is_vllm_running(self) -> bool:
        """Check if vLLM service is running."""
        try:
            result = subprocess.run(
                ['systemctl', 'is-active', self.vllm_service],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _stop_vllm(self):
        """Stop vLLM service."""
        if self._is_vllm_running():
            self.logger.info(f"Stopping {self.vllm_service} for GPU-intensive process")
            try:
                subprocess.run(['sudo', 'systemctl', 'stop', self.vllm_service], check=True)
                self.vllm_was_running = True
                # Give time for GPU memory to be freed
                time.sleep(3)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to stop {self.vllm_service}: {e}")
    
    def _start_vllm(self):
        """Start vLLM service if we stopped it."""
        if self.vllm_was_running:
            self.logger.info(f"Restarting {self.vllm_service}")
            try:
                subprocess.run(['sudo', 'systemctl', 'start', self.vllm_service], check=True)
                self.vllm_was_running = False
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to start {self.vllm_service}: {e}")
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def run(self):
        """Main monitoring loop."""
        self.logger.info("GPU Coordinator started")
        self.logger.info(f"Monitoring for GPU processes: {list(self.gpu_intensive_processes)}")
        self.logger.info(f"Managing service: {self.vllm_service}")
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        
        try:
            while self.running:
                gpu_processes = self._get_gpu_intensive_processes()
                
                if gpu_processes:
                    # GPU-intensive process detected
                    if self.gpu_process_start_time is None:
                        self.gpu_process_start_time = time.time()
                        proc_names = [proc.name() for proc in gpu_processes]
                        self.logger.info(f"GPU-intensive processes detected: {proc_names}")
                    
                    # Check if grace period has elapsed
                    elapsed = time.time() - self.gpu_process_start_time
                    if elapsed >= self.grace_period and self._is_vllm_running():
                        self._stop_vllm()
                
                else:
                    # No GPU-intensive processes
                    if self.gpu_process_start_time is not None:
                        self.logger.info("GPU-intensive processes finished")
                        self.gpu_process_start_time = None
                        
                        # Restart vLLM if we stopped it
                        if self.vllm_was_running:
                            # Wait a bit for processes to fully exit
                            time.sleep(2)
                            self._start_vllm()
                
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            # Cleanup: restart vLLM if we stopped it
            if self.vllm_was_running:
                self.logger.info("Cleanup: restarting vLLM service")
                self._start_vllm()
            
            self.logger.info("GPU Coordinator stopped")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("""GPU Coordinator - Automatic vLLM service management

Monitors for GPU-intensive processes and automatically pauses vLLM service
to prevent GPU memory conflicts. Restarts vLLM when processes complete.

To customize settings, edit the defaults at the top of this script.
""")
        return
    
    coordinator = GPUCoordinator()
    coordinator.run()


if __name__ == "__main__":
    main()
    echo ""
    echo "REQUIRED: Install ML dependencies manually:"
    echo ""
    echo "Option 1 - Using pip (recommended):"
    echo "  pip install --user faiss-cpu" 
    echo ""
    echo "Option 2 - Using conda (alternative):"
    echo "  conda install -c conda-forge faiss-cpu"
    echo ""
    echo "Note: The faiss AUR package is currently broken (known issue)"
    echo "Run 'rdb status' after installing dependencies to verify setup"
    echo ""
}

