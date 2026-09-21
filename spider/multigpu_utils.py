# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for launching one vLLM server per GPU.

Used only by the local-model runner (run_vllm.py); the hosted-API runner does
not import this module.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

BASE_PORT = 8100


def start_vllm_server(gpu_id: int, port: int, model: str,
                      max_model_len: int, gpu_mem_util: float) -> subprocess.Popen:
    """Launch a vLLM server on a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
    ]
    # Put logs + CWD on a LOCAL disk, never on a network mount. Every `python -m`
    # import does a readdir() on sys.path[0] (= CWD); several vLLM processes
    # racing on network-filesystem inode locks can hang startup indefinitely.
    # Override with VLLM_LOG_DIR if /tmp is not local on your machine.
    local_dir = Path(os.environ.get("VLLM_LOG_DIR", "/tmp/spider-vllm-logs"))
    local_dir.mkdir(parents=True, exist_ok=True)
    log_file = local_dir / f"vllm_gpu{gpu_id}.log"
    fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(local_dir))
    logger.info(f"Started vLLM on GPU {gpu_id}, port {port}, PID {proc.pid} (log: {log_file})")
    return proc


def wait_for_server(port: int, timeout: int = 300) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY").models.list()
            return True
        except Exception:
            time.sleep(5)
    return False


def stop_servers(procs: list):
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def launch_all_servers(gpus: List[int], model: str, max_model_len: int,
                       gpu_mem_util: float, base_port: int = BASE_PORT,
                       skip_launch: bool = False,
                       startup_timeout: int = 300) -> Tuple[List[subprocess.Popen], List[int]]:
    """Launch vLLM servers on all GPUs and wait for readiness.

    Returns: (processes, ports). If skip_launch=True, returns ([], ports) assuming servers are up.
    """
    ports = [base_port + i for i in range(len(gpus))]
    if skip_launch:
        logger.info(f"Assuming servers already running on ports {ports}")
        # Still verify each is up
        for gpu_id, port in zip(gpus, ports):
            if not wait_for_server(port, timeout=10):
                raise RuntimeError(f"Server on port {port} (GPU {gpu_id}) not responsive")
        return [], ports

    procs = []
    logger.info("Launching vLLM servers...")
    for gpu_id, port in zip(gpus, ports):
        procs.append(start_vllm_server(gpu_id, port, model, max_model_len, gpu_mem_util))

    logger.info(f"Waiting for all servers to be ready (timeout={startup_timeout}s per GPU)...")
    for gpu_id, port, proc in zip(gpus, ports, procs):
        if wait_for_server(port, timeout=startup_timeout):
            logger.info(f"  GPU {gpu_id} (port {port}): ready")
        else:
            logger.error(f"  GPU {gpu_id} (port {port}): FAILED")
            stop_servers(procs)
            raise RuntimeError(f"vLLM server on GPU {gpu_id} failed to start "
                               f"within {startup_timeout}s")
    logger.info("All servers ready.")
    return procs, ports


def shard_instances(instance_ids: List[str], num_shards: int) -> List[List[str]]:
    """Round-robin split instances across shards."""
    shards = [[] for _ in range(num_shards)]
    for i, iid in enumerate(instance_ids):
        shards[i % num_shards].append(iid)
    return shards
