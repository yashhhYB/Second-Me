#!/usr/bin/env python3
import os
import json
import argparse
import socket
import subprocess
import sys
import time

try:
    import torch
except ImportError:
    torch = None


# ---------- Utility Functions ----------

def check_file_exists(filepath, required=True):
    if os.path.exists(filepath):
        print(f"[✔] File found: {filepath}")
        return True
    else:
        msg = f"[✘] Missing required file: {filepath}" if required else f"[!] Optional file missing: {filepath}"
        print(msg)
        return False if required else True


def check_gpu_available():
    if torch:
        if torch.cuda.is_available():
            print(f"[✔] GPU Available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("[✘] GPU not available (torch.cuda.is_available() returned False)")
            return False
    else:
        print("[!] PyTorch not installed. GPU check skipped.")
        return False


def check_internet_connectivity(timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        print(" Internet connectivity confirmed")
        return True
    except Exception:
        print(" No internet connectivity")
        return False


def validate_json_config(path):
    if not os.path.exists(path):
        print(f"[✘] Config file not found: {path}")
        return False
    try:
        with open(path, "r") as f:
            json.load(f)
        print(f" Valid JSON config: {path}")
        return True
    except json.JSONDecodeError:
        print(f" Invalid JSON format in config: {path}")
        return False


def check_env_variable(var_name):
    value = os.getenv(var_name)
    if value:
        print(f" Environment variable set: {var_name}={value}")
        return True
    else:
        print(f" Environment variable not set: {var_name}")
        return False


def check_command_exists(command):
    try:
        subprocess.run(command.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f" Command available: {command}")
        return True
    except Exception:
        print(f" Command not found or failed: {command}")
        return False


# ---------- Main Checker ----------

def run_checks(args):
    print("\n===  Pre-Training Environment Check ===\n")

    all_ok = True

    # Check required files
    all_ok &= check_file_exists("resources/entities.parquet")
    all_ok &= check_file_exists("resources/memory_config.json")
    all_ok &= validate_json_config("resources/memory_config.json")

    # Optional (commonly missing)
    check_file_exists("resources/metadata.csv", required=False)

    # Check GPU
    if args.require_gpu:
        all_ok &= check_gpu_available()
    else:
        check_gpu_available()

    # Check Internet
    if args.check_internet:
        all_ok &= check_internet_connectivity()

    # Check environment variables
    check_env_variable("MODEL_PATH")
    check_env_variable("OLLAMA_HOST")

    # Command checks
    check_command_exists("ollama list")
    check_command_exists("python --version")

    # Docker/Integrated setup check
    if os.path.exists("docker-compose.yml"):
        print("[✔] Docker-based setup detected.")
    else:
        print("[✔] Integrated (non-Docker) setup detected.")

    print("\n Check completed.")

    if all_ok:
        print(" Environment looks good! You can start training safely.")
    else:
        print("⚠ Some checks failed. Please fix them before training.")

    return all_ok


# ---------- Entry Point ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-training environment validation script.")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if GPU is not available.")
    parser.add_argument("--check-internet", action="store_true", help="Check internet connectivity.")
    args = parser.parse_args()

    status = run_checks(args)
    sys.exit(0 if status else 1)
