#!/usr/bin/env python3
"""
Automated dependency installer for Amazon ML Challenge 2026 Entity Resolution Suite.
Prefers uv for instant installations, falling back to standard pip.
"""

import sys
import subprocess
import shutil

def main():
    print("==================================================")
    print(" Amazon ML Challenge 2026 - Dependency Installer")
    print("==================================================")
    
    use_uv = shutil.which("uv") is not None
    cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    if use_uv:
        print("[*] 'uv' detected! Using uv pip for high-speed package installation.")
        cmd = ["uv", "pip", "install", "-r", "requirements.txt"]
    else:
        print("[*] Using standard pip to install requirements.")

    try:
        subprocess.check_call(cmd)
        print("\n[✓] All dependencies successfully installed!")
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Installation failed with error code: {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
