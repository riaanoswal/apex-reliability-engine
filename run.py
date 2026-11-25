#!/usr/bin/env python3
"""
APEX Industrial AI Reliability Engine - single-command launcher.

    python run.py

Starts the FastAPI backend and the React dashboard, waits for both to come
up, then opens the dashboard in your browser. Ctrl+C stops everything.

Options:
    python run.py --no-browser     start the services, don't open a browser
    python run.py --skip-install   don't check or install dependencies
    python run.py --port 3100      ask the dashboard for a specific port
    python run.py --api-port 8100  ask the backend for a specific port
"""

import argparse
import atexit
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
IS_WIN = os.name == "nt"
PY_PACKAGES = ("fastapi", "uvicorn", "numpy", "scipy", "sklearn", "pydantic")

# ----------------------------------------------------------------- output


class C:
    """ANSI colours, disabled when the terminal cannot show them."""

    on = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
    if IS_WIN and on:
        try:  # make the Windows console interpret escapes
            import ctypes

            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            on = False

    @classmethod
    def _w(cls, code, s):
        return f"\033[{code}m{s}\033[0m" if cls.on else s

    @classmethod
    def dim(cls, s):
        return cls._w("2", s)

    @classmethod
    def bold(cls, s):
        return cls._w("1", s)

    @classmethod
    def green(cls, s):
        return cls._w("32", s)

    @classmethod
    def red(cls, s):
        return cls._w("31", s)

    @classmethod
    def yellow(cls, s):
        return cls._w("33", s)

    @classmethod
    def cyan(cls, s):
        return cls._w("36", s)


def step(n, total, msg):
    print(f"{C.cyan(f'[{n}/{total}]')} {msg}", flush=True)


def ok(msg):
    print(f"      {C.green('OK')}  {msg}", flush=True)


def warn(msg):
    print(f"      {C.yellow('!')}   {msg}", flush=True)


def die(msg, *hints):
    print(f"\n{C.red('ERROR')}  {msg}", flush=True)
    for h in hints:
        print(f"       {h}", flush=True)
    print(flush=True)
    sys.exit(1)


# ----------------------------------------------------------------- helpers


def free_port(start, limit=40):
    """First free TCP port at or after `start`."""
    for p in range(start, start + limit):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    die(f"No free port between {start} and {start + limit - 1}.")


def port_busy(p):
    with socket.socket() as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", p)) == 0


def npm_cmd():
    """npm is a .cmd shim on Windows, which needs resolving to run without a shell."""
    for name in (("npm.cmd", "npm") if IS_WIN else ("npm",)):
        p = shutil.which(name)
        if p:
            return p
    return None


def http_ok(url, timeout=1.5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 500
    except urllib.error.HTTPError:
        return True  # it answered, which is all we need to know
    except Exception:
        return False


# ----------------------------------------------------------------- children

CHILDREN = []


def spawn(cmd, cwd, pipe=False):
    kw = dict(cwd=cwd, env=dict(os.environ))
    if pipe:
        kw.update(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
    if IS_WIN:
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    p = subprocess.Popen(cmd, **kw)
    CHILDREN.append(p)
    return p


def shutdown():
    for p in CHILDREN:
        if p.poll() is not None:
            continue
        try:
            if IS_WIN:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(p.pid)],
                    capture_output=True,
                )
            else:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    deadline = time.time() + 5
    for p in CHILDREN:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


atexit.register(shutdown)


def _signal_shutdown(signum, frame):
    """Take the services down when this launcher is killed or its window closes.

    atexit does not fire on SIGTERM / console-close, so without this the backend
    and the dashboard would be orphaned and keep holding their ports.
    """
    print(f"\n{C.dim('Stopping APEX...')}", flush=True)
    shutdown()
    print(C.dim("Stopped."), flush=True)
    os._exit(0)


for _sig in ("SIGTERM", "SIGHUP", "SIGBREAK"):
    _s = getattr(signal, _sig, None)
    if _s is not None:
        try:
            signal.signal(_s, _signal_shutdown)
        except (ValueError, OSError):
            pass  # not available on this platform / not the main thread


def relay(proc, label, colour, sink=None):
    """Print a child's output with a label; optionally tee lines to `sink`."""

    def run():
        tag = colour(f"{label:9}")
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if sink is not None:
                sink.append(line)
            if line:
                print(f"{tag}{C.dim('|')} {line}", flush=True)
        proc.stdout.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


# ----------------------------------------------------------------- install


def ensure_python_packages():
    missing = []
    for mod in PY_PACKAGES:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if not missing:
        ok("Python packages present.")
        return
    req = os.path.join(ROOT, "backend", "requirements.txt")
    if not os.path.isfile(req):
        die(
            f"Missing {req}",
            "Run this script from the folder you extracted, with backend/ next to it.",
        )
    print(f"      installing: {', '.join(missing)} {C.dim('(first run only)')}", flush=True)
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", req], cwd=ROOT
    )
    if r.returncode != 0:
        die(
            "Python package install failed.",
            "Run this by hand to see the reason:",
            f'  "{sys.executable}" -m pip install -r backend{os.sep}requirements.txt',
        )
    ok("Python packages installed.")


def ensure_node_packages(npm):
    mods = os.path.join(ROOT, "frontend", "node_modules")
    if os.path.isdir(mods) and os.path.isdir(os.path.join(mods, "vite")):
        ok("Dashboard packages present.")
        return
    print(
        f"      running npm install {C.dim('(first run only, this takes a few minutes)')}",
        flush=True,
    )
    r = subprocess.run([npm, "install", "--no-fund", "--no-audit"],
                       cwd=os.path.join(ROOT, "frontend"))
    if r.returncode != 0 or not os.path.isdir(mods):
        die(
            "npm install failed.",
            "Run it by hand to see the reason:",
            f"  cd frontend  &&  npm install",
        )
    ok("Dashboard packages installed.")


# ----------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--port", type=int, default=3000, help="dashboard port")
    ap.add_argument("--api-port", type=int, default=8000, help="backend port")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()

    print()
    print(C.bold("  APEX Industrial AI Reliability Engine"))
    print(C.dim("  ------------------------------------------------------------"))
    print()

    for need in ("backend/server.py", "frontend/package.json"):
        if not os.path.isfile(os.path.join(ROOT, *need.split("/"))):
            die(
                f"Cannot find {need}",
                f"run.py must sit in the extracted folder, next to backend/ and frontend/.",
                f"It is currently in: {ROOT}",
            )

    total = 4 if args.skip_install else 5
    n = 0

    if not args.skip_install:
        n += 1
        step(n, total, "Checking dependencies")
        ensure_python_packages()
        npm = npm_cmd()
        if not npm:
            die(
                "Node.js was not found.",
                "Install it from https://nodejs.org, reopen this terminal, try again.",
            )
        ensure_node_packages(npm)
    else:
        npm = npm_cmd()
        if not npm:
            die("Node.js was not found.", "Install it from https://nodejs.org.")

    # ---- ports
    n += 1
    step(n, total, "Choosing ports")
    api_port = args.api_port
    if port_busy(api_port):
        new = free_port(api_port + 1)
        warn(f"Port {api_port} is in use; the backend will use {new} instead.")
        api_port = new
    web_port = args.port
    if port_busy(web_port):
        new = free_port(web_port + 1)
        warn(f"Port {web_port} is in use; the dashboard will use {new} instead.")
        web_port = new
    ok(f"backend {api_port}, dashboard {web_port}")

    # ---- backend
    n += 1
    step(n, total, "Starting the backend")
    be = spawn(
        [sys.executable, "-u", "-m", "uvicorn", "backend.server:app",
         "--host", "127.0.0.1", "--port", str(api_port)],
        cwd=ROOT,
        pipe=True,
    )
    relay(be, "backend", C.cyan)

    api = f"http://127.0.0.1:{api_port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        if be.poll() is not None:
            die("The backend stopped while starting. Its output is above.")
        if http_ok(f"{api}/api/telemetry"):
            break
        time.sleep(0.4)
    else:
        die("The backend did not answer within 60 seconds. Its output is above.")
    ok(f"{api}/api/telemetry")

    # ---- dashboard
    n += 1
    step(n, total, "Starting the dashboard")
    env_port = dict(os.environ)
    env_port["APEX_API"] = api  # vite.config.ts proxies /api here
    fe = subprocess.Popen(
        [npm, "run", "dev", "--", "--port", str(web_port), "--strictPort"],
        cwd=os.path.join(ROOT, "frontend"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
        env=env_port,
        **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WIN
           else {"start_new_session": True}),
    )
    CHILDREN.append(fe)
    lines = []
    relay(fe, "dashboard", C.green, sink=lines)

    url = None
    deadline = time.time() + 180
    while time.time() < deadline:
        if fe.poll() is not None:
            die("The dashboard stopped while starting. Its output is above.")
        for ln in list(lines):
            m = re.search(r"https?://(?:localhost|127\.0\.0\.1):(\d+)", ln)
            if m:
                url = f"http://localhost:{m.group(1)}"
                break
        if url and http_ok(url, timeout=2):
            break
        if url is None and http_ok(f"http://127.0.0.1:{web_port}/", timeout=1):
            url = f"http://localhost:{web_port}"
            break
        time.sleep(0.4)
    if not url:
        die("The dashboard did not come up within 3 minutes. Its output is above.")
    ok(url)

    # ---- browser
    n += 1
    step(n, total, "Opening the dashboard")
    if args.no_browser:
        ok("skipped (--no-browser)")
    else:
        try:
            webbrowser.open(url)
            ok("browser opened")
        except Exception:
            warn(f"Could not open a browser. Go to {url} yourself.")

    print()
    print(C.dim("  ------------------------------------------------------------"))
    print(f"  {C.bold('APEX is running.')}")
    print(f"    Dashboard  {C.bold(url)}")
    print(f"    API        {api}/api/telemetry")
    print()
    print(f"  {C.dim('Press Ctrl+C to stop both services.')}")
    print(C.dim("  ------------------------------------------------------------"))
    print()

    # ---- hold until Ctrl+C or a child dies
    try:
        while True:
            if be.poll() is not None:
                print(f"\n{C.red('The backend stopped.')} Shutting down.", flush=True)
                return 1
            if fe.poll() is not None:
                print(f"\n{C.red('The dashboard stopped.')} Shutting down.", flush=True)
                return 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(f"\n{C.dim('Stopping APEX...')}", flush=True)
        shutdown()
        print(C.dim("Stopped."), flush=True)
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
