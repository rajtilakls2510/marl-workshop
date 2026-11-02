# Route A — GitHub Codespaces (no download)

A codespace is a small Linux machine in the cloud with VS Code in your browser. Ours starts from the workshop image:
Python, PyTorch, all the environments and the trained agents are already inside. Nothing is installed on your laptop,
so it works on any machine - Windows, Linux, any Mac (including Apple M-series), even a tablet.

## 1. A GitHub account
Sign in at [github.com](https://github.com), or create a free account. Your codespace runs on your account's
**free monthly allowance** (120 core-hours = 60 hours on the default 2-core machine), far more than the workshop needs.

## 2. Create your codespace
Open **<https://codespaces.new/rajtilakls2510/marl-workshop>** and click **Create codespace**.

(Same thing from the repository page: green **Code** button -> **Codespaces** tab -> **Create codespace on main**.)

The first start downloads the workshop image (about 2 GB) on GitHub's side: expect **a few minutes**.
Start it at the beginning of the session, while the introduction is running. A VS Code window then opens in your browser,
with this repository at `/workspaces/marl-workshop`.

Every attendee creates their own codespace: it is private to you, and nothing you change affects anyone else.

## 3. Extensions
The **Python** and **Jupyter** extensions (both by Microsoft) are installed automatically.
Check in the Extensions panel (left sidebar, the icon with four squares). If one is missing: search for it and click **Install**,
then reload the window (Ctrl+Shift+P -> `Developer: Reload Window`).

## 4. Open a notebook and choose the Python environment
1. In the file explorer, open `notebooks/00_check_setup.ipynb`.
2. Top right of the notebook: **Select Kernel** -> **Python Environments...** -> **`venv`** (`/opt/venv/bin/python`, Python 3.10).
   This is the workshop's environment; do not pick any other Python.
3. Click **Run All**. All cells green = ready.

Each notebook remembers the kernel; for a new notebook, pick `venv` once more if VS Code asks.
After restarting a kernel, always run the notebook **from the top** (Run All): later cells use what the first cells import.

## 5. (Optional) JupyterLab instead of VS Code notebooks
The codespace also runs JupyterLab. Bottom panel -> **PORTS** tab -> port **8888** ("JupyterLab") -> click the globe icon.
Token: `marl`.

## 6. When you are done
Go to [github.com/codespaces](https://github.com/codespaces) -> **...** next to your codespace -> **Stop codespace**
(it also stops by itself after 30 minutes without activity). **Delete** it when you no longer need it: a stored codespace
counts against your free storage. Your edits to the notebooks live only in that codespace - download anything you want to keep.
