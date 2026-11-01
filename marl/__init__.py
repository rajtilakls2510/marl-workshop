"""Shared helpers for the MARL workshop."""
from marl.utils import ensure_display

# VMAS renders with pyglet, which needs an X display from the moment it is first loaded. Inside the Docker image
# (also in Codespaces / VS Code kernels that did not inherit the entrypoint's DISPLAY) this points to - or
# starts - the virtual display. On a normal desktop it does nothing.
ensure_display()
