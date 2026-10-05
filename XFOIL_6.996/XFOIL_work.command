#!/bin/zsh
# Compatibility entry point. The maintained interface is in the Python package.
set -euo pipefail
XFOIL_LAUNCH_DIR="${0:A:h}"
exec "$XFOIL_LAUNCH_DIR/../启动工作台.command" "$@"
