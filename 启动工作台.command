#!/bin/zsh
# One daily entry point; arguments still reach the full compatible CLI.
set -euo pipefail
XFOIL_PROJECT_DIR="${0:A:h}"
cd "$XFOIL_PROJECT_DIR"
if (( $# == 0 )); then
  set -- --workbench
fi
candidates=("${XFOIL_PYTHON:-}" /opt/anaconda3/envs/xfoil-work/bin/python "$HOME/miniconda3/envs/xfoil-work/bin/python" "$HOME/anaconda3/envs/xfoil-work/bin/python" /opt/homebrew/bin/python3 /usr/bin/python3)
for candidate in "${candidates[@]}"; do
  if [[ -n "$candidate" && -x "$candidate" ]] && "$candidate" -c 'import numpy, matplotlib' >/dev/null 2>&1; then
    exec "$candidate" "$XFOIL_PROJECT_DIR/xfoil_work.py" "$@"
  fi
done
print -u2 '没有找到带 numpy / matplotlib 的 Python。请按 docs/MACOS_SETUP.md 创建 xfoil-work 环境，或设置 XFOIL_PYTHON。'
read 'reply?按回车关闭…'
exit 1
