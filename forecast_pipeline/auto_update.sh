#!/usr/bin/env bash
#
# Weekly automated influenza data update.
#
#   WHO fluID download -> per-country extraction -> inference (frozen best model)
#   -> frontend JSON -> deploy
#
# Safety properties:
#   - set -euo pipefail: any failing step aborts the run.
#   - A failing run NEVER deploys: the live site keeps the last good data.
#   - The WHO guard aborts if the fresh file is older or drastically smaller
#     than the backup (protects against partial WHO releases).
#   - Catch-up safe: if a week is missed, the next run picks up all new weeks.
#
# Log: auto_update.log (this directory).
set -euo pipefail
# cron's PATH lacks conda; make sure python3/torch resolve to the project environment
[ -d /home/david/miniconda3/bin ] && export PATH="/home/david/miniconda3/bin:$PATH"
cd "$(dirname "$0")"

LOG="auto_update.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M')] $*" | tee -a "$LOG"; }
trap 'log "!!! AUTO UPDATE FAILED — see log above. Live site keeps last good data."' ERR

PIPELINE_DIR="$(pwd)"          # this script's directory (no package.json here — matters for step 6)
FRONTEND_DIR="$(cd ../frontend && pwd)"

log "=== auto update start ==="

# 1. WHO download (backs up the previous CSV; exits non-zero on failure)
python3 update_who_data.py --backup 2>&1 | tee -a "$LOG"

# 2. WHO guard: fresh data must not be older / drastically smaller than the backup.
#    (heredoc on the same command as the output redirect; stdout goes to the log)
python3 - >>"$LOG" 2>&1 <<'EOF'
import os
import sys
import pandas as pd

new_path = "data/who_flu_data.csv"
bak_path = "data/who_flu_data.csv.bak"
new = pd.read_csv(new_path, usecols=["ISO_WEEKSTARTDATE"], low_memory=False)
if not os.path.exists(bak_path):
    print("guard: no previous file to compare against — skipping")
    sys.exit(0)
prev = pd.read_csv(bak_path, usecols=["ISO_WEEKSTARTDATE"], low_memory=False)

new_max = str(new["ISO_WEEKSTARTDATE"].dropna().max())
prev_max = str(prev["ISO_WEEKSTARTDATE"].dropna().max())
print(f"guard: latest week new={new_max} previous={prev_max} rows {len(prev)} -> {len(new)}")
if new_max < prev_max:
    sys.exit(f"aborting: new data ends BEFORE previous data ({new_max} < {prev_max}) — possible partial WHO release")
if len(new) < 0.5 * len(prev):
    sys.exit(f"aborting: row count dropped from {len(prev)} to {len(new)} — suspicious WHO file")
EOF
log "WHO guard passed"

# 3. Per-country extraction (data/extracted_data/*_combined.csv)
python3 enhanced_extract_country_data.py --all 2>&1 | tee -a "$LOG"

# 4. Inference with the frozen best model (hemisphere covariate, matching its training).
#    Use the GPU when at least 4 GiB is free, otherwise fall back to CPU
#    (e.g. while vLLM or another job occupies the GPU).
INFER_ARGS=(--model_path models/ablation/hemi/finetuned-ckpt
            --countries_file data/training_countries.json
            --covariates hemisphere)
FREE_MIB=""
if command -v nvidia-smi >/dev/null 2>&1; then
  FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
fi
if [ -n "$FREE_MIB" ] && [ "$FREE_MIB" -ge 4096 ]; then
  log "inference on GPU (${FREE_MIB} MiB free)"
  INFER_ARGS+=(--device cuda)
else
  log "GPU unavailable or busy (free: ${FREE_MIB:-n/a} MiB) — inference on CPU"
  INFER_ARGS+=(--device cpu)
fi
python3 run_all_country_inference.py "${INFER_ARGS[@]}" 2>&1 | tee -a "$LOG"

# 5. Frontend JSON (map + per-country details)
python3 generate_country_details.py 2>&1 | tee -a "$LOG"
python3 generate_map_data.py 2>&1 | tee -a "$LOG"

# 6. Deploy to Cloudflare Pages (static export) -> https://influenza-dashboard.pages.dev
#    Auth: CLOUDFLARE_API_TOKEN in the environment, else the token file
#    ~/.cloudflare_api_token (chmod 600). If neither exists the deploy is
#    skipped (with a warning) and the run still counts as successful.
#    Gotchas baked in:
#      - run wrangler from a NON-Next.js directory: from frontend/ its
#        framework detection would try to `npm install` @cloudflare/next-on-pages
#        and crash on the project's react-simple-maps/React-19 peer conflict.
#      - TMPDIR must be a writable scratch dir (on this machine /tmp contains
#        an unreadable snap directory that breaks wrangler's temp-dir discovery).
#      - NO --force: the Pages project was created once via the REST API
#        (2026-08-23) and subsequent deploys go straight to classic Pages.
#        If the project is ever deleted, recreate it first:
#          curl -X POST https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/pages/projects \
#            -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" -H "Content-Type: application/json" \
#            -d '{"name":"influenza-dashboard","production_branch":"main"}'
cd "$FRONTEND_DIR"
[ -z "${CLOUDFLARE_API_TOKEN:-}" ] && [ -f "$HOME/.cloudflare_api_token" ] && export CLOUDFLARE_API_TOKEN="$(cat "$HOME/.cloudflare_api_token")"
if [ -n "${CLOUDFLARE_API_TOKEN:-}" ] || [ -f "$HOME/.wrangler/config/default.toml" ]; then
  export TMPDIR="${TMPDIR:-$HOME/.wrangler-tmp}"
  mkdir -p "$TMPDIR"
  log "building static export and deploying to Cloudflare Pages"
  STATIC_EXPORT=1 NEXT_TELEMETRY_DISABLED=1 ./node_modules/.bin/next build >>"$LOG" 2>&1
  cd "$PIPELINE_DIR"
  "$FRONTEND_DIR/node_modules/.bin/wrangler" pages deploy "$FRONTEND_DIR/out" --project-name=influenza-dashboard >>"$LOG" 2>&1
  log "deployed https://influenza-dashboard.pages.dev"
else
  log "WARN: no Cloudflare auth (CLOUDFLARE_API_TOKEN / ~/.cloudflare_api_token / ~/.wrangler) — deploy skipped"
fi
log "=== auto update complete ==="
