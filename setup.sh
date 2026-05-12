#!/bin/bash
# ============================================================
#  PFFT GPU Miner Setup Script
#  Tested on: Ubuntu 22.04, Vast.ai (CUDA 12.x)
# ============================================================

set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}"
echo "  ██████╗ ███████╗███████╗████████╗"
echo "  ██╔══██╗██╔════╝██╔════╝╚══██╔══╝"
echo "  ██████╔╝█████╗  █████╗     ██║   "
echo "  ██╔═══╝ ██╔══╝  ██╔══╝     ██║   "
echo "  ██║     ██║     ██║        ██║   "
echo "  ╚═╝     ╚═╝     ╚═╝        ╚═╝   "
echo -e "  PFFT GPU Miner Setup${NC}"
echo ""

# ── Args ─────────────────────────────────────────────────────
RPC_URL="${1:-}"
PRIVATE_KEY="${2:-}"
GPU_ID="${3:-0}"

if [ -z "$RPC_URL" ] || [ -z "$PRIVATE_KEY" ]; then
  echo -e "${YELLOW}Usage: ./setup.sh <RPC_URL> <PRIVATE_KEY> [GPU_ID]${NC}"
  echo ""
  echo "  RPC_URL     : Ethereum RPC (Alchemy/Infura/etc)"
  echo "                e.g. https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"
  echo "  PRIVATE_KEY : Wallet private key (0x...)"
  echo "  GPU_ID      : GPU index (default: 0)"
  echo ""
  echo -e "${RED}Exiting — RPC_URL and PRIVATE_KEY required.${NC}"
  exit 1
fi

# ── Detect GPU arch ──────────────────────────────────────────
echo -e "${CYAN}[1/4] Detecting GPU...${NC}"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
echo "  GPU: $GPU_NAME"

# Map GPU to CUDA arch
if echo "$GPU_NAME" | grep -qi "5090\|5080\|5070\|5060\|Blackwell"; then
  ARCH="sm_120"
elif echo "$GPU_NAME" | grep -qi "4090\|4080\|4070\|4060\|Ada"; then
  ARCH="sm_89"
elif echo "$GPU_NAME" | grep -qi "3090\|3080\|3070\|3060\|Ampere"; then
  ARCH="sm_86"
elif echo "$GPU_NAME" | grep -qi "2080\|2070\|2060\|Turing"; then
  ARCH="sm_75"
else
  ARCH="sm_86"
  echo -e "  ${YELLOW}Unknown GPU, defaulting to sm_86${NC}"
fi
echo "  CUDA arch: $ARCH"

# ── Install deps ─────────────────────────────────────────────
echo -e "${CYAN}[2/4] Installing dependencies...${NC}"
apt-get update -qq 2>/dev/null
apt-get install -y nodejs npm 2>/dev/null | tail -3

# ── Compile CUDA kernel ──────────────────────────────────────
echo -e "${CYAN}[3/4] Compiling CUDA miner...${NC}"
nvcc -O3 -arch=$ARCH -o pfft_miner pfft_miner.cu
echo -e "${GREEN}  Compiled: pfft_miner (arch=$ARCH)${NC}"

# ── Install Node deps ────────────────────────────────────────
echo -e "${CYAN}[4/4] Installing Node.js dependencies...${NC}"
npm install --silent

# ── Launch ───────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Launching PFFT GPU Miner${NC}"
echo -e "${GREEN}  GPU    : $GPU_NAME (id=$GPU_ID)${NC}"
echo -e "${GREEN}  Log    : /tmp/pfft-mine.log${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

GPU_ID=$GPU_ID nohup node miner.js \
  --rpc "$RPC_URL" \
  --key "$PRIVATE_KEY" \
  --gpu "$GPU_ID" \
  > /tmp/pfft-mine.log 2>&1 &

echo -e "${GREEN}  Miner started — PID: $!${NC}"
echo "$!" > /tmp/pfft-miner.pid

echo ""
echo "  Monitor: tail -f /tmp/pfft-mine.log"
echo "  Stop   : kill \$(cat /tmp/pfft-miner.pid)"
echo ""

sleep 4
echo -e "${CYAN}── Initial output ──────────────────────────${NC}"
tail -15 /tmp/pfft-mine.log
