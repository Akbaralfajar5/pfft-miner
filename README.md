# PFFT GPU Miner

GPU miner untuk **$PFFT (Pow Free Fair Token)** di Ethereum mainnet.

**Contract:** `0xEFAd2Eab7172dDEbE5Ce7a41f5Ddf8fCcE4Ca0CB`

## PoW Algorithm

```
hash = keccak256(encodePacked(challenge_bytes32, nonce_uint256))
valid if: hash <= POW_TARGET
```

Challenge bersifat per-wallet — `currentPowChallenge(walletAddress)` — jadi nonce valid untuk wallet A tidak bisa dipakai wallet B.

## Quick Start (Vast.ai)

```bash
# 1. Clone repo
git clone https://github.com/Akbaralfajar5/pfft-miner.git
cd pfft-miner
chmod +x setup.sh

# 2. Jalankan setup
./setup.sh "https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY" "0xYOUR_PRIVATE_KEY"
```

## Files

| File | Fungsi |
|------|--------|
| `pfft_miner.cu` | CUDA kernel — keccak256 GPU solver |
| `miner.js` | Node.js orchestrator — fetch challenge, solve, submit tx |
| `setup.sh` | One-command setup: compile CUDA, install deps, launch |
| `package.json` | Node.js dependencies |

## Manual Setup

```bash
# Install deps
apt-get install -y nodejs npm
npm install

# Compile CUDA (pilih arch sesuai GPU)
nvcc -O3 -arch=sm_89 -o pfft_miner pfft_miner.cu   # RTX 4090
nvcc -O3 -arch=sm_86 -o pfft_miner pfft_miner.cu   # RTX 3090
nvcc -O3 -arch=sm_120 -o pfft_miner pfft_miner.cu  # RTX 5090

# Jalankan
node miner.js --rpc "https://eth-mainnet.g.alchemy.com/v2/KEY" --key "0xPRIVATE_KEY"
```

## Token Info

| Parameter | Value |
|-----------|-------|
| Token | PFFT (Pow Free Fair Token) |
| Network | Ethereum mainnet |
| Max supply | 21,000,000 PFFT |
| Base mint | 1,000 PFFT (decay seiring supply) |
| Wallet cap | 10,000 PFFT per wallet |
| PoW stages | 24 / 28 / 32 / 36 / 40-bit |
| Algorithm | keccak256(encodePacked(challenge, nonce)) |

## GPU Performance (estimasi)

| GPU | Hashrate | 24-bit (~16M) | 32-bit (~4.3B) |
|-----|----------|---------------|----------------|
| RTX 4090 | ~4.4 GH/s | <0.01s | ~1s |
| RTX 3090 | ~2.5 GH/s | <0.01s | ~2s |
| RTX 5090 | ~9.9 GH/s | <0.01s | <0.5s |

## Vast.ai Setup

1. Rent instance: template `nvidia/cuda:12.4.1-devel-ubuntu22.04`
2. GPU: RTX 4090 atau 5090 recommended
3. Tambah SSH key sebelum launch
4. SSH ke instance, clone repo, jalankan `setup.sh`

## Notes

- Wallet cap: 10,000 PFFT per wallet — pakai multiple wallet untuk mine lebih banyak
- Butuh ETH untuk gas fee per mint (~0.001-0.005 ETH tergantung gas price)
- Challenge bersifat per-wallet, tidak bisa share nonce antar wallet
- Miner otomatis re-fetch challenge jika berubah

## Source

PFFT website: https://pffthash.com
Whitepaper: https://pffthash.com/#pow-gate
