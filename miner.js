#!/usr/bin/env node
/**
 * PFFT Miner — Node.js orchestrator
 *
 * Flow:
 *   1. Connect to Ethereum via RPC
 *   2. Fetch currentPowChallenge(wallet) + POW_TARGET from contract
 *   3. Call CUDA binary to solve PoW
 *   4. Submit freeMint(powNonce) transaction
 *   5. Repeat
 *
 * Usage:
 *   node miner.js --rpc <RPC_URL> --key <PRIVATE_KEY> [--gpu <0>] [--batch <64>]
 *
 * Requirements:
 *   npm install ethers
 */

const { ethers } = require("ethers");
const { execSync, spawnSync } = require("child_process");
const path = require("path");
const fs = require("fs");

// ── Config ───────────────────────────────────────────────────────────────────

const CONTRACT_ADDRESS = "0xEFAd2Eab7172dDEbE5Ce7a41f5Ddf8fCcE4Ca0CB";

const ABI = [
  "function freeMint(uint256 powNonce) external",
  "function currentPowChallenge(address user) view returns (bytes32)",
  "function POW_TARGET() view returns (uint256)",
  "function POW_DIFFICULTY_BITS() view returns (uint256)",
  "function currentPowStage() view returns (uint256)",
  "function getInfo() view returns (uint256 currentMinted, uint256 remainingSupply, uint256 currentDecayRate, uint256 nextMintAmount)",
  "function minted(address user) view returns (uint256)",
  "function MAX_SUPPLY() view returns (uint256)",
  "function BASE_MINT_AMOUNT() view returns (uint256)",
  "function calculateActualMint(uint256 requested) view returns (uint256)",
];

const CUDA_BINARY = path.join(__dirname, "pfft_miner");
const BATCH_PER_THREAD = 64;   // nonces per GPU thread per kernel call
const BLOCKS = 4096;
const THREADS = 256;

// ── Args ─────────────────────────────────────────────────────────────────────

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = { gpu: 0, batch: BATCH_PER_THREAD };
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--rpc")   opts.rpc = args[++i];
    if (args[i] === "--key")   opts.key = args[++i];
    if (args[i] === "--gpu")   opts.gpu = parseInt(args[++i]);
    if (args[i] === "--batch") opts.batch = parseInt(args[++i]);
  }
  if (!opts.rpc || !opts.key) {
    console.error("Usage: node miner.js --rpc <RPC_URL> --key <PRIVATE_KEY> [--gpu 0] [--batch 64]");
    process.exit(1);
  }
  return opts;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function log(msg, level = "info") {
  const ts = new Date().toISOString().slice(11, 19);
  const prefix = level === "ok" ? "✓" : level === "warn" ? "⚠" : level === "err" ? "✗" : "·";
  console.log(`[${ts}] ${prefix} ${msg}`);
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function targetToHex(target) {
  // target is BigInt, convert to 32-byte hex
  return "0x" + target.toString(16).padStart(64, "0");
}

// ── GPU solver ───────────────────────────────────────────────────────────────

function solvePoW(challengeHex, targetHex, startNonce, batchPerThread) {
  if (!fs.existsSync(CUDA_BINARY)) {
    throw new Error(`CUDA binary not found: ${CUDA_BINARY}\nRun: nvcc -O3 -arch=sm_86 -o pfft_miner pfft_miner.cu`);
  }

  const env = { ...process.env, CUDA_VISIBLE_DEVICES: String(process.env.GPU_ID || 0) };
  const result = spawnSync(
    CUDA_BINARY,
    [challengeHex, targetHex, String(startNonce), String(batchPerThread)],
    { env, timeout: 120000 }
  );

  if (result.error) throw result.error;

  const stdout = (result.stdout || "").toString().trim();
  const stderr = (result.stderr || "").toString().trim();

  if (stderr) log(`GPU stderr: ${stderr}`, "warn");

  if (stdout.startsWith("FOUND:")) {
    return { found: true, nonce: BigInt(stdout.slice(6)) };
  } else if (stdout.startsWith("EXHAUSTED:")) {
    return { found: false, nextNonce: BigInt(stdout.slice(10)) };
  } else {
    throw new Error(`Unexpected GPU output: ${stdout}`);
  }
}

// ── Main mining loop ─────────────────────────────────────────────────────────

async function main() {
  const opts = parseArgs();

  process.env.GPU_ID = String(opts.gpu);

  const provider = new ethers.JsonRpcProvider(opts.rpc);
  const wallet = new ethers.Wallet(opts.key, provider);
  const contract = new ethers.Contract(CONTRACT_ADDRESS, ABI, wallet);

  log(`PFFT GPU Miner starting`);
  log(`Wallet  : ${wallet.address}`);
  log(`Contract: ${CONTRACT_ADDRESS}`);
  log(`GPU     : ${opts.gpu}`);

  // Check wallet minted so far
  const alreadyMinted = await contract.minted(wallet.address);
  const powStage = await contract.currentPowStage();
  const diffBits = await contract.POW_DIFFICULTY_BITS();
  log(`Wallet minted: ${ethers.formatUnits(alreadyMinted, 18)} PFFT`);
  log(`PoW stage: ${powStage}  difficulty: ${diffBits}-bit`);

  let totalMints = 0;
  let totalPfft = 0n;
  const startTime = Date.now();

  while (true) {
    try {
      // Fetch current state
      const [challenge, target, info] = await Promise.all([
        contract.currentPowChallenge(wallet.address),
        contract.POW_TARGET(),
        contract.getInfo(),
      ]);

      const challengeHex = challenge; // bytes32 hex string
      const targetHex = targetToHex(target);

      log(`Round start — challenge: ${challengeHex.slice(0, 10)}...  target: ${targetHex.slice(0, 10)}...`);
      log(`Next mint: ~${ethers.formatUnits(info.nextMintAmount, 18)} PFFT  |  minted: ${ethers.formatUnits(info.currentMinted, 18)} / 21M`);

      // Solve PoW
      let nonce = null;
      let startNonce = 0n;
      const solveStart = Date.now();
      let totalAttempts = 0n;

      while (!nonce) {
        const batchNonces = BigInt(BLOCKS) * BigInt(THREADS) * BigInt(opts.batch);
        const result = solvePoW(challengeHex, targetHex, startNonce, opts.batch);

        totalAttempts += batchNonces;

        if (result.found) {
          nonce = result.nonce;
          const elapsed = (Date.now() - solveStart) / 1000;
          const hashrate = Number(totalAttempts) / elapsed;
          log(`PoW solved! nonce=${nonce}  attempts=${totalAttempts}  time=${elapsed.toFixed(1)}s  hashrate=${(hashrate/1e6).toFixed(1)} MH/s`, "ok");
        } else {
          startNonce = result.nextNonce;
          const elapsed = (Date.now() - solveStart) / 1000;
          const hashrate = Number(totalAttempts) / elapsed;
          process.stdout.write(`\r  · grinding... ${(Number(totalAttempts)/1e6).toFixed(0)}M hashes  ${(hashrate/1e6).toFixed(1)} MH/s  `);

          // Re-fetch challenge every ~30s to avoid stale challenge
          if (elapsed > 30 && Number(totalAttempts) % (batchNonces * 10n === 0n ? 1 : 1) === 0) {
            const newChallenge = await contract.currentPowChallenge(wallet.address);
            if (newChallenge !== challengeHex) {
              log(`\nChallenge changed — restarting round`);
              startNonce = 0n;
              break;
            }
          }
        }
      }

      if (!nonce) continue; // challenge changed, restart

      // Submit transaction
      console.log(""); // newline after progress
      log(`Submitting freeMint(${nonce})...`);

      const tx = await contract.freeMint(nonce, {
        gasLimit: 300000,
      });

      log(`Tx sent: ${tx.hash}`);
      const receipt = await tx.wait();

      if (receipt.status === 1) {
        totalMints++;
        totalPfft += info.nextMintAmount;
        const uptime = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
        log(`MINTED! +${ethers.formatUnits(info.nextMintAmount, 18)} PFFT  |  total: ${totalMints} mints  ${ethers.formatUnits(totalPfft, 18)} PFFT  uptime: ${uptime}m`, "ok");
        log(`Etherscan: https://etherscan.io/tx/${tx.hash}`);
      } else {
        log(`Tx failed: ${tx.hash}`, "err");
      }

      await sleep(1000);
    } catch (err) {
      log(`Error: ${err.message}`, "err");
      await sleep(5000);
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
