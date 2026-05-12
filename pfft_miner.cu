/*
 * PFFT GPU Miner — CUDA keccak256 PoW solver
 *
 * PoW formula (from pffthash.com frontend):
 *   hash = keccak256(encodePacked(challenge_bytes32, nonce_uint256))
 *   valid if: BigInt(hash) <= POW_TARGET
 *
 * Contract: 0xEFAd2Eab7172dDEbE5Ce7a41f5Ddf8fCcE4Ca0CB (Ethereum mainnet)
 *
 * Usage:
 *   ./pfft_miner <challenge_hex> <target_hex> <start_nonce> <batch_size>
 *
 * Output on solve:
 *   FOUND:<nonce_decimal>
 *
 * Compile:
 *   nvcc -O3 -arch=sm_86 -o pfft_miner pfft_miner.cu   # RTX 3090/4090
 *   nvcc -O3 -arch=sm_89 -o pfft_miner pfft_miner.cu   # RTX 4090
 *   nvcc -O3 -arch=sm_120 -o pfft_miner pfft_miner.cu  # RTX 5090
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>

// ── Keccak-256 constants ─────────────────────────────────────────────────────

__device__ __constant__ uint64_t keccak_rc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ __constant__ int keccak_rho[24] = {
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

__device__ __constant__ int keccak_pi[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
};

#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

__device__ void keccak_f(uint64_t state[25]) {
    uint64_t C[5], D[5], temp;
    for (int round = 0; round < 24; round++) {
        // Theta
        for (int i = 0; i < 5; i++)
            C[i] = state[i] ^ state[i+5] ^ state[i+10] ^ state[i+15] ^ state[i+20];
        for (int i = 0; i < 5; i++) {
            D[i] = C[(i+4)%5] ^ ROTL64(C[(i+1)%5], 1);
            for (int j = 0; j < 25; j += 5)
                state[i+j] ^= D[i];
        }
        // Rho + Pi
        temp = state[1];
        for (int i = 0; i < 24; i++) {
            int j = keccak_pi[i];
            uint64_t t = state[j];
            state[j] = ROTL64(temp, keccak_rho[i]);
            temp = t;
        }
        // Chi
        for (int j = 0; j < 25; j += 5) {
            uint64_t t[5];
            for (int i = 0; i < 5; i++) t[i] = state[j+i];
            for (int i = 0; i < 5; i++)
                state[j+i] = t[i] ^ ((~t[(i+1)%5]) & t[(i+2)%5]);
        }
        // Iota
        state[0] ^= keccak_rc[round];
    }
}

// keccak256 of 64-byte input (bytes32 challenge + uint256 nonce)
// Input layout: challenge[32] || nonce_big_endian[32]
__device__ void keccak256_64(const uint8_t *input, uint8_t *output) {
    uint64_t state[25] = {0};

    // Absorb 64 bytes (rate = 136 bytes for keccak256)
    // XOR input into state (little-endian lane loading)
    for (int i = 0; i < 8; i++) {
        uint64_t lane = 0;
        for (int b = 0; b < 8; b++)
            lane |= ((uint64_t)input[i*8 + b]) << (b * 8);
        state[i] ^= lane;
    }

    // Padding: 0x01 at byte 64, 0x80 at byte 135
    state[8] ^= 0x01ULL;
    state[16] ^= 0x8000000000000000ULL;

    keccak_f(state);

    // Squeeze 32 bytes
    for (int i = 0; i < 4; i++) {
        uint64_t lane = state[i];
        for (int b = 0; b < 8; b++)
            output[i*8 + b] = (lane >> (b * 8)) & 0xFF;
    }
}

// ── Compare hash <= target (big-endian 32 bytes) ─────────────────────────────
__device__ bool hash_lte_target(const uint8_t *hash, const uint8_t *target) {
    for (int i = 0; i < 32; i++) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return true; // equal
}

// ── Main kernel ──────────────────────────────────────────────────────────────
__global__ void pfft_mine(
    const uint8_t *challenge,   // 32 bytes, big-endian
    const uint8_t *target,      // 32 bytes, big-endian
    uint64_t start_nonce,       // starting nonce
    uint64_t batch_size,        // nonces per thread
    uint64_t *found_nonce,      // output: found nonce (0 = not found)
    int *found_flag             // output: 1 if found
) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce_start = start_nonce + tid * batch_size;

    if (*found_flag) return;

    uint8_t input[64];
    uint8_t hash[32];

    // Copy challenge into first 32 bytes
    for (int i = 0; i < 32; i++)
        input[i] = challenge[i];

    for (uint64_t n = nonce_start; n < nonce_start + batch_size; n++) {
        if (*found_flag) return;

        // Encode nonce as big-endian uint256 (32 bytes)
        // nonce fits in 8 bytes, upper 24 bytes are 0
        for (int i = 0; i < 24; i++) input[32 + i] = 0;
        for (int i = 0; i < 8; i++)
            input[32 + 24 + i] = (n >> (56 - i * 8)) & 0xFF;

        keccak256_64(input, hash);

        if (hash_lte_target(hash, target)) {
            if (atomicCAS(found_flag, 0, 1) == 0) {
                *found_nonce = n;
            }
            return;
        }
    }
}

// ── Host helpers ─────────────────────────────────────────────────────────────

void hex_to_bytes(const char *hex, uint8_t *bytes, int len) {
    // Strip 0x prefix
    if (hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X')) hex += 2;
    for (int i = 0; i < len; i++) {
        unsigned int byte;
        sscanf(hex + i * 2, "%02x", &byte);
        bytes[i] = (uint8_t)byte;
    }
}

void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <challenge_hex> <target_hex> <start_nonce> <batch_per_thread>\n", prog);
    fprintf(stderr, "  challenge_hex   : 32-byte hex (with or without 0x)\n");
    fprintf(stderr, "  target_hex      : 32-byte hex POW_TARGET (with or without 0x)\n");
    fprintf(stderr, "  start_nonce     : starting nonce (decimal)\n");
    fprintf(stderr, "  batch_per_thread: nonces per GPU thread per call (e.g. 64)\n");
    fprintf(stderr, "\nOutput: FOUND:<nonce> or EXHAUSTED\n");
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    uint8_t challenge[32], target[32];
    hex_to_bytes(argv[1], challenge, 32);
    hex_to_bytes(argv[2], target, 32);
    uint64_t start_nonce = strtoull(argv[3], NULL, 10);
    uint64_t batch_per_thread = strtoull(argv[4], NULL, 10);

    // GPU config
    int threads_per_block = 256;
    int blocks = 4096;
    uint64_t total_threads = (uint64_t)threads_per_block * blocks;
    uint64_t total_nonces = total_threads * batch_per_thread;

    // Allocate device memory
    uint8_t *d_challenge, *d_target;
    uint64_t *d_found_nonce;
    int *d_found_flag;

    cudaMalloc(&d_challenge, 32);
    cudaMalloc(&d_target, 32);
    cudaMalloc(&d_found_nonce, sizeof(uint64_t));
    cudaMalloc(&d_found_flag, sizeof(int));

    cudaMemcpy(d_challenge, challenge, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, 32, cudaMemcpyHostToDevice);

    uint64_t h_found_nonce = 0;
    int h_found_flag = 0;
    cudaMemcpy(d_found_nonce, &h_found_nonce, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_flag, &h_found_flag, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    pfft_mine<<<blocks, threads_per_block>>>(
        d_challenge, d_target,
        start_nonce, batch_per_thread,
        d_found_nonce, d_found_flag
    );

    cudaDeviceSynchronize();

    // Read results
    cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_found_nonce, d_found_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    if (h_found_flag) {
        printf("FOUND:%llu\n", (unsigned long long)h_found_nonce);
    } else {
        printf("EXHAUSTED:%llu\n", (unsigned long long)(start_nonce + total_nonces));
    }

    cudaFree(d_challenge);
    cudaFree(d_target);
    cudaFree(d_found_nonce);
    cudaFree(d_found_flag);

    return h_found_flag ? 0 : 1;
}
