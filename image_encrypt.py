#!/usr/bin/env python3
"""
image_encrypt.py
Simple image encrypt/decrypt using pixel XOR + permutation (learning tool).
Usage:
  python image_encrypt.py encrypt input.png output_enc.png --key "mypassword"
  python image_encrypt.py decrypt output_enc.png restored.png --key "mypassword"
Options:
  --method {xor,shuffle,both}  (default: both)
  --drop-alpha                 (when decrypting, save as RGB instead of RGBA)
"""
import argparse
import hashlib
from PIL import Image
import numpy as np

def derive_seed(password: str, salt: bytes = b'image_encrypt_salt_v1') -> int:
    # PBKDF2 to produce deterministic seed from password
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100_000, dklen=8)
    return int.from_bytes(dk, 'big')

def process_image(in_path, out_path, password, action='encrypt', method='both', drop_alpha=False):
    im = Image.open(in_path).convert('RGBA')   # use RGBA for consistency
    arr = np.array(im)                         # shape (H, W, C)
    H, W, C = arr.shape
    flat = arr.reshape(-1, C)                 # shape (N, C)
    N = flat.shape[0]

    seed = derive_seed(password)
    rng = np.random.default_rng(seed)

    # Generate resources deterministically (same order for encrypt/decrypt)
    perm = None
    keystream = None

    if method in ('shuffle', 'both'):
        perm = rng.permutation(N)            # permutation array
    if method in ('xor', 'both'):
        keystream = rng.integers(0, 256, size=flat.shape, dtype=np.uint8)

    if action == 'encrypt':
        if method in ('xor', 'both'):
            flat = np.bitwise_xor(flat, keystream)
        if method in ('shuffle', 'both'):
            flat = flat[perm]
    elif action == 'decrypt':
        if method in ('shuffle', 'both'):
            inv_perm = np.argsort(perm)
            flat = flat[inv_perm]
        if method in ('xor', 'both'):
            flat = np.bitwise_xor(flat, keystream)
    else:
        raise ValueError("action must be 'encrypt' or 'decrypt'")

    out_arr = flat.reshape((H, W, C)).astype(np.uint8)
    out_img = Image.fromarray(out_arr, mode='RGBA')

    if drop_alpha:
        out_img = out_img.convert('RGB')

    out_img.save(out_path)
    print(f"{action.title()}ion complete — saved to: {out_path}")

def main():
    p = argparse.ArgumentParser(description="Image encrypt/decrypt (pixel XOR + permutation)")
    p.add_argument('action', choices=['encrypt', 'decrypt'])
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('--key', required=True, help="Password/key (string)")
    p.add_argument('--method', choices=['xor', 'shuffle', 'both'], default='both')
    p.add_argument('--drop-alpha', action='store_true', help="When decrypting, save as RGB")
    args = p.parse_args()

    process_image(args.input, args.output, args.key, action=args.action, method=args.method, drop_alpha=args.drop_alpha)

if __name__ == '__main__':
    main()