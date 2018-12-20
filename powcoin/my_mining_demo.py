import time, hashlib

def get_proof(header,nonce):
    preimage = f"{header}:{nonce}".encode()
    proof_hex = hashlib.sha256(preimage).hexdigest()
    return int(proof_hex, 16)

def mine(header, target, nonce):

    while(get_proof(header, nonce) >= target):
        #new guess
        nonce += 1
    return nonce

def mining_demo(header):
    previous_nonce = -1
    for difficulty_bits in range(1, 25):
        target = 2 ** (256 - difficulty_bits)
        start = time.time()
        nonce = mine(header, target, previous_nonce)
        elapsed = time.time() - start
        proof = get_proof(header,nonce)
        bin_proof_str = f"{proof:0256b}"[:50]
        target_str = f"{target:.0e}"
        elapsed_str = f"{elapsed:.0e}" if nonce != previous_nonce else ""
        print(f"bits: {difficulty_bits:>3} target: {target_str:>7} time: {elapsed_str:>7} nonce: {nonce:>10} proof: {bin_proof_str}...")
        previous_nonce = nonce

if __name__ == "__main__":
    header = "hello"
    mining_demo(header)
