"""
POW Coin

Usage:
  mypowcoin.py serve
  mypowcoin.py ping
  mypowcoin.py tx <from> <to> <amount> [--node <node>]
  mypowcoin.py balance <name> [--node <node>]

Options:
  -h --help     Show this screen.
  --node=<node>   Which node to talk to [default: node0].
"""

import uuid
import socketserver
import socket
import sys
import argparse
import time
import os
import logging
import threading
import time
import hashlib
import random
import re
import pickle

from docopt import docopt
from copy import deepcopy
from ecdsa import SigningKey, SECP256k1

HOST, PORT = '0.0.0.0', 10000
ADDRESS = (HOST, PORT)
GET_BLOCK_CHUNKS = 10
BLOCK_SUBSIDY = 50
ENFORCE_AUTHENTICATION = False
node = None
lock = threading.Lock()

logging.basicConfig(level="INFO", format='%(threadName)-6s | %(message)s')
logger = logging.getLogger(__name__)



def spend_message(tx, index):
    outpoint = tx.tx_ins[index].outpoint
    return serialize(outpoint) + serialize(tx.tx_outs)

def total_work(blocks):
    return len(blocks)

def tx_in_to_tx_out(tx_in, blocks):
    for block in blocks:
        for tx in block.txns:
            if tx.id == tx_in.tx_id:
                return tx.tx_outs[tx_in.index]

class Block:

    def __init__(self, txns, prev_id, nonce):
        self.txns = txns
        self.prev_id = prev_id
        self.nonce = nonce

    @property
    def header(self):
        return serialize(self)

    @property
    def id(self):
        return hashlib.sha256(self.header).hexdigest()

    @property
    def proof(self):
        return int(self.id, 16)

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        prev_id = self.prev_id[:10] if self.prev_id else None
        return f"Block prev_id={prev_id}... id={self.id} ..."

class Tx:

    def __init__(self, id, tx_ins, tx_outs):
        self.id = id
        self.tx_ins = tx_ins
        self.tx_outs = tx_outs

    def sign_input(self, index, private_key):
        message = spend_message(self, index)
        signature = private_key.sign(message)
        self.tx_ins[index].signature = signature

    def verify_input(self, index, public_key):
        tx_in = self.tx_ins[index]
        message = spend_message(self, index)
        return public_key.verify(tx_in.signature, message)

    def __eq__(self, other):
        return self.id == other.id
    @property
    def is_coinbase(self):
        return self.tx_ins[0].tx_id is None


class TxIn:

    def __init__(self, tx_id, index, signature=None):
        self.tx_id = tx_id
        self.index = index
        self.signature = signature

    @property
    def outpoint(self):
        return (self.tx_id, self.index)


class TxOut:

    def __init__(self, tx_id, index, amount, public_key):
        self.tx_id = tx_id
        self.index = index
        self.amount = amount
        self.public_key = public_key

    @property
    def outpoint(self):
        return (self.tx_id, self.index)


class Node:

    def __init__(self, address):
        # empty list
        self.blocks = []
        self.branches = []
        # empty dictionary
        self.utxo_set = {}
        self.mempool = []
        self.peers = []
        self.pending_peers = []
        self.address = address
        self.syncing = False

    def connect(self, peer):
        if peer not in self.peers and peer != self.address:
            logger.info(f'(handshake) Sent "connect" to {peer[0]}')
            try:
                send_message(peer, "connect", None)
                self.pending_peers.append(peer)
            except:
                logger.info(f'(handshake) Node {peer[0]} offline')

    def sync(self):
        self.syncing = True
        blocks = self.blocks[-GET_BLOCK_CHUNKS:]
        block_ids = [block.id for block in blocks]
        #for peer in self.peers:
        #    send_message(peer, "sync", block_ids)
        if len(self.peers):
            send_message(self.peers[0], "sync", block_ids)

    def connect_tx(self, tx):
        if not tx.is_coinbase:
            for tx_in in tx.tx_ins:
                del self.utxo_set[tx_in.outpoint]

        for tx_out in tx.tx_outs:
            self.utxo_set[tx_out.outpoint] = tx_out

        # cleanup mempool
        if tx in self.mempool:
            self.mempool.remove(tx)

    def disconnect_tx(self, tx):
        # Add back UTXO's spent by this tx
        if not tx.is_coinbase:
            for tx_in in tx.tx_ins:
                tx_out = tx_in_to_tx_out(tx_in, self.blocks)
                self.utxo_set[tx_out.outpoint] = tx_out

        # Remove UTXO's that were created by the tx
        for tx_out in tx.tx_outs:
            del self.utxo_set[tx_out.outpoint]

        # Put tx back in the mempool
        if tx not in self.mempool and not tx.is_coinbase:
            self.mempool.append(tx)
            logger.info(f"Added tx to the mempool")

    def validate_tx(self, tx):
        in_sum = 0
        out_sum = 0

        for index, tx_in in enumerate(tx.tx_ins):
            # TxIn spending unspent output
            assert tx_in.outpoint in self.utxo_set
            # Grab the tx_out
            tx_out = self.utxo_set[tx_in.outpoint]

            # Verify signature using public key of TxOut we're spending
            public_key = tx_out.public_key
            tx.verify_input(index, public_key)

            # Sum up the total inputs
            amount = tx_out.amount
            in_sum += amount

        for tx_out in tx.tx_outs:
            out_sum += tx_out.amount

        assert in_sum == out_sum

    def handle_tx(self, tx):
        # Save to self.utxo_set if it's valid
        self.validate_tx(tx)
        self.mempool.append(tx)

        # Propogate transaction
        for peer in self.peers:
            send_message(peer, "tx", tx)

    def validate_block(self, block, validate_txns=False):
        assert block.proof < POW_TARGET, "insufficient Proof-of-Work"

        if validate_txns:
            # Validate prepare_coinbase
            self.validate_coinbase(block.txns[0])
            # Verify every transaction
            for tx in block.txns[1:]:
                self.validate_tx(tx)

    def validate_coinbase(self, tx):
        assert(len(tx.tx_ins) == len(tx.tx_outs) == 1)
        assert(tx.tx_outs[0].amount <= BLOCK_SUBSIDY)

    def find_in_branch(self, block_id):
        for br_idx, branch in enumerate(self.branches):
            for height, block in enumerate(branch):
                if block.id == block_id:
                    return branch, br_idx, height
        return None, None, None

    def handle_block(self, block):
        # Ignore block if we already sean it
        found_in_chain = block in self.blocks
        found_in_branch = self.find_in_branch(block.id)[0] is not None
        if found_in_chain or found_in_branch:
            raise Exception("Received duplicate block")

        # Lookup the previous blocks
        branch, branch_index, height = self.find_in_branch(block.prev_id)
        # Conditions
        extends_chain = block.prev_id == self.blocks[-1].id
        forks_chain = not extends_chain and \
                block.prev_id in [b.id for b in self.blocks]
        extends_branch = branch and height == len(branch) - 1
        forks_branch = branch and height != len(branch) - 1

        # Always validate, but txns only when extending
        self.validate_block(block, validate_txns=extends_chain)

        # handle each condition seperately
        if extends_chain:
            self.connect_block(block)
            logger.info(f"Extended chain to height {len(self.blocks) -1}")
        elif forks_chain:
            self.branches.append([block])
            logger.info(f"Created branch {len(self.branches) - 1}")
        elif extends_branch:
            branch.append(block)
            logger.info(f"Extend branch {branch_index} to height {len(branch) -1} ")

            # Reorg if branch now has more work than main chain
            chain_ids = [b.id for b in self.blocks]
            fork_height = chain_ids.index(branch[0].prev_id)
            chain_since_fork = self.blocks[fork_height + 1:]
            if total_work(branch) > total_work(chain_since_fork):
                logger.info(f"Reorging to branch {branch_index}")
                self.reorg(branch, branch_index)


        elif forks_branch:
            self.branches.append(branch[:height+1] + [block])
            logger.info(f"Created branch {len(self.branches) -1} to height \
                    {len(self.branches[-1]) -1}")
        else:
            self.sync()
            raise Exception("Encountered block with unknown parent, syncing...")

        # Block propagation
        if not self.syncing:
            for peer in self.peers:
                disrupt(func = send_message, args = [peer, "blocks", [block]])

    def reorg(self, branch, branch_index):
        # Disconnect to fork block, preserving as a branch
        disconnected_blocks = []
        while self.blocks[-1].id != branch[0].prev_id:
            block = self.blocks.pop()
            for tx in block.txns:
                self.disconnect_tx(tx)
            disconnected_blocks.insert(0, block)

        # Replace branch with newly disconnected chain
        self.branches[branch_index] = disconnected_blocks

        # Connect branch, rollback if error detected (try...except...)
        for block in branch:
            try:
                self.validate_block(block, validate_txns=True)
                self.connect_block(block)
            except:
                self.reorg(disconnected_blocks, branch_index)
                logger.info(f"Reorg failed")
                return

    def connect_block(self, block):
        # update self.blocks
        self.blocks.append(block)

        # After all tx are valid, update UTXO set / mempool
        for tx in block.txns:
            self.connect_tx(tx)

    def fetch_utxos(self, public_key):
        return [utxo for utxo in self.utxo_set.values()
                if utxo.public_key == public_key]

    def fetch_balance(self, public_key):
        # Fetch utxos associated with this public key
        utxos = self.fetch_utxos(public_key)
        # Sum the amounts
        return sum([tx_out.amount for tx_out in utxos])


def prepare_simple_tx(utxos, sender_private_key, recipient_public_key, amount):
    sender_public_key = sender_private_key.get_verifying_key()

    # Construct tx.tx_outs
    tx_ins = []
    tx_in_sum = 0
    for tx_out in utxos:
        tx_ins.append(
            TxIn(tx_id=tx_out.tx_id, index=tx_out.index, signature=None))
        tx_in_sum += tx_out.amount
        if tx_in_sum >= amount:
            break

    # Make sure sender can afford it
    assert tx_in_sum >= amount

    # Construct tx.tx_outs
    tx_id = uuid.uuid4()
    change = tx_in_sum - amount
    tx_outs = [
        TxOut(tx_id=tx_id, index=0, amount=amount,
              public_key=recipient_public_key),
        TxOut(tx_id=tx_id, index=1, amount=change,
              public_key=sender_public_key),
    ]

    # Construct tx and sign inputs
    tx = Tx(id=tx_id, tx_ins=tx_ins, tx_outs=tx_outs)
    for i in range(len(tx.tx_ins)):
        tx.sign_input(i, sender_private_key)

    return tx

def prepare_coinbase(public_key, tx_id = None):
    if not tx_id:
        tx_id = uuid.uuid4()
    return Tx(
        id = tx_id,
        tx_ins = [
            TxIn(None,None,None),
        ],
        tx_outs = [
            TxOut(
                tx_id = tx_id,
                index = 0,
                amount = BLOCK_SUBSIDY,
                public_key = public_key,
            ),
        ],
    )

##########
# Mining #
##########


DIFFICULTY_BITS = 15
POW_TARGET = 2 ** (256 - DIFFICULTY_BITS)
mining_interrupt = threading.Event()


def mine_block(block):
    while(block.proof >= POW_TARGET):
        if mining_interrupt.is_set():
            mining_interrupt.clear()
            return None
        block.nonce += 1
    return block


def mine_forever(public_key):
    logger.info("Miner started")
    while True:
        coinbase = prepare_coinbase(public_key)
        unmined_block = Block(
            txns = [coinbase] + node.mempool,
            prev_id=node.blocks[-1].id,
            nonce=random.randint(0, 10000000),
        )
        mined_block = mine_block(unmined_block)
        if mined_block:
            logger.info("")
            logger.info("Mined new Block")
            with lock:
                node.handle_block(mined_block)
        else:
            logger.info("")
            logger.info("Mining Interrupted")


def mine_genesis_block(node, public_key):
    coinbase = prepare_coinbase(public_key,"abcd123")
    unmined_block = Block(
        txns=[coinbase],
        prev_id=None,
        nonce=0,
    )
    mined_block = mine_block(unmined_block)
    node.blocks.append(mined_block)
    # update UTXO set, award coinbase, etc
    node.connect_tx(coinbase)
    return mined_block

##############
# Networking #
##############

def serialize(coin):
    return pickle.dumps(coin)

def deserialize(serialized):
    return pickle.loads(serialized)

def read_message(s):
    message = b''
    # our protocol is first 4 bytes signify message length
    raw_message_length = s.recv(4) or b"\x00"
    message_length = int.from_bytes(raw_message_length, 'big')
    while message_length > 0:
        chunk = s.recv(1024)
        message += chunk
        message_length -= len(chunk)

    return deserialize(message)

def prepare_message(command, data):
    message = {
        "command": command,
        "data": data,
    }
    serialized_message = serialize(message)
    length = len(serialized_message).to_bytes(4, 'big')
    return length + serialized_message

def disrupt(func, args):
    # simulate packet packet loss
    if random.randint(0,10) != 0:
        # simulate network latency
        threading.Timer(random.random(), func, args).start()

class TCPHandler(socketserver.BaseRequestHandler):

    def get_canonical_peer_address(self):
        ip = self.client_address[0]
        try:
            hostname = socket.gethostbyaddr(ip)
            hostname = re.search(r"_(.*?)_", hostname[0]).group(1)
        except:
            hostname = ip
        return (hostname, PORT)

    def respond(self, command, data):
        response = prepare_message(command, data)
        return self.request.sendall(response)

    def handle(self):
        message = read_message(self.request)
        command = message["command"]
        data = message["data"]

        # logger.info(f"Received {message}")

        peer = self.get_canonical_peer_address()
        # Handshake and Authentication
        if command == "connect":
            # TODO add more conditions (max_peer) - restrict acceptance of new nodes
            if peer not in node.peers and peer not in node.pending_peers:
                # only totally unrelated peers
                node.pending_peers.append(peer)
                logger.info(f'(handshake) Accept "connect" request from "{peer[0]}"')
                send_message(peer, "connect-response", None)
        elif command == "connect-response":
            if peer in node.pending_peers and peer not in node.peers:
                node.pending_peers.remove(peer)
                node.peers.append(peer)
                logger.info(f'(handshake) Connected to "{peer[0]}"')
                # Resend connect-response
                send_message(peer, "connect-response", None)
                # Request their peers
                send_message(peer, "peers", None)
        else:
            # we don't want to handle a message if he isn't connected to us
            if ENFORCE_AUTHENTICATION:
                assert peer in node.peers, f"Rejecting {command} from unconnected {peer[0]}"

        # Business logic

        if command == "peers":
            send_message(peer, "peers-response", node.peers)
        if command == "peers-response":
            for peer in data:
                node.connect(peer)

        if command == "sync":
            # find our most recent block peer doesn't know about but builds off a block they do know about
            peer_block_ids = data
            for block in node.blocks[::-1]:
                if block.id not in peer_block_ids \
                        and block.prev_id in peer_block_ids:
                    height = node.blocks.index(block)
                    blocks = node.blocks[height:height+GET_BLOCK_CHUNKS]
                    send_message(peer, "blocks", blocks)
                    logger.info('Served "sync" request')
                    return

            logger.info('Couldn\'t serve "sync" request')


        if command == "ping":
            self.respond(command="pong", data="")

        if command == "tx":
            try:
                node.handle_tx(data)
                self.respond(command="tx-response", data="accepted")
            except:
                self.respond(command="tx-response", data="rejected")

        if command == "blocks":
            for block in data:
                try:
                    with lock:
                        node.handle_block(block)
                    mining_interrupt.set()
                except:
                    logger.info(f"Rejected bad block")

            if len(data) == GET_BLOCK_CHUNKS:
                node.sync()
            else:
                node.syncing = False

        if command == "utxos":
            balance = node.fetch_utxos(data)
            self.respond(command="utxos-response", data=balance)

        if command == "balance":
            balance = node.fetch_balance(data)
            self.respond(command="balance-response", data=balance)


def serve():
    logger.info("Server started")
    server = socketserver.TCPServer(ADDRESS, TCPHandler)
    server.serve_forever()


def send_message(address, command, data, response=False):
    message = prepare_message(command, data)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(address)
        s.sendall(message)
        if response:
            return read_message(s)

def external_address(node):
    i = int(node[-1])
    port = PORT + i
    return ("localhost", port)

#######
# CLI #
#######

def lookup_private_key(name):
    exponent = {
        "alice": 1, "bob": 2, "node0": 3, "node1": 4, "node2": 5
    }[name]
    return SigningKey.from_secret_exponent(exponent, curve=SECP256k1)

def lookup_public_key(name):
    return lookup_private_key(name).get_verifying_key()

def main(args):
    threading.current_thread().name = "main"
    if args["serve"]:
        name = os.environ["NAME"]
        duration = 30 * ["node0", "node1", "node2"].index(name)
        time.sleep(duration)

        global node
        node = Node(address=(name, PORT))

        # Mine genesis block to Alice!!!
        mine_genesis_block(node, lookup_public_key("alice"))

        # start a server thread
        server_thread = threading.Thread(target=serve, name="server")
        server_thread.start()

        # Join the network
        peers = [(p, PORT) for p in os.environ['PEERS'].split(',')]
        for peer in peers:
            node.connect(peer)

        # wait for peer connection
        time.sleep(1)

        # Do initial block download
        node.sync()
        # Wait for ibd to finish
        time.sleep(1)

        # start a miner thread
        miner_public_key = lookup_public_key(name)
        miner_thread = threading.Thread(target=mine_forever,
            args = [miner_public_key], name="miner")
        miner_thread.start()

    elif args["ping"]:
        response = send_message(ADDRESS, "ping", "", response=True)
        print(response)
    elif args["balance"]:
        name = args["<name>"]
        address = external_address(args["--node"])
        public_key = lookup_public_key(name)
        response = send_message(address, "balance", public_key, response=True)
        print(response)
    elif args["tx"]:
        # Grab parameters
        nodes = args["--node"].split(',')
        addresses = list(map(external_address, nodes))
        sender_private_key = lookup_private_key(args["<from>"])
        sender_public_key = sender_private_key.get_verifying_key()
        recipient_private_key = lookup_private_key(args["<to>"])
        recipient_public_key = recipient_private_key.get_verifying_key()
        amount = int(args["<amount>"])

        # Fetch utxos available to spend
        response = send_message(
            addresses[0], "utxos", sender_public_key, response=True)
        utxos = response["data"]

        # Prepare transaction
        tx = prepare_simple_tx(utxos, sender_private_key,
                               recipient_public_key, amount)

        # send to node
        for addr in addresses:
            response = send_message(addr, "tx", tx, response=True)
        print(response)
    else:
        print("Invalid commands")


if __name__ == '__main__':
    # print(docopt(__doc__))
    main(docopt(__doc__))
