"""
POW Synca-Coin

Usage:
  my_pow_syndacoin.py serve
  my_pow_syndacoin.py ping
  my_pow_syndacoin.py tx <from> <to> <amount> [--node <node>]
  my_pow_syndacoin.py balance <name> [--node <node>]

Options:
  -h --help     Show this screen.
  --node=<node>   Which node to talk to [default: node0].
"""

import uuid, socketserver, socket, sys, argparse, time, os, logging, threading, time, hashlib, random

from docopt import docopt
from copy import deepcopy
from ecdsa import SigningKey, SECP256k1
from utils import serialize, deserialize

from identities import user_public_key, user_private_key

HOST, PORT = '0.0.0.0',10000
ADDRESS = (HOST, PORT)
node = None

logging.basicConfig(level="INFO",format='%(threadName)-6s | %(message)s')
logger = logging.getLogger(__name__)

def spend_message(tx, index):
    outpoint = tx.tx_ins[index].outpoint
    return serialize(outpoint) + serialize(tx.tx_outs)

class Block:

    def __init__(self, txns, prev_id, nonce):
        self.txns = txns
        self.prev_id = prev_id
        self.nonce = nonce
        # add stuff here

    @property
    def header(self):
        return serialize(self)
        # return serialize([self.txns, self.prev_id])

    @property
    def id(self):
        return hashlib.sha256(self.header).hexdigest()

    @property
    def proof(self):
        return int(self.id, 16)

    def __repr__(self):
        return f"Block prev_id={self.prev_id}... id={self.id} ..."

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

    def __init__(self):
        self.id = id
        # empty list
        self.blocks = []
        # empty dictionary
        self.utxo_set = {}
        self.zq_utxo_set = {}
        self.mempool = []
        self.peer_addresses = {(hostname, PORT) for hostname in os.environ['PEERS'].split(',')}

    def update_utxo_set(self, tx, final):
        if final:
            for tx_out in tx.tx_outs:
                self.utxo_set[tx_out.outpoint] = tx_out
            for tx_in in tx.tx_ins:
                del self.utxo_set[tx_in.outpoint]

        for tx_out in tx.tx_outs:
            self.zq_utxo_set[tx_out.outpoint] = tx_out
        for tx_in in tx.tx_ins:
            try:
                del self.zq_utxo_set[tx_in.outpoint]
            except KeyError:
                pass

    def validate_tx(self, tx, temp_utxo=None):
        in_sum = 0
        out_sum = 0
        if temp_utxo == None:
            utxo = self.zq_utxo_set
        else:
            utxo = temp_utxo

        for index, tx_in in enumerate(tx.tx_ins):
            # TxIn spending unspent output
            assert tx_in.outpoint in utxo
            # Grab the tx_out
            tx_out = utxo[tx_in.outpoint]

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
        self.update_utxo_set(tx, False)
        self.mempool.append(tx)

    def validate_block(self, block):
        assert block.proof < POW_TARGET, "insufficient Proof-of-Work"
        assert block.prev_id == self.blocks[-1].id, "Previous-Id do not match"


    def handle_block(self, block):
        # Check work and chain ordering
        self.validate_block(block)
        # Verify every transaction
        # copy utxo set to validate on the fly
        temp_utxo = deepcopy(self.utxo_set)
        for tx in block.txns:
            self.validate_tx(tx, temp_utxo)
            # After each TX update the temporary utxo set
            for tx_out in tx.tx_outs:
                temp_utxo[tx_out.outpoint] = tx_out
            for tx_in in tx.tx_ins:
                del temp_utxo[tx_in.outpoint]

        # After all tx are valid, update real utxo set
        for tx in block.txns:
            self.update_utxo_set(tx,True)

        # Update (clean) mempool HOMEWORK
        self.mempool = [tx for tx in self.mempool if tx.id not in [btx.id for btx in block.txns]]
        # update self.blocks
        self.blocks.append(block)

        logger.info(f"Block accepted: height={len(self.blocks) - 1}")
        # TODO Block propagation
        for peer_address in self.peer_addresses:
            send_message(peer_address, "block", block)

    def fetch_utxos(self, public_key, zero_conf):
        if zero_conf:
            return [utxo for utxo in self.zq_utxo_set.values()
                if utxo.public_key == public_key]
        else:
            return [utxo for utxo in self.utxo_set.values()
                if utxo.public_key == public_key]

    def fetch_balance(self, public_key, zero_conf):
        # Fetch utxos associated with this public key
        utxos = self.fetch_utxos(public_key, zero_conf)
        # Sum the amounts
        return sum([tx_out.amount for tx_out in utxos])

def prepare_simple_tx(utxos, sender_private_key, recipient_public_key, amount):
    sender_public_key = sender_private_key.get_verifying_key()

    # Construct tx.tx_outs
    tx_ins = []
    tx_in_sum = 0
    for tx_out in utxos:
        tx_ins.append(TxIn(tx_id=tx_out.tx_id, index=tx_out.index, signature=None))
        tx_in_sum += tx_out.amount
        if tx_in_sum >= amount:
            break

    # Make sure sender can afford it
    assert tx_in_sum >= amount

    # Construct tx.tx_outs
    tx_id = uuid.uuid4()
    change = tx_in_sum - amount
    tx_outs = [
        TxOut(tx_id=tx_id, index=0, amount=amount, public_key=recipient_public_key),
        TxOut(tx_id=tx_id, index=1, amount=change, public_key=sender_public_key),
    ]

    # Construct tx and sign inputs
    tx = Tx(id=tx_id, tx_ins=tx_ins, tx_outs=tx_outs)
    for i in range(len(tx.tx_ins)):
        tx.sign_input(i, sender_private_key)

    return tx

##########
# Mining #
##########

DIFFICULTY_BITS = 20
POW_TARGET = 2 ** (256 - DIFFICULTY_BITS)
mining_interrupt = threading.Event()

def mine_block(block):
    while(block.proof >= POW_TARGET):
        if mining_interrupt.is_set():
            mining_interrupt.clear()
            return None
        block.nonce += 1
    return block

def mine_forever():
    logger.info("Miner started")
    while True:
        unmined_block = Block(
            txns = node.mempool,
            prev_id = node.blocks[-1].id,
            nonce = random.randint(0, 10000000),
        )
        mined_block = mine_block(unmined_block)
        if mined_block:
            logger.info("")
            logger.info("Mined new Block")
            node.handle_block(mined_block)
        else:
            logger.info("")
            logger.info("Mining Interrupted")

def mine_genesis_block():
    global node
    unmined_block = Block(
        txns = [],
        prev_id = None,
        nonce = 0,
    )
    mined_block = mine_block(unmined_block)
    node.blocks.append(mined_block)
    # update UTXO set, award coinbase, etc

##############
# Networking #
##############
def prepare_message(command, data):
    return {
        "command": command,
        "data": data,
    }


class TCPHandler(socketserver.BaseRequestHandler):

    def respond(self, command, data):
        response = prepare_message(command, data)
        return self.request.sendall(serialize(response))

    def handle(self):
        message_bytes = self.request.recv(1024*8).strip()
        message = deserialize(message_bytes)
        command = message["command"]
        data = message["data"]

        # logger.info(f"Received {message}")
        if command == "ping":
            self.respond(command="pong", data="")

        if command == "tx":
            try:
                node.handle_tx(data)
                self.respond(command="tx-response", data="accepted")
            except:
                self.respond(command="tx-response", data="rejected")

        if command == "block":
            # handle only if it builds on the tip
            if data.prev_id == node.blocks[-1].id:
                node.handle_block(data)
                # interrupt mining thread
                mining_interrupt.set()

        if command == "utxos":
            balance = node.fetch_utxos(data, True)
            self.respond(command="utxos-response", data=balance)

        if command == "balance":
            balance = node.fetch_balance(data, True)
            self.respond(command="balance-response", data=balance)

def serve():
    logger.info("Server started")
    server = socketserver.TCPServer(ADDRESS, TCPHandler)
    server.serve_forever()

def send_message(address, command, data, response=False):
    message = prepare_message(command, data)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(address)
        s.sendall(serialize(message))
        if response:
            return deserialize(s.recv(5000))

def external_address(node):
    i = int(node[-1])
    port = PORT + i
    return ("localhost", port)

#######
# CLI #
#######
def main(args):
    if args["serve"]:
        global node
        node = Node()

        # TODO mine genesis block
        mine_genesis_block()
        # start a server thread
        server_thread = threading.Thread(target=serve, name="server")
        server_thread.start()

        miner_thread = threading.Thread(target=mine_forever, name="miner")
        miner_thread.start()

    elif args["ping"]:
        response = send_message(ADDRESS, "ping", "", response=True)
        print(response)
    elif args["balance"]:
        name = args["<name>"]
        address = external_address(args["--node"])
        public_key = user_public_key(name)
        response = send_message(address, "balance", public_key, response=True)
        print(response)
    elif args["tx"]:
        # Grab parameters
        nodes = args["--node"].split(',');
        addresses = list(map(external_address, nodes))
        sender_private_key = user_private_key(args["<from>"])
        sender_public_key = sender_private_key.get_verifying_key()
        recipient_private_key = user_private_key(args["<to>"])
        recipient_public_key = recipient_private_key.get_verifying_key()
        amount = int(args["<amount>"])

        # Fetch utxos available to spend
        response = send_message(addresses[0], "utxos", sender_public_key, response=True)
        utxos = response["data"]

        # Prepare transaction
        tx = prepare_simple_tx(utxos, sender_private_key, recipient_public_key, amount)

        # send to node
        for addr in addresses:
            response = send_message(addr, "tx", tx, response=True)
        print(response)
    else:
        print("Invalid commands")


if __name__ == '__main__':
    #print(docopt(__doc__))
    main(docopt(__doc__))
