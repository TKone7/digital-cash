"""
BankNetCoin.

Usage:
  mybanknetcoin.py serve
  mybanknetcoin.py ping
  mybanknetcoin.py balance <name>
  mybanknetcoin.py tx <from> <to> <amount>

Options:
  -h --help     Show this screen.


"""
import socketserver, socket, sys, uuid 
from copy import deepcopy
from ecdsa import SigningKey, SECP256k1
from utils import serialize, deserialize
from identities import user_public_key, user_private_key 
from docopt import docopt

class Tx:

    def __init__(self, id, tx_ins, tx_outs):
        self.id = id
        self.tx_ins = tx_ins
        self.tx_outs = tx_outs

    def sign_input(self, index, private_key):
        message = self.spend_message(index)
        signature = private_key.sign(message)
        self.tx_ins[index].signature = signature

    def verify_input(self, index, public_key):
        print('index in verify_input: ' + str(index) + " " + str(self.id))
        tx_in = self.tx_ins[index]
        message = self.spend_message(index)
        public_key.verify(tx_in.signature, message)
    
    def spend_message(self, index):
        tx_in = self.tx_ins[index]
        outpoint = tx_in.outpoint
        return serialize(outpoint) + serialize(self.tx_outs)

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

class Bank:

    def __init__(self):
        # all transactions
        #self.txs = {}
        # (tx_id, index) -> tx_out
        self.utxo = {}

    def update_utxo(self, tx):
        for tx_in in tx.tx_ins:
            del self.utxo[tx_in.outpoint]
        for tx_out in tx.tx_outs:
            self.utxo[tx_out.outpoint] = tx_out

    def issue(self, amount, public_key):
        id_ = str(uuid.uuid4())
        tx_ins = []
        tx_outs = [TxOut(tx_id=id_, index=0, amount=amount, public_key=public_key)]
        tx = Tx(id=id_, tx_ins=tx_ins, tx_outs=tx_outs)
        #self.txs[tx.id] = tx
        self.update_utxo(tx)
        return tx

    def validate_tx(self, tx):
        in_sum = 0
        out_sum = 0
        for index, tx_in in enumerate(tx.tx_ins):
            # check if tx_in is unspent
            assert tx_in.outpoint in self.utxo

            tx_out = self.utxo[tx_in.outpoint]
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
        # Save to self.txs if it's valid
        self.validate_tx(tx)
        #self.txs[tx.id] = tx
        self.update_utxo(tx)

    def fetch_utxo(self, public_key):
        return [utxo for utxo in self.utxo.values()
                if utxo.public_key == public_key]

    def fetch_balance(self, public_key):
        # Fetch utxo associated with this public key
        unspents = self.fetch_utxo(public_key)
        # Sum the amounts
        return sum([tx_out.amount for tx_out in unspents])

def prepare_message(command, data):
    return {
        "command":command,
        "data":data,
    }

#####################
# Networking Code
#####################

host = "0.0.0.0"
port = 10000
address = (host, port)
bank = Bank()

class MyTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

class TCPHandler(socketserver.BaseRequestHandler):
    
    def respond(self, command, data):
        response = prepare_message(command,data)
        serialized_response = serialize(response)  
        self.request.sendall(serialized_response)
        

    def handle(self):
        message_data = self.request.recv(5000).strip()
        message = deserialize(message_data)
        print(f"got a message {message}")
        command = message["command"]
        
        if command == "ping":
            self.respond("ping", "")

        elif command == "balance":
            public_key = message["data"]
            balance = bank.fetch_balance(public_key)
            self.respond("balance-response", balance)

        elif command == "utxo":
            public_key = message["data"]
            utxo = bank.fetch_utxo(public_key)
            self.respond("utxo-response", utxo)

        elif command == "tx":
            tx = message["data"]
            try:
                bank.handle_tx(tx)
                self.respond("tx-response", "accepted")
            except:
                self.respond("tx-response", "rejected")

            
def serve():
    server = MyTCPServer(address, TCPHandler)
    server.serve_forever();

def send_message(command, data):
    # create instance of a socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(address)
    message = prepare_message(command, data)
    ser_message = serialize(message)
    sock.sendall(ser_message)

    message_data = sock.recv(5000)
    message = deserialize(message_data)
    #print(f"Received {message}")
    return message


def main(args):
    #parse command line args
    if args["ping"]:
        send_message("ping", "")
    elif args["serve"]:
        # issue some coins on server startup
        alice_public_key = user_public_key("alice")
        bank.issue(1000, alice_public_key)
        serve()
    elif args["balance"]:
        name = args["<name>"] 
        public_key = user_public_key(name)

        balance_response = send_message("balance", public_key)
        balance = balance_response["data"]
        print(name + ": " + str(balance))
    elif args["tx"]:
        sender_private_key = user_private_key(args["<from>"])
        sender_public_key = sender_private_key.get_verifying_key()
        recipient_public_key = user_public_key(args["<to>"])
        amount = int(args["<amount>"])

        # fetch sender utxo
        utxo_response = send_message("utxo",sender_public_key)
        utxo = utxo_response["data"]

        # prepare tx
        tx = prepare_simple_tx(
            utxo, 
            sender_private_key,
            recipient_public_key,
            amount,
            )
        # send tx as a message
        tx_response = send_message("tx", tx)
        print("Result: " + tx_response["data"]) 
    else:
        print("not a valid command")

def prepare_simple_tx(utxos, sender_private_key, recipient_public_key, amount):
    sender_public_key = sender_private_key.get_verifying_key()

    # Construct tx.tx_outs
    tx_ins = []
    tx_in_sum = 0
    for tx_out in utxos:
        tx_ins.append(TxIn(tx_id=tx_out.tx_id, index=tx_out.index, signature=None))
        tx_in_sum += tx_out.amount
        if tx_in_sum > amount:
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

if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
