import uuid 
from copy import deepcopy
from ecdsa import SigningKey, SECP256k1
from utils import serialize


def transfer_message(previous_signature, public_key):
    return serialize({
        "previous_signature": previous_signature,
        "next_owner_public_key": public_key,
    })


class Transfer:
    
    def __init__(self, signature, public_key):
        self.signature = signature
        self.public_key = public_key

    def __eq__(self, other):
        return self.signature == other.signature and \
                self.public_key.to_string() == other.public_key.to_string()
class BankCoin:
    
    def __init__(self, transfers):
        self.transfers = transfers
        # create a unique id
        self.id = uuid4()

    def __eq__(self, other):
        return self.id == other.id and \
                self.transfers == other.transfers

    def validate(self):
        # Check the subsequent transfers
        previous_transfer = self.transfers[0]
        for transfer in self.transfers[1:]:
            # Check previous owner signed this transfer using their private key
            assert previous_transfer.public_key.verify(
                transfer.signature,
                transfer_message(previous_transfer.signature, transfer.public_key),

            )
            previous_transfer = transfer

    def transfer(self, owner_private_key, recipient_public_key):
        message = transfer_message(
                previous_signature = self.transfers[-1].signature,
                public_key = recipient_public_key,
                )
        t = Transfer(
                signature = owner_private_key.sign(message),
                public_key = recipient_public_key
                )

        self.transfers.append(t)

class Bank: 
    def __init__(self):
        # coin.id > coin
        self.coins = {}

    def issue(self, public_key):
        # Create the first transfer, signing with the banks private key
        transfer = Transfer(
            signature=None,
            public_key=public_key,
        )
        
        # Create and return the coin with just the issuing transfer
        coin = BankCoin(transfers=[transfer])
        # Put the new coin into the database
        self.coins[coin.id] = deepcopy(coin)
        return coin

    def observe_coin(self, coin):
        last_observations = self.coins[coin.id]
        assert last_observations.transfers == coin.transfers[:-1]
        
        # we validate the whole thing (even though only the newest transfer must be validated)
        coin.validate()

        # update database
        self.coins[coin.id] = deepcopy(coin)        

    def fetch_coins(self, public_key):
        # initiate empty list of coins
        coins = []
        # iterate not over id but over values()
        for coin in self.coins.values():
            # only compare if the current owner of the key equeals the public key from the argument
            if coin.transfers[-1].public_key.to_string() == public_key.to_string():
               coins.append(coin) 
        return coins

