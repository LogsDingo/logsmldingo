Practical 1A
pip install pycryptodome
from Crypto.PublicKey import RSA
key = RSA.generate(2048)
private_key = key.export_key()
with open("private_key.pem", "wb") as private_file:
    private_file.write(private_key)
public_key = key.publickey().export_key()
with open("public_key.pem", "wb") as public_file:
    public_file.write(public_key)
print("Private and public keys have been generated and saved as 'private_key.pem' and 'public_key.pem'.")
**********************************************************************************************************
Practical 1B
import binascii
import collections
import datetime
from Crypto.Hash import SHA
from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
from Crypto import Random
class Client:
def __init__(self):
random = Random.new().read
self._private_key = RSA.generate(1024, random)
self._public_key = self._private_key.publickey()
self._signer = PKCS1_v1_5.new(self._private_key)
@property
def identity(self):
return
binascii.hexlify(self._public_key.exportKey(format="DER")).decode("asci
i")
class Transaction:
def __init__(self, sender, recipient, value):
self.sender = sender
self.recipient = recipient
self.value = value
self.time = datetime.datetime.now()
def to_dict(self):
identity = "Genesis" if self.sender == "Genesis" else
self.sender.identity
return collections.OrderedDict(
{
"sender": identity,
"recipient": self.recipient,
"value": self.value,
"time": self.time,
}
)
def sign_transaction(self):
private_key = self.sender._private_key
signer = PKCS1_v1_5.new(private_key)
h = SHA.new(str(self.to_dict()).encode("utf8"))
return binascii.hexlify(signer.sign(h)).decode("ascii")
UDIT = Client()
UGC = Client()
t = Transaction(UDIT, UGC.identity, 5.0)
print("\nTransaction Recipient:\n", t.recipient) # 
print("\nTransaction Sender:\n", t.sender) print("\nTransaction 
Value:\n", t.value)
signature = t.sign_transaction()
print("\nSignature:\n", signature)

**********************************************************************************************************
Practical 1C
# Practical1 C
import binascii
import collections
import datetime
from Crypto.Hash import SHA
from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
from Crypto import Random
class Client:
def __init__(self):
random = Random.new().read
self._private_key = RSA.generate(1024, random)
self._public_key = self._private_key.publickey()
self._signer = PKCS1_v1_5.new(self._private_key)
@property
def identity(self):
return
binascii.hexlify(self._public_key.exportKey(format="DER")).decode("asci
i")
class Transaction:
def __init__(self, sender, recipient, value):
self.sender = sender
self.recipient = recipient
self.value = value
self.time = datetime.datetime.now()
def to_dict(self):
identity = "Genesis" if self.sender == "Genesis" else
self.sender.identity
return collections.OrderedDict(
{
"sender": identity,
"recipient": self.recipient,
"value": self.value,
"time": self.time,
}
)
def sign_transaction(self):
private_key = self.sender._private_key
signer = PKCS1_v1_5.new(private_key)
h = SHA.new(str(self.to_dict()).encode("utf8"))
return binascii.hexlify(signer.sign(h)).decode("ascii")
def display_transaction(transaction):
# for transaction in transactions:
dict = transaction.to_dict()
print("sender: " + dict['sender'])
print('-----')
print("recipient: " + dict['recipient'])
print('-----')
print("value: " + str(dict['value']))
print('-----')
print("time: " + str(dict['time']))
print('-----')
UDIT = Client()
UGC = Client()
AICTE = Client()
MU = Client()
t1 = Transaction(UDIT, UGC.identity, 15.0)
t1.sign_transaction()
transactions = [t1]
t2 = Transaction(UDIT, AICTE.identity, 6.0)
t2.sign_transaction()
transactions.append(t2)
t3 = Transaction(UGC, MU.identity, 2.0)
t3.sign_transaction()
transactions.append(t3)
t4 = Transaction(AICTE, UGC.identity, 4.0)
t4.sign_transaction()
transactions.append(t4)
for transaction in transactions:
Transaction.display_transaction(transaction)
print(" ")

**********************************************************************************************************
Practical 1D
#Practical1 D
import binascii
import collections
import datetime
from Crypto.Hash import SHA
from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
from Crypto import Random
class Client:
def __init__(self):
random = Random.new().read
self._private_key = RSA.generate(1024, random)
self._public_key = self._private_key.publickey()
self._signer = PKCS1_v1_5.new(self._private_key)
@property
def identity(self):
return
binascii.hexlify(self._public_key.exportKey(format="DER")).decode("asci
i")
class Transaction:
def __init__(self, sender, recipient, value):
self.sender = sender
self.recipient = recipient
self.value = value
self.time = datetime.datetime.now()
def to_dict(self):
identity = "Genesis" if self.sender == "Genesis" else
self.sender.identity
return collections.OrderedDict(
{
"sender": identity,
"recipient": self.recipient,
"value": self.value,
"time": self.time,
}
)
def sign_transaction(self):
private_key = self.sender._private_key
signer = PKCS1_v1_5.new(private_key)
h = SHA.new(str(self.to_dict()).encode("utf8"))
return binascii.hexlify(signer.sign(h)).decode("ascii")
def display_transaction(transaction):
# for transaction in transactions:
dict = transaction.to_dict()
print("sender: " + dict['sender'])
print('-----')
print("recipient: " + dict['recipient'])
print('-----')
print("value: " + str(dict['value']))
print('-----')
print("time: " + str(dict['time']))
print('-----')
class Block:
def __init__(self, client):
self.verified_transactions = []
self.previous_block_hash = ""
self.Nonce = ""
self.client = client
def dump_blockchain(blocks):
print(f"\nNumber of blocks in the chain: {len(blocks)}")
for i, block in enumerate(blocks):
print(f"block # {i}")
for transaction in block.verified_transactions:
Transaction.display_transaction(transaction)
print(" ")
print(" ")
UDIT = Client()
t0 = Transaction("Genesis", UDIT.identity, 500.0)
block0 = Block(UDIT)
block0.previous_block_hash = ""
NONCE = None
block0.verified_transactions.append(t0)
digest = hash(block0)
last_block_hash = digest
TPCoins = [block0]
dump_blockchain(TPCoins)
**********************************************************************************************************
Practical 1E
#Practical1 E
import hashlib
def sha256(message):
return hashlib.sha256(message.encode("ascii")).hexdigest()
def mine(message, difficulty=1):
assert difficulty >= 1
prefix = "1" * difficulty
for i in range(1000):
digest = sha256(str(hash(message)) + str(i))
if digest.startswith(prefix):
print(f"After {str(i)} iterations found nonce: {digest}")
return digest
print(mine("test message", 2))

**********************************************************************************************************
Practical 1F
#Practical1 F
import datetime
import hashlib
class Block:
def __init__(self, data, previous_hash):
self.timestamp = datetime.datetime.now(datetime.timezone.utc)
self.data = data
self.previous_hash = previous_hash
self.hash = self.calc_hash()
def calc_hash(self):
sha = hashlib.sha256()
hash_str = self.data.encode("utf-8")
sha.update(hash_str)
return sha.hexdigest()
blockchain = [Block("First block", "0")]
blockchain.append(Block("Second block", blockchain[0].hash))
blockchain.append(Block("Third block", blockchain[1].hash))
# Dumping the blockchain
for block in blockchain:
print(
f"Timestamp: {block.timestamp}\nData: {block.data}\nPrevious Hash: 
{block.previous_hash}\nHash: {block.hash}\n"
)
**********************************************************************************************************

Practical6
from hashlib import sha256
def hashGenerator(text):
    return sha256(text.encode("ascii")).hexdigest()
    

class Block:
    def __init__(self,data,hash,prev_hash):
        self.data=data
        self.hash=hash
        self.prev_hash=prev_hash

class Blockchain:
    def __init__(self):
      hashLast=hashGenerator('gen_last')
      hashStart=hashGenerator('gen_hash')
      genesis=Block('gen-data',hashStart,hashLast)
      self.chain=[genesis]
      
    def add_block(self,data):
        prev_hash=self.chain[-1].hash
        hash=hashGenerator(data+prev_hash)
        block=Block(data,hash,prev_hash)
        self.chain.append(block)

bc=Blockchain()
bc.add_block('1')
bc.add_block('2')
bc.add_block('3')

for block in bc.chain:
    print(block.__dict__)

Practical5
from hashlib import sha256
import time
MAX_NONCE = 100000000000
def hashGenerator(text):
 return sha256(text.encode("ascii")).hexdigest()
def mine(block_number, transactions, previous_hash, prefix_zeros):
 prefix_str = '0'*prefix_zeros 
 for nonce in range(MAX_NONCE):
 text = str(block_number) + transactions + previous_hash + str(nonce)
 new_hash = hashGenerator(text)
 if new_hash.startswith(prefix_str):
 print(f"Successfully mined Ethers with nonce value : {nonce}")
 return new_hash
 raise BaseException(f"Couldn't find correct hash after trying {MAX_NONCE} times")
if __name__ == '__main__':
 transactions = '''
 Jhon->Paul->77,
 Akon->Bruno->18
 '''
 difficulty = 4 
 start = time.time()
 print("Ether mining started.")
 new_hash = 
mine(5,transactions,'0000000xa036944e29568d0cff17edbe038f81208fecf9a66be9a2b8321c6
ec7', difficulty)
total_time = str((time.time() - start))
 print(f"Ether mining finished.")
 print(f"Ether Mining took : {total_time} seconds")
print(f"Calculated Hash = {new_hash}")
**********************************************************************************************************
Practical6
from hashlib import sha256
def hashGenerator(text):
 return sha256(text.encode("ascii")).hexdigest()
class Block:
 def __init__(self,data,hash,prev_hash):
 self.data=data
 self.hash=hash
 self.prev_hash=prev_hash
class Blockchain:
 def __init__(self):
 hashLast=hashGenerator('gen_last')
 hashStart=hashGenerator('gen_hash')
 genesis=Block('gen-data',hashStart,hashLast)
 self.chain=[genesis]
 def add_block(self,data):
 prev_hash=self.chain[-1].hash
 hash=hashGenerator(data+prev_hash)
 block=Block(data,hash,prev_hash)
 self.chain.append(block)
bc=Blockchain()
bc.add_block('1')
bc.add_block('2')
bc.add_block('3')
for block in bc.chain:
 print(block.__dict__)
**********************************************************************************************************



