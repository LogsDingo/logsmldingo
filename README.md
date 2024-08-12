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




