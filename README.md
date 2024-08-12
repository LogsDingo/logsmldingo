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

