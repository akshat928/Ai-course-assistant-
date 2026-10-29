# Unit 2: Applied Cryptography

## Symmetric Encryption

Symmetric encryption uses the same secret key to encrypt and decrypt. It is fast and suited to bulk data. AES (Advanced Encryption Standard) is the standard symmetric cipher, with 128, 192 or 256-bit keys and a 128-bit block size.

A block cipher needs a mode of operation to encrypt data longer than one block. ECB mode encrypts each block independently, so identical plaintext blocks produce identical ciphertext blocks and leak patterns; it should never be used. CBC mode chains each block to the previous one using an initialization vector (IV). GCM mode provides both confidentiality and integrity (authenticated encryption) and is the recommended default.

The main weakness of symmetric encryption is key distribution: both parties must somehow share the secret key securely in advance.

## Asymmetric Encryption

Asymmetric (public-key) cryptography uses a key pair: a public key that anyone can know, and a private key kept secret. Data encrypted with the public key can only be decrypted with the matching private key. RSA bases its security on the difficulty of factoring the product of two large primes. Elliptic Curve Cryptography (ECC) achieves similar security with much smaller keys; a 256-bit ECC key is roughly as strong as a 3072-bit RSA key.

Asymmetric encryption is slow, so real systems use it only to exchange or agree on a symmetric session key. This combination is called hybrid encryption, and it is how TLS works.

## Hashing

A cryptographic hash function maps input of any size to a fixed-size digest. A good hash is one-way (you cannot recover the input from the digest), collision-resistant (it is infeasible to find two inputs with the same digest), and shows the avalanche effect (a one-bit change in input changes about half the output bits). SHA-256 is widely used. MD5 and SHA-1 are broken for collision resistance and should not be used for security.

Passwords must never be stored in plain text or with a plain fast hash. Instead, use a slow, salted password hashing function such as bcrypt, scrypt or Argon2. A salt is a random value stored alongside each hash; it makes identical passwords hash differently and defeats precomputed rainbow tables.

## Digital Signatures and TLS

A digital signature proves who created a message and that it has not been altered. The sender hashes the message and signs the hash with their private key; anyone can verify the signature with the sender's public key.

TLS (Transport Layer Security) secures HTTPS. During the handshake the server presents a certificate signed by a Certificate Authority (CA), which binds the server's public key to its domain name. The client verifies the certificate chain up to a trusted root CA. The two sides then use an ephemeral Diffie-Hellman key exchange (ECDHE) to agree on session keys, which provides forward secrecy: compromising the server's long-term private key later does not reveal past sessions. Application data is then encrypted with a symmetric cipher such as AES-GCM.
