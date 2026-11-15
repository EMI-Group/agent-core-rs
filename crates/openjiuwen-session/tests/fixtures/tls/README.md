# TLS test fixtures

Static certificates for the `rediss://` (TLS) tests of the Redis checkpoint
backend. Nothing here is secret: the certificates are for `localhost` /
`127.0.0.1` only, the CA private keys were deleted right after signing, and the
server private key exists solely so the in-process test server in
`tests/support/resp_server.rs` can complete a handshake.

| File | Contents |
| --- | --- |
| `ca.pem` | the test CA certificate, PEM (client trust anchor) |
| `ca.der` | the same certificate, DER |
| `other-ca.pem` | a second, unrelated CA, PEM ("wrong trust anchor" tests) |
| `other-ca.der` | the same certificate, DER |
| `server.pem` | the server certificate, PEM (`CN=localhost`, `SAN=DNS:localhost,IP:127.0.0.1`) |
| `server.der` | the same certificate, DER (the test server loads this one) |
| `server.key.pem` | the server private key, PEM (PKCS#8) |
| `server.key.der` | the same key, DER (PKCS#8, the test server loads this one) |

Every certificate is valid from `2020-01-01` to `2040-01-01`, so the fixtures
never expire under a normal machine clock. The tests that need a malformed or
unreadable CA file write one into a `tempfile` directory instead of using a
fixture, so no deliberately broken files live here.

## Regeneration

Regenerate from this directory with OpenSSL 3.5 (any 3.x with `-not_before`
works); `openssl` is not on `PATH` in the workspace sandbox, so call the one in
the Nix store explicitly:

```sh
cd crates/openjiuwen-session/tests/fixtures/tls
export PATH=/nix/store/gvglnxscmvx4mh9my0cy0xgi0l1z6l9s-openssl-3.5.9-bin/bin:$PATH

NB=20200101000000Z
NA=20400101000000Z

# 1. The test CA (its key is deleted below: it only signs the server cert).
openssl req -x509 -newkey rsa:2048 -sha256 -nodes -not_before $NB -not_after $NA \
  -keyout ca.key.pem -out ca.pem -subj "/CN=openjiuwen-session test CA" \
  -addext "basicConstraints=critical,CA:TRUE" \
  -addext "keyUsage=critical,keyCertSign,cRLSign"

# 2. A second, unrelated CA for the "wrong trust anchor" tests.
openssl req -x509 -newkey rsa:2048 -sha256 -nodes -not_before $NB -not_after $NA \
  -keyout other-ca.key.pem -out other-ca.pem \
  -subj "/CN=openjiuwen-session other test CA" \
  -addext "basicConstraints=critical,CA:TRUE" \
  -addext "keyUsage=critical,keyCertSign,cRLSign"

# 3. The server key and certificate signing request.
openssl req -newkey rsa:2048 -sha256 -nodes \
  -keyout server.key.pem -out server.csr -subj "/CN=localhost"

# 4. The server certificate, signed by the test CA, valid for localhost.
printf 'subjectAltName=DNS:localhost,IP:127.0.0.1\nextendedKeyUsage=serverAuth\nbasicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\n' > server.ext
openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key.pem -CAcreateserial \
  -sha256 -not_before $NB -not_after $NA -extfile server.ext -out server.pem

# 5. The DER forms the tests load.
openssl x509 -in ca.pem -outform DER -out ca.der
openssl x509 -in other-ca.pem -outform DER -out other-ca.der
openssl x509 -in server.pem -outform DER -out server.der
openssl pkcs8 -topk8 -nocrypt -in server.key.pem -outform DER -out server.key.der

# 6. Drop the intermediate files and every private CA key.
rm -f server.csr server.ext ca.srl ca.key.pem other-ca.key.pem
```

Check the result with:

```sh
openssl x509 -in server.pem -noout -subject -issuer -dates -ext subjectAltName
openssl x509 -in ca.pem -noout -dates -ext basicConstraints
```
