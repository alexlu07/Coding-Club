from Crypto.Util.number import *

flag = bytes_to_long(open('flag.txt', 'rb').read().strip())
big_dog = 314159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223
p = 1
q = 1
while not isPrime(p) or not isPrime(q):
    p = getPrime(512)
    q = p + 2 * big_dog

assert isPrime(p)
assert isPrime(q)

N = p * q
e = 65537
c = pow(flag, e, N)

print(N)
print(e)
print(c)