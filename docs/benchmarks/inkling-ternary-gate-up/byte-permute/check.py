def original(x):
    x=(x|(x<<8))&0x00ff00ff
    x=(x|(x<<4))&0x0f0f0f0f
    x=(x|(x<<2))&0x33333333
    return (x&0x11111111)*7 | ((x&0x22222222)<<2)
def perm(lut,selector):
    return sum(((lut>>(8*((selector>>(4*i))&3)))&255)<<(8*i) for i in range(4))
for x in range(65536):
    assert original(x)==(perm(0x0f080700,x&0x3333)|perm(0xf0807000,(x>>2)&0x3333))
print("65,536 halfwords matched")
