# import torch
#
# vecs = []
# while True:
#     for i in range(1,8):
#         try:
#             vecs.append(torch.randn((10000,10000)).to(i))
#         except RuntimeError:
#             try:
#                 vecs.append(torch.randn((100,100)).to(i))
#             except RuntimeError:
#                 pass


import numpy as np

x ={
    "CC0117": 0.34414670579604767,
    "CC0092": 0.515753389449676,
    "CC0106": 0.6378637507589305,
    "CC0066": 0.6600772585284231,
    "CC0061": 0.5265767552802033,
    "CC0069": 0.6736109582419103,
    "CC0096": 0.5356782246070549,
    "CC0102": 0.5470201578409791,
    "CC0109": 0.7263345205344471,
    "CC0107": 0.7316189173475895
}
print(np.mean(list(x.values())))