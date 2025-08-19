import matplotlib.pyplot as plt
import numpy as np

attn_mat = np.load('./attention_matrix/attn_cls_fwd2_head0.txt.npz')['data']
print(attn_mat.shape)
