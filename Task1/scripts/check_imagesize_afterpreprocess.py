import os
import numpy as np

OUT_DIR = "/home/mi2488/hot/datasets/AutoPET/fdg_bodycrop_ssl_npz"

shapes = []

for f in os.listdir(OUT_DIR):
    if f.endswith(".npz"):
        path = os.path.join(OUT_DIR, f)
        data = np.load(path, allow_pickle=True)
        img = data["image"]
        shapes.append(img.shape)

print("num cases:", len(shapes))
print("first 10 shapes:")
for s in shapes[:10]:
    print(s)

xs = [s[1] for s in shapes]
ys = [s[2] for s in shapes]
zs = [s[3] for s in shapes]

print("X min/max/mean:", min(xs), max(xs), sum(xs) / len(xs))
print("Y min/max/mean:", min(ys), max(ys), sum(ys) / len(ys))
print("Z min/max/mean:", min(zs), max(zs), sum(zs) / len(zs))


# num cases: 1014
# first 10 shapes:
# (1, 130, 135, 326)
# (1, 115, 111, 326)
# (1, 146, 125, 326)
# (1, 152, 133, 363)
# (1, 134, 115, 326)
# (1, 122, 110, 268)
# (1, 134, 125, 326)
# (1, 137, 115, 284)
# (1, 176, 119, 535)
# (1, 128, 118, 326)
# X min/max/mean: 108 183 139.68540433925048
# Y min/max/mean: 79 148 121.79289940828403
# Z min/max/mean: 200 661 350.03944773175544