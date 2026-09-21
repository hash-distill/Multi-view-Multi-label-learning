"""Forensic check of the iaprtc12 view order.

Six papers (TOCL, UGRFS, DHLI, EF2FS, I2VSLC) all tabulate the 5-view
Corel5K / IAPRTC12 layout as

    DH(100), DHV3H1(300), GIST(512), HHV3H1(300), HH(100)   -> [100,300,512,300,100]

while data/iaprtc12.mat holds [100, 300, 512, 100, 300], i.e. positions 4 and 5
look swapped. "V3H1" suggests a 3-level spatial pyramid over a 100-dim base
histogram, so a 300-dim V3H1 view should contain its 100-dim parent as one of
three 100-wide blocks. That gives a way to identify which physical view is which:
match each 100-dim view against every 100-wide block of each 300-dim view.
"""
import numpy as np
import scipy.io as io


def zscore(a):
    a = a - a.mean(0, keepdims=True)
    s = a.std(0, keepdims=True)
    s[s == 0] = 1.0
    return a / s


def block_sim(A, B):
    """Mean matched-column Pearson correlation between A and B (same width)."""
    Az, Bz = zscore(A), zscore(B)
    num = (Az * Bz).sum(0)
    den = np.sqrt((Az ** 2).sum(0) * (Bz ** 2).sum(0))
    den[den == 0] = 1.0
    return float(np.mean(num / den))


def main():
    mat = io.loadmat("data/iaprtc12.mat")
    v = mat["view"]
    parts = [np.asarray(v[0][i], dtype=float) for i in range(v.shape[1])]
    dims = [p.shape[1] for p in parts]
    print("repo view dims :", dims)
    for i, p in enumerate(parts):
        print(f"  view{i + 1}: dim={p.shape[1]:4d} mean={p.mean():8.4f} "
              f"max={p.max():8.1f} zero_frac={(p == 0).mean():.3f}")

    hundred = [i for i, d in enumerate(dims) if d == 100]
    threehundred = [i for i, d in enumerate(dims) if d == 300]
    print(f"\n100-dim views    : {[i + 1 for i in hundred]}")
    print(f"300-dim views    : {[i + 1 for i in threehundred]}")

    print("\nmean matched-column correlation of each 100-dim view against each")
    print("100-wide block of each 300-dim view (higher = parent/child relation)\n")
    header = f"{'100-dim view':>14} |"
    for j in threehundred:
        header += f"  view{j+1}.blk0  blk1  blk2 |"
    print(header)
    print("-" * len(header))
    for i in hundred:
        A = parts[i]
        line = f"{'view' + str(i + 1):>14} |"
        for j in threehundred:
            B = parts[j]
            sims = [block_sim(A, B[:, k * 100:(k + 1) * 100]) for k in range(3)]
            line += f"  {sims[0]:11.4f} {sims[1]:5.3f} {sims[2]:5.3f} |"
        print(line)

    print("\ncontrol (unrelated pairs should be near 0):")
    print(f"  view{hundred[0]+1} vs view{hundred[1]+1} : "
          f"{block_sim(parts[hundred[0]], parts[hundred[1]]):.4f}")
    print(f"  view{threehundred[0]+1}.blk0 vs view{threehundred[1]+1}.blk0 : "
          f"{block_sim(parts[threehundred[0]][:, :100], parts[threehundred[1]][:, :100]):.4f}")


if __name__ == "__main__":
    main()
