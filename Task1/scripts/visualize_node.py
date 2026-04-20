import numpy as np
import matplotlib.pyplot as plt
import os

data_root = "/data/HECKTOR2025/Task1"
case_id = "CHUS-027"

ct_path = os.path.join(data_root, "imagesTr_resampled_npy", f"{case_id}_ct.npz")
pet_path = os.path.join(data_root, "imagesTr_resampled_npy", f"{case_id}_pet.npz")
label_path = os.path.join(data_root, "labelsTr_resampled_npy", f"{case_id}_label.npz")

ct_data = np.load(ct_path, allow_pickle=True)
pet_data = np.load(pet_path, allow_pickle=True)
label_data = np.load(label_path, allow_pickle=True)

print("CT keys:", ct_data.files)
print("PET keys:", pet_data.files)
print("Label keys:", label_data.files)

ct = ct_data["image"][0]
pet = pet_data["image"][0]
label = label_data["image"][0]

print("raw shape:", ct_data["image"].shape)
print("after squeeze shape:", ct.shape)

z_indices = np.where(label == 2)[2]
if len(z_indices) == 0:
    print("This case has no GTVn")
    z = ct.shape[2] // 2
else:
    z = int(np.median(z_indices))
    print("Selected z =", z)

save_path = f"{case_id}_z{z}_viz.png"

plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)
plt.title("CT")
plt.imshow(ct[:, :, z], cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("PET")
plt.imshow(pet[:, :, z], cmap="hot")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Label")
plt.imshow(label[:, :, z], cmap="jet", vmin=0, vmax=2)
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("CT + GTVp")
plt.imshow(ct[:, :, z], cmap="gray")
plt.imshow(label[:, :, z] == 1, cmap="Greens", alpha=0.4)
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("CT + GTVn")
plt.imshow(ct[:, :, z], cmap="gray")
plt.imshow(label[:, :, z] == 2, cmap="Reds", alpha=0.5)
plt.axis("off")

plt.subplot(2, 3, 6)
plt.title("PET + GTVn")
plt.imshow(pet[:, :, z], cmap="hot")
plt.imshow(label[:, :, z] == 2, cmap="Blues", alpha=0.4)
plt.axis("off")

plt.tight_layout()
plt.savefig(save_path, dpi=200, bbox_inches="tight")
print(f"Saved to {save_path}")