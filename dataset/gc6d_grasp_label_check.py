import numpy as np

p = "/data2/robotarm/dataset/GraspClutter6D/economic_grasp_label_300views_extend_angle/000005_labels.npz"
x = np.load(p)

for k in x.files:
    print(k, x[k].shape, x[k].dtype)

print("obj_ids:", x["obj_ids"])
print("collision_arr_indices:", x["collision_arr_indices"])