# params_transform = dict()
# params_transform['mode'] = 5
# # === aug_scale ===
# params_transform['scale_min'] = 0.5
# params_transform['scale_max'] = 1.1
# params_transform['scale_prob'] = 1
# params_transform['target_dist'] = 0.6       #
# # === aug_rotate ===
# params_transform['max_rotate_degree'] = 40
#
# # ===
# params_transform['center_perterb_max'] = 40
#
# # === aug_flip ===
# params_transform['flip_prob'] = 0.5
#
# params_transform['np'] = 56
# params_transform['sigma'] = 7.0
# params_transform['limb_width'] = 1.289


#
# ---------------------------- training ---------------------------
params_transform = dict()
params_transform['mode'] = 5
# === aug_scale ===
params_transform['scale_min'] = 0.8
params_transform['scale_max'] = 1.1
params_transform['scale_prob'] = 0.6
params_transform['target_dist'] = 0.6
# === aug_rotate ===
params_transform['max_rotate_degree'] = 10

# ===
params_transform['center_perterb_max'] = 20

# === aug_flip ===
params_transform['flip_prob'] = 0.0  # No flip for robots

params_transform['np'] = 56
params_transform['sigma'] = 6.0
params_transform['limb_width'] = 1.

# === ===

IMAGE_W = 640
IMAGE_H = 360

NUM_KEYPOINTS = 4
NUM_LIMBS = 4
