import numpy as np
import os
import pandas as pd

experiment_root = 'experiment'
camera_type = 'realsense'
model_list = ['economicgrasp_multi_depth', 'economicgrasp_multi_depth_cls', 'economicgrasp_c1_detach']
noise_type = None

column = ['AP', 'AP0.8', 'AP0.4', 'AP', 'AP0.8', 'AP0.4', 'AP', 'AP0.8', 'AP0.4', 'AP_mean']
split_data = []
epoch_data = []
for model in model_list:
    root = os.path.join(experiment_root, model)
    data = []
    split_ap = []
    for split in ['seen', 'similar', 'novel']:
    # for split in ['seen']:
        res = np.load(os.path.join(root, 'ap_test_{}_{}.npy'.format(split, camera_type)))

        ap_top50 = np.mean(res[:, :, :50, :])
        print('\nEvaluation Result of Top 50 Grasps:\n----------\n{}, AP {}={:6f}'.format(camera_type, split, ap_top50))

        ap_top50_0dot2 = np.mean(res[..., :50, 0])
        print('----------\n{}, AP0.2 {}={:6f}'.format(camera_type, split, ap_top50_0dot2))

        ap_top50_0dot4 = np.mean(res[..., :50, 1])
        print('----------\n{}, AP0.4 {}={:6f}'.format(camera_type, split, ap_top50_0dot4))

        ap_top50_0dot6 = np.mean(res[..., :50, 2,])
        print('----------\n{}, AP0.6 {}={:6f}'.format(camera_type, split, ap_top50_0dot6))

        ap_top50_0dot8 = np.mean(res[..., :50, 3])
        print('----------\n{}, AP0.8 {}={:6f}'.format(camera_type, split, ap_top50_0dot8))

        split_ap.append(ap_top50)
        data.extend([ap_top50, ap_top50_0dot8, ap_top50_0dot4])

    data.extend([np.mean(split_ap)])
    epoch_data.append(data)
    split_data.append(split_ap)
    
if noise_type is not None:
    save_column = ['AP_seen', 'AP_similar', 'AP_novel']
    data_table = pd.DataFrame(columns=save_column, index=model_list, data=split_data)
    data_table.to_csv(f'{noise_type}_noise.csv')
    
for model_name, data in zip(model_list, epoch_data):
    print(model_name, data)
    print("\t")

# print(split_cf_rate)