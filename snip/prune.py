import time
from PIL import Image
import numpy as np
from numpy import linalg as LA
import os
import sys

def prune(args, model, sess, dataset):
    print('|========= START PRUNING =========|')
    t_start = time.time()
    batch = dataset.get_next_batch('train', args.batch_size)
    feed_dict = {}
    feed_dict.update({model.inputs[key]: batch[key] for key in ['input', 'label']})
    feed_dict.update({model.compress: True, model.is_train: False, model.pruned: False})
    result = sess.run([model.outputs, model.sparsity], feed_dict)

    # For improvement
    # mask = result[0]['mask']
    # for k in mask.keys():
    #     mask_vec = np.reshape(mask[k], [-1])
    #     size_of_vec = mask_vec.shape[0]
    #     alived_elements = np.sum(mask_vec)
    #     print(k)
    #     print(size_of_vec)
    #     print(alived_elements)
    #     print("sparsity: ", 1 - alived_elements / size_of_vec)
    # sys.exit(1)

    # For expr4
    # mkdir_for_path(args.path_visualization_result)
    # path = os.path.join(args.path_visualization_result, str(args.target_label))
    # mkdir_for_path(path)
    # file_name = 'weight1_for_visualization-' + str(model.target_sparsity) + '.jpeg'
    # file_path = os.path.join(path, file_name)
    
    # weight1_for_visualization = result[0]['weight1_for_visualization']
    # print(weight1_for_visualization)
    # print(np.mean(weight1_for_visualization))
    # img = Image.fromarray(weight1_for_visualization, 'L')
    # img.save(file_path)

    print('Pruning: {:.3f} global sparsity (t:{:.1f})'.format(result[-1], time.time() - t_start))

def mkdir_for_path(path):
    if not os.path.exists(path):
        original_umask = os.umask(0)
        desired_permission = 0o0777
        os.makedirs(path, desired_permission)
        os.umask(original_umask)