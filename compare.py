import argparse
import os
import glob
import ot
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
import seaborn as sns
import numpy
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from torch.utils.data import DataLoader

from generate import gen_testloss, gen_training_accuracy
import train_func as tf
import utils
import scipy


def get_cost_matrix(feature1, feature2, metric= "cosine"):

    C = scipy.spatial.distance.cdist(feature1, feature2, metric=metric)

    return C


def prolong_batch(batch_vector1, batch_vector2):
    gap = len(batch_vector1) - len(batch_vector2)
    assert gap >= 0

    if gap == 0:
        return batch_vector1, batch_vector2

    rest = abs(gap)
    step = len(batch_vector2)
    extend_list = [batch_vector2]

    while rest > step:
        extend_list.append(batch_vector2)
        rest = rest - step

    copied_samples = batch_vector2[-abs(rest):]
    extend_list.append(copied_samples)
    batch_vector2 = np.concatenate(extend_list, axis=0)

    return batch_vector1, batch_vector2


def align_batch(batch_vector1, batch_vector2):
    gap = len(batch_vector1) - len(batch_vector2)
    if gap >= 0:
        aligned_batch_vector1, aligned_batch_vector2 = prolong_batch(batch_vector1, batch_vector2)
    else:
        aligned_batch_vector1, aligned_batch_vector2 = prolong_batch(batch_vector2, batch_vector1)

    return aligned_batch_vector1, aligned_batch_vector2

def compare_feature(features_before, labels_before, features_after, labels_after, num_classes=4):
    num_sample = len(features_before)

    features_sort_before, labels_sort_before = utils.sort_dataset(features_before.numpy(), labels_before.numpy(),
                            num_classes=num_classes, stack=False)

    features_sort_after, labels_sort_after = utils.sort_dataset(features_after.numpy(), labels_after.numpy(),
                            num_classes=num_classes, stack=False)

    outer_scores = []
    inner_scores = []
    before_distr = []
    after_distr = []

    for class_feature_before, class_feature_after in zip(features_sort_before, features_sort_after):
        # inner class score
        _, s_b, _ = np.linalg.svd(class_feature_before)
        _, s_a, _ = np.linalg.svd(class_feature_after)

        inner_scores.append(np.log(numpy.prod(s_a[:10])/numpy.prod(s_b[:10])))

        # between-class score
        class_num_before = len(class_feature_before)
        class_num_after = len(class_feature_after)
        aug = 2.0 * (class_num_after > class_num_before) - 1.0
        minor = 2.0 * (class_num_before < num_sample/num_classes) - 1.0
        before_distr.append(class_num_before/num_sample)
        after_distr.append(class_num_after/num_sample)

        class_num_samples = max(len(class_feature_before), len(class_feature_after))
        aligned_class_feature_before, aligned_class_feature_after = \
            align_batch(class_feature_before, class_feature_after)
        cost_matrix = get_cost_matrix(aligned_class_feature_before, aligned_class_feature_after)
        a = numpy.ones(class_num_samples) / class_num_samples
        b = numpy.ones(class_num_samples) / class_num_samples
        Wd = ot.emd2(a, b, cost_matrix)

        outer_scores.append(minor*aug*Wd)

    return inner_scores, outer_scores, before_distr, after_distr


def tsne_vis(feature_origin, pre_labels_origin, feature_before, pre_labels_before, feature_after, pre_labels_after, \
             path="tsne_figs"):
    num_samples = len(feature_origin)
    assert len(feature_origin) == len(feature_before) == len(feature_after)
    all_features = np.concatenate([feature_origin, feature_before, feature_after], axis=0)
    all_labels = np.concatenate([labels_origin, labels_before, labels_after], axis=0)

    tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=300, metric="cosine")
    tsne_res_all = tsne.fit_transform(all_features)

    tsne_res_before_mix = tsne_res_all[: 2*num_samples]
    tsne_res_after_mix = tsne_res_all[-2*num_samples:]
    pre_labels_before_mix = all_labels[: 2*num_samples]
    pre_labels_after_mix = all_labels[-2*num_samples:]

    tsne_res_org = tsne_res_all[: num_samples]
    tsne_res_before = tsne_res_all[num_samples: 2*num_samples]
    tsne_res_after = tsne_res_all[-num_samples:]

    tsne_list = [tsne_res_org, tsne_res_before, tsne_res_after, tsne_res_before_mix, tsne_res_after_mix]
    pre_labels_list = [pre_labels_origin, pre_labels_before, pre_labels_after, pre_labels_before_mix, pre_labels_after_mix]
    classes = [i for i in range(4)]
    name_list = ["tsne_res_org", "tsne_res_before", "tsne_res_after", "tsne_res_before_mix", "tsne_res_after_mix"]

    for i, (tsne_res, pre_labels) in enumerate(zip(tsne_list, pre_labels_list)):
        plt.figure(figsize=(10, 10))
        for label in classes:
            indices = pre_labels == label
            plt.scatter(tsne_res[indices, 0], tsne_res[indices, 1], label=str(label))
        plt.legend()
        plt.title(f"{name_list[i]}")
        plt.savefig(f"{path}/{name_list[i]}.png")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ploting')
    parser.add_argument('--model_dir', type=str, help='base directory for saving PyTorch model.')
    parser.add_argument('--before', type=str, help='before augmentation dir')
    parser.add_argument('--after', type=str, help='after augmentation dir')
    parser.add_argument('--epoch', type=int, default=None, help='which epoch for evaluation')
    args = parser.parse_args()
    params = utils.load_params(args.model_dir)
    net, epoch = tf.load_checkpoint(args.model_dir, args.epoch, eval_=True)
    transforms = tf.load_transforms('test')

    origin = tf.load_trainset("compare", transforms, file_path=args.before, max_imgNum=1000)
    trainloader_origin = DataLoader(origin, batch_size=64, num_workers=4)
    features_origin, labels_origin = tf.get_features(net, trainloader_origin)

    before = tf.load_trainset("compare", transforms, file_path=args.before, max_imgNum=1000)
    trainloader_before = DataLoader(before, batch_size=64, num_workers=4)
    features_before, labels_before = tf.get_features(net, trainloader_before)

    after = tf.load_trainset("compare", transforms, file_path=args.after, max_imgNum=1000)
    trainloader_after = DataLoader(after, batch_size=64, num_workers=4)
    features_after, labels_after = tf.get_features(net, trainloader_after)

    inner_score_org, outer_score_org, distr_org, distr_before = compare_feature(features_origin, labels_origin, features_before, labels_before,
                                  num_classes=before.num_classes)

    inner_score_sty, outer_score_sty, distr_org, distr_after = compare_feature(features_origin, labels_origin, features_after, labels_after, \
                                  num_classes=before.num_classes)

    import csv

    file_name = "feature_compare_res.csv"
    with open(file_name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        names = ["class", "inner_org", "inner_sty", "inter_org", "inter_sty", "distr_org", "distr_before", "distr_after"]
        csv_writer.writerow(names),
        for i in range(before.num_classes):
            data = [i, inner_score_org[i], inner_score_sty[i], outer_score_org[i], outer_score_sty[i], \
                    distr_org[i], distr_before[i], distr_after[i]]
            csv_writer.writerow(data)
            print(f"class{i} | inner_org: {inner_score_org[i]} | inner_sty: {inner_score_sty[i]}")
            print(f"class{i} | inter_org: {outer_score_org[i]} | inter_sty: {outer_score_sty[i]}")
            print(f"class{i} | before_distr: {distr_before[i]} | after_distr: {distr_after[i]}")

    tsne_vis(features_origin, labels_origin, features_before, labels_before, features_after, labels_after)
    # tsne_vis(features_after, labels_after, "after.png")

