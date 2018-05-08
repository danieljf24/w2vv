"""
Evaluation code for multimodal ranking
Throughout, we assume 5 captions per image, and that
captions[5i:5i+5] are GT descriptions of images[i]
"""
import numpy as np
from basic.metric import getScorer

def t2i(c2i, vis_details=False, n_caption=5):
    """
    Text->Images (Image Search)
    c2i: (5N, N) matrix of caption to image errors
    vis_details: if true, return a dictionary for ROC visualization purposes
    """

    ranks = np.zeros(c2i.shape[0])


    vis_dict = {'sentences': []}

    for i in range(len(ranks)):
        d_i = c2i[i]
        inds = np.argsort(d_i)

        rank = np.where(inds == i/n_caption)[0][0]
        ranks[i] = rank

        def image_dict(k):
            return {'id': k, 'score': float(d_i[k])}

        if vis_details:  # save top 10 images as well as GT image and their scores
            vis_dict['sentences'].append({
                'id': i,
                'rank': rank + 1,
                'gt_image': image_dict(i/n_caption),
                'top_images': map(image_dict, inds[:10])
            })

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    stats = map(float, [r1, r5, r10, medr, meanr])

    if not vis_details:
        return stats
    else:
        vis_dict['stats'] = {'R@1': r1, 'R@5': r5, 'R@10': r10, 'median_rank': medr, 'mean_rank': meanr}
        return stats, vis_dict


def i2t(c2i, n_caption=5):
    """
    Images->Text (Text Search)
    c2i: (5N, N) matrix of caption to image errors
    """

    ranks = np.zeros(c2i.shape[1])

    for i in range(len(ranks)):
        d_i = c2i[:, i]
        inds = np.argsort(d_i)

        rank = np.where(inds/n_caption == i)[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return map(float, [r1, r5, r10, medr, meanr])



def i2t_map(c2i, n_caption=5):
    """
    Images->Text (Text Search)
    c2i: (5N, N) matrix of caption to image errors
    """
    scorer = getScorer('AP')
    perf_list = []
    for i in range(c2i.shape[1]):
        d_i = c2i[:, i]
        labels = [0]*len(d_i)
        labels[i*n_caption:(i+1)*n_caption] = [1]*n_caption

        sorted_labels = [labels[x] for x in np.argsort(d_i)]
        current_score = scorer.score(sorted_labels)
        perf_list.append(current_score)

    return np.mean(perf_list)



def i2t_inv_rank(c2i, n_caption=2):
    """
    Images->Text (Text Search)
    c2i: (5N, N) matrix of caption to image errors
    n_caption: number of captions of each image/video
    """
    assert c2i.shape[0] / c2i.shape[1] == n_caption
    inv_ranks = np.zeros(c2i.shape[1])

    for i in range(len(inv_ranks)):
        d_i = c2i[:, i]
        inds = np.argsort(d_i)

        rank = np.where(inds/n_caption == i)[0]
        inv_ranks[i] = sum(1.0 / (rank +1 ))

    return np.mean(inv_ranks)


def i2t_inv_rank_multi(c2i, n_caption=2):
    """
    Images->Text (Text Search)
    c2i: (5N, N) matrix of caption to image errors
    n_caption: number of captions of each image/video
    """
    result = []
    for i in range(n_caption):
        idx = range(i, c2i.shape[0], n_caption)
        sub_c2i = c2i[idx, :]
        score = i2t_inv_rank(sub_c2i, n_caption=1)
        result.append(score)
    return result


def i2t_mc_accuracy(all_errors, label_list):
    assert len(all_errors[0]) == 5
    pred_label_list = np.argmin(all_errors, axis=-1)
    assert len(pred_label_list) == len(label_list)
    accuracy = 1.0 * np.sum(np.array(pred_label_list) == np.array(label_list)) / len(label_list)
    return pred_label_list, accuracy
