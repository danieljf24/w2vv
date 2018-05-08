import numpy as np
from scipy.spatial import distance

def l2_norm(a):
    return a / np.linalg.norm(a, ord=2, axis=1).reshape(-1,1)
    
def order_violations(s, im):
    """ Computes the order violations (Equation 2 in the paper) """
    return np.power(np.linalg.norm(np.maximum(0, s - im)),2)


class CosineLosser_batch():
    # calculate cosine similarity between matrix and matrix
    def calculate(self, matrix_a, matrix_b):
        A = np.array(matrix_a)
        B = np.array(matrix_b)
        result = distance.cdist(A,B,'cosine') - 1
        return result.tolist()

class DotLosser_batch():
    # calculate cosine similarity between matrix and matrix
    def calculate(self, matrix_a, matrix_b):
        result = - np.dot(matrix_a, np.transpose(matrix_b))
        return result.tolist()


class OrderLosser_batch():
    # calculate cosine similarity between matrix and matrix
    def calculate(self, s_emb, im_emb):
        """ Given sentence and image embeddings, compute the error matrix """
        erros = [order_violations(x, y) for x in np.abs(l2_norm(s_emb)) for y in np.abs(l2_norm(im_emb))]
        return np.asarray(erros).reshape((len(s_emb), len(im_emb)))


NAME_TO_ENCODER = {'cosine': CosineLosser_batch, 'order': OrderLosser_batch, 'dot': DotLosser_batch}

def get_losser(name):
    return NAME_TO_ENCODER[name]


if __name__ == "__main__":
    losser = get_losser('order')()
    a = [[0.2,2,0.1],[1,0,0.8]]
    b = [[1,2,3], [2,3,4]]
    print losser.calculate(a, b)
    print losser.calculate(b, a)