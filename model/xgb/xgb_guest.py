from xgb.xgb_base import XGBTree, DTree, Node
import numpy as np
from linear_reg.tools import ConnectLocal
from np_paillier.paillier import generate_paillier_keypair


class DTreeGuest(DTree):

    def __init__(self, max_depth: int = 5):
        super(DTreeGuest, self).__init__(max_depth=max_depth)
        self.public_key, self.private_key = generate_paillier_keypair()
        self.connect = ConnectLocal(role='guest')

    def get_en_g_h(self, pred, y):
        self.loss_func = self.loss_func(pred, y)
        en_g = self.public_key.encrypt(self.loss_func.g())
        en_h = self.public_key.encrypt(self.loss_func.h())
        return en_g, en_h

    def send_en_g_h(self, y, last_pred):
        en_g, en_h = self.get_en_g_h(last_pred, y)
        self.connect.push([en_g, en_h], role='host')

    def get_sum(self, x, y):
        right = y[x:].sum(axis=0)
        # right = y[x:].sum(axis=0).reshape((-1,1))
        return right.tolist()

    def get_sum_(self, x, y):
        left = y[0:x].sum(axis=0)
        right = y[x:].sum(axis=0)
        return left.tolist() + right.tolist()

    def split_feature_one(self, g_h, is_left=False):
        index = np.arange(1, g_h.shape[0])
        if is_left:
            score = np.frompyfunc(lambda x: self.get_sum_(x, g_h), 1, 1)(index).tolist()
        else:
            score = np.frompyfunc(lambda x: self.get_sum(x, g_h), 1, 1)(index).tolist()

        return score

    def calculate_host_score(self, index):
        en_g = self.loss_func.g()[index]
        en_h = self.loss_func.h()[index]
        temp = np.concatenate([en_g, en_h], axis=2)
        g_right = []
        for i in range(temp.shape[1]):
            r = self.split_feature_one(temp[:, i, :])
            g_right.append(r)
        g_right = np.array(g_right)
        g_left = self.private_key.decrypt(self.connect.get(role='host'))
        score_left = g_left[:, :, 0] ** 2 / (g_right[:, :, 1] + self.alpha)
        score_right = g_right[:, :, 0] ** 2 / (g_right[:, :, 1] + self.alpha)
        score_before = temp.sum(axis=0)
        score_before = np.pad((score_before[:, 0] ** 2 / (score_before[:, 1] + self.alpha)).reshape((-1, 1)),
                              pad_width=((0, 0), (0, score_left.shape[1] - 1)), mode='edge')
        score = 1 / 2 * (score_left + score_right - score_before) - self.beta
        w_left, w_right = g_left[:, :, 0] / (g_right[:, :, 1] + self.alpha), g_right[:, :, 0] / (
                    g_right[:, :, 1] + self.alpha)
        w_all = np.pad((score_before[:, 0] / (score_before[:, 1] + self.alpha)).reshape((-1, 1)),
                       pad_width=((0, 0), (0, score_left.shape[1] - 1)), mode='edge')
        return score, w_left, w_right, w_all

    def calculate_guest_score(self, x):
        index = np.argsort(x, axis=0)
        en_g = self.loss_func.g()[index]
        en_h = self.loss_func.h()[index]
        temp = np.concatenate([en_g, en_h], axis=2)
        g_h = []
        for i in range(temp.shape[1]):
            r = self.split_feature_one(temp[:, i, :], is_left=True)
            g_h.append(r)
        g_h = np.array(g_h)
        g_h = g_h.reshape((g_h.shape[0], g_h.shape[1], 2, 2))
        g_left = g_h[:, :, :, 0]
        g_right = g_h[:, :, :, 1]

        score_left = g_left[:, :, 0] ** 2 / (g_right[:, :, 1] + self.alpha)
        score_right = g_right[:, :, 0] ** 2 / (g_right[:, :, 1] + self.alpha)
        score_before = temp.sum(axis=0)
        score_before = np.pad((score_before[:, 0] / (score_before[:, 1] + self.alpha)).reshape((-1, 1)),
                              pad_width=((0, 0), (0, score_left.shape[1] - 1)), mode='edge')
        score = 1 / 2 * (score_left + score_right - score_before) - self.beta
        w_left, w_right = g_left[:, :, 0] / (g_right[:, :, 1] + self.alpha), g_right[:, :, 0] / (
                    g_right[:, :, 1] + self.alpha)
        w_all = np.pad((score_before[:, 0] / (score_before[:, 1] + self.alpha)).reshape((-1, 1)),
                       pad_width=((0, 0), (0, score_left.shape[1] - 1)), mode='edge')
        return score, w_left, w_right, w_all

    def calculate_best_feature(self, host_score, guest_score):
        max_host = np.max(host_score)
        max_guest = np.max(guest_score)
        if max_guest >= max_host:
            max_score = max_guest
            max_role = 'guest'
            index = np.unravel_index(np.argmax(max_guest), shape=guest_score.shape)
        else:
            max_score = max_host
            max_role = 'host'
            index = np.unravel_index(np.argmax(max_guest), shape=host_score.shape)

        return max_role, max_score, index

    def split_data(self, x, y, last_pred, max_role, index, sort_index):
        # sort_index:["feature", "sample"]
        # x_left, x_right, y_left, y_right, last_pred_left, last_pred_right, feature_index, point_score, max_score, w_left, w_right, w_node
        x, y, last_pred = x[sort_index], y[sort_index], last_pred[sort_index]
        x_left, x_right = x[0:sort_index[1]], x[sort_index[1]:]
        y_left, y_right = y[0:sort_index[1]], y[sort_index[1]:]
        last_pred_left, last_pred_right = last_pred[0:sort_index[1]], last_pred[sort_index[1]:]
        # w_left, w_right =

    def fit(self, x, y, last_pred):
        self.send_en_g_h(y, last_pred)
        sort_index = self.connect.get(role='host')
        host_score, w_left, w_right, w_all = self.calculate_host_score(sort_index)
        print(w_left)
        guest_score, w_left, w_right, w_all = self.calculate_guest_score(x)
        print(w_left)
        # print(host_score.shape, guest_score.shape)
        # max_role, max_score, max_index = self.calculate_best_feature(host_score, guest_score)

        # score_all = 1/2()


data = np.random.random_sample((5, 4))
y = np.random.random_sample((5, 1))
last_pred = np.zeros(shape=(5, 1))
tree = DTreeGuest()
tree.fit(data, y, last_pred)
