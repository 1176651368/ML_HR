from xgb.xgb_base import XGBTree, DTree
import numpy as np
from linear_reg.tools import ConnectLocal
from np_paillier.paillier import generate_paillier_keypair


class DTreeHost():

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.connect = ConnectLocal(role='host')
        self.en_g = None
        self.en_h = None

    def get_en_g_h(self):
        [self.en_g, self.en_h] = self.connect.get(role='guest')

    def send_split_result(self, data):
        self.connect.push(data, role='guest')

    def get_sum(self,x,y):
        left = y[0:x].sum(axis=0)
        # right = y[x:].sum(axis=0).reshape((-1,1))
        return left.tolist()

    def split_feature_one(self, g_h):
        index = np.arange(1,g_h.shape[0])
        score = np.frompyfunc(lambda x:self.get_sum(x,g_h),1,1)(index).tolist()
        return score

    def get_gain(self, index):
        en_g = self.en_g.toArray()[index]
        en_h = self.en_h.toArray()[index]
        temp = np.concatenate([en_g,en_h], axis=2)
        score = []
        for i in range(temp.shape[1]):
            r = self.split_feature_one(temp[:,i,:])
            score.append(r)

        # 每个特征每个划分点的左 g h
        # [feature,split_point,g or h]
        score = np.array(score)
        return score

    def push_gain_guest(self,score):
        self.connect.push(score,'guest')

    def fit(self, x, y=None, last_pred=None):
        self.get_en_g_h()
        # todo:could be better
        index = np.argsort(x, axis=0)
        self.connect.push(index,'guest')
        self.push_gain_guest(self.get_gain(index))


x = np.random.random_sample((5,3))
tree = DTreeHost()
tree.fit(x)

# class XGBGuest(XGBTree):
#     def __init__(self, tree_num=5, max_depth=5, lr=0.5, base_score=0.5, role='guest'):
#         super(XGBGuest, self).__init__(tree_num=tree_num, max_depth=max_depth, lr=lr, base_score=base_score)
#         self.connect = ConnectLocal(role=role)
#         self.public_key, self.private_key = generate_paillier_keypair()
