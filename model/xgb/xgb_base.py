from loss import MSE
import numpy as np
from typing import Any, List, Dict


class Tree:

    def __init__(self):
        super(Tree, self).__init__()
        self.loss_func = MSE()
        self.alpha: float = 0
        self.beta: float = 0

    def _node_score(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        """
        calculate node score and node weight
        :param x: pred
        :param y: true
        :return: [m,n]:[node score,node weight]
        """
        self.loss_func = self.loss_func(x, y)
        return [self.g.sum() ** 2 / (self.h.sum() + self.alpha * self.g.shape[0]),
                - self.g.sum() / (self.h.sum() + self.alpha * self.g.shape[0])]

    @property
    def g(self):
        """
        :return: loss function's grad
        """
        return self.loss_func.g()

    @property
    def h(self):
        """
        :return: loss function's hess
        """
        return self.loss_func.h()

    def get_gain(self, data: np.ndarray, y: np.ndarray, last_pred: np.ndarray) -> List:
        """
        Traverse the partition points and find the score of all the partition points
        :param data: feature
        :param y: true label
        :param last_pred: last tree's pred
        :return: score list
        """
        score = []
        # todo not itr
        for index in range(1, data.shape[0]):
            left_score, w_left = self._node_score(last_pred[0:index, :], y[0:index, :])
            right_score, w_right = self._node_score(last_pred[index:, :], y[index:, :])
            before_split_score, w_node = self._node_score(last_pred, y)
            all_score = 1 / 2 * (left_score + right_score - before_split_score) - self.beta
            score.append([all_score, w_left, w_right, w_node])
        return score


class Node:
    def __init__(self, x: np.ndarray, y: np.ndarray = None, last_pred: np.ndarray = None, depth: int = None,
                 pred: np.ndarray = None):
        self.x = x
        self.y = y
        self.pred = pred
        self.last_pred = last_pred
        self.depth = depth
        self.left = None
        self.right = None
        self.is_leaf = False
        self.feature_index = None
        self.point = None
        self.max_score = None

    def set_leaf(self, pred=None):
        """
        set this node as leaf node.
        The left and right nodes of this node are None by default
        :param pred: this node's pred
        :return: None
        """
        self.is_leaf = True
        self.left = None
        self.right = None
        self.pred = pred

    def set_no_leaf(self, feature_index: int, point: float, max_score: float, left: "Node", pred: np.ndarray,
                    right: "Node" = None):
        """
        :param feature_index: which feature to split
        :param point: split score
        :param max_score: The maximum gain divided by this feature and this point
        :param left: left node
        :param pred: pred
        :param right: right node
        :return: None
        """
        self.left = left
        self.right = right
        self.feature_index = feature_index
        self.point = point
        self.max_score = max_score
        self.pred = pred

    def predict(self):
        """
        :return: all leaf node's result
        """
        if self.is_leaf:
            return self.pred
        else:
            return self.left.predict() + self.right.predict()


class DTree(Tree):

    def __init__(self, max_depth=5):
        super(DTree, self).__init__()
        self.max_depth: int = max_depth
        self.result = None
        self.pred = []

    def get_dict(self, root: Node) -> Dict:
        """
        :param root:
        :return:dict, include this tree's all information,can use to draw tree
        """
        if root.is_leaf:
            return {"pred": root.pred}
        else:
            dicts = {"feature": root.feature_index, "point": root.point, "pred": root.pred, "x": root.x.shape}
            left = self.get_dict(root.left)
            right = self.get_dict(root.right)
            dicts.update({"left": left, "right": right})
            return dicts

    def draw(self):
        """
        todo: will try to draw
        :return: dicts
        """
        dicts = {}
        dicts.update({"top": self.get_dict(self.result)})
        return dicts

    def split_node(self, node: Node):
        """
        :param node:
        :return:
        """
        x, y, last_pred = node.x, node.y, node.last_pred
        # get all feature sort result
        # 对x的每一列进行升序排序 获取索引
        index = np.argsort(x, axis=0)
        score_feature = []
        # for all feature
        # todo not loop
        # 遍历特征
        # score [feature,split_score]
        for i in range(index.shape[1]):
            # 每个特征最佳划分点获取最高得分
            index_feature = index[:, i:i + 1]
            score = self.get_gain(data=x[index_feature], y=y[index_feature], last_pred=last_pred[index_feature])
            score_feature.append(score)

        score_feature = np.array(score_feature)
        # (x,y) x:best feature index  y:best split index
        # 找到得分最高的特征索引和划分点
        max_score = score_feature[:, :, 0].max()
        (feature_index, split_point) = np.unravel_index(np.argmax(score_feature[:, :, 0], axis=None),
                                                        shape=(score_feature.shape[0], score_feature.shape[1]))

        w_left, w_right, w_node = score_feature[feature_index, split_point, 1], score_feature[
            feature_index, split_point, 2], score_feature[feature_index, split_point, 3]

        # 获取按照第feature_index排列的data索引
        data_index = index[:, feature_index]
        point_score = x[data_index][split_point, feature_index]

        # 去除这个索引
        col_range = list(range(x.shape[1]))
        col_range.remove(feature_index)
        # 获取到其它feature值
        x, y, last_pred = x[data_index][:, col_range], y[data_index], last_pred[data_index]

        # 按照节点划分左树和右树
        x_left, x_right, y_left, y_right = x[0:split_point], x[split_point:], y[0:split_point], y[split_point:]

        # 预测结果也划分
        last_pred_left, last_pred_right = last_pred[0:split_point], last_pred[split_point:]
        # split score
        return x_left, x_right, y_left, y_right, last_pred_left, last_pred_right, feature_index, point_score, max_score, w_left, w_right, w_node

    def fit(self, x, y, last_pred):
        if x.shape[1] < self.max_depth:
            self.max_depth = x.shape[1]
        root = Node(x, y, last_pred, self.max_depth)
        self.result = self.create_node(root)

    def predict(self, x):
        pred = self.create_predict_node(x, self.result, np.arange(0, x.shape[0]))
        result = np.array(pred)
        result = result[np.argsort(np.array(pred)[:, 0])][:, 1:2]
        return result

    def create_predict_node(self, x, or_node, x_index=None):
        if not or_node.is_leaf:
            # 非叶节点
            feature_index, split_score = or_node.feature_index, or_node.point
            # get split bool matrix
            left, right = x[:, feature_index] < split_score, x[:, feature_index] >= split_score
            left_index, right_index = np.argwhere(left), np.argwhere(right)

            # 找出left和right在原数据集上的索引
            or_left_index = x_index[left_index.flatten().tolist()]
            or_right_index = x_index[right_index.flatten().tolist()]

            new_feature_index = list(range(x.shape[1]))
            new_feature_index.remove(feature_index)

            # 得到删除当前特征后的索引
            x_left, x_right = x[left][:, new_feature_index], \
                              x[right][:, new_feature_index]

            # 对预测数据建立left 和right节点
            return self.create_predict_node(x_left, or_node.left, or_left_index) + self.create_predict_node(x_right,
                                                                                                            or_node.right,
                                                                                                            or_right_index)
        else:
            if len(x_index) > 0:
                return np.frompyfunc(lambda x, y: [x, y], 2, 1)(x_index, or_node.pred).tolist()
            else:
                return []

    def create_node(self, root: Node):
        """
        Recursively build the tree
        :param root:
        :return:
        """
        if root.x.shape[1] == 0:
            root.set_leaf(pred=root.pred)
            return root

        data_left, data_right, y_left, y_right, last_pred_left, last_pred_right, \
        feature_index, point_score, max_score, w_left, w_right, w_node = self.split_node(root)

        left = Node(data_left, y_left, last_pred_left, root.depth - 1, pred=w_left)
        if data_left.shape[0] == 1 or data_left.shape[0] == 0:
            left.set_leaf(pred=w_left)
            root.set_no_leaf(feature_index=feature_index, point=point_score, max_score=max_score, left=left,
                             pred=w_left)
        else:
            left = self.create_node(left)
            root.set_no_leaf(feature_index=feature_index, point=point_score, max_score=max_score, left=left,
                             pred=w_left)

        right = Node(data_right, y_right, last_pred_right, root.depth - 1, pred=w_right)
        if data_right.shape[0] == 1 or data_right.shape[0] == 0:
            right.set_leaf(pred=w_right)
            root.set_no_leaf(feature_index=feature_index, point=point_score, max_score=max_score, right=right,
                             left=root.left, pred=w_right)
        else:
            right = self.create_node(right)
            root.set_no_leaf(feature_index=feature_index, point=point_score, max_score=max_score, right=right,
                             left=root.left, pred=w_right)

        return root


class XGBTree:

    def __init__(self, tree_num=5, max_depth=5, lr=0.5, base_score=0.5):
        self.max_depth = max_depth
        self.tree_num = tree_num
        self.lr = lr
        self.base_score = base_score
        self.tree_nodes = []

    def fit(self, x, y):
        last_pred = np.full(shape=y.shape, fill_value=self.base_score)
        for i in range(self.tree_num):
            tree_i = DTree(self.max_depth)
            tree_i.fit(x=x, y=y, last_pred=last_pred)
            pred_i = tree_i.predict(x)
            last_pred = last_pred + pred_i * self.lr
            self.tree_nodes.append(tree_i)

    def predict(self, x):
        pred = self.base_score
        for tree in self.tree_nodes:
            pred_tree = tree.predict(x)
            pred = pred + pred_tree * self.lr
        return pred
