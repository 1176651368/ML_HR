import numpy as np


class MSE:
    def __init__(self):
        self.x = None
        self.y = None

    def __call__(self, x, y):
        self.x = x
        self.y = y
        return self

    def forward(self):
        return 1 / 2 * (self.y - self.x) ** 2

    def g(self):
        return self.x - self.y

    def h(self):
        return np.ones(self.x.shape)


# [-0.6331047 - 0.39753547 - 0.29014707 - 0.30772012]

s = {'top': {'feature_index': 3, 'split_point_score': 1.186831, 'max_score': 35.648153933711114, 'depth': 4,
             'pred': None,
             'left4': {'feature_index': 2, 'split_point_score': -1.616584, 'max_score': 2.8236933407578597, 'depth': 3,
                       'pred': None, 'left3': 1.5371086999999999,
                       'right3': {'feature_index': 1, 'split_point_score': -0.73472506, 'max_score': 3.631664996627533,
                                  'depth': 2, 'pred': None, 'left2': -0.04844601388225806,
                                  'right2': -0.2939109742857143}},
             'right4': {'feature_index': 1, 'split_point_score': -0.22165419, 'max_score': 11.429821139493882,
                        'depth': 3, 'pred': None, 'left3': {'feature_index': 0, 'split_point_score': 0.0048223916,
                                                            'max_score': 4.9942699529414085, 'depth': 2, 'pred': None,
                                                            'left2': -0.057078903333333354,
                                                            'right2': 0.9188207704166667},
                        'right3': {'feature_index': 0, 'split_point_score': 0.8598636999999999,
                                   'max_score': 13.481769837280213, 'depth': 2, 'pred': None,
                                   'left2': 0.06478614138870968, 'right2': -1.2224268999999999}}}}
