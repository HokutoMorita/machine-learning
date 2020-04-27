import numpy as np


class Node(object):
    '''
    ノードクラス
    '''
    def __init__(self, criterion="gini", max_depth=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state
        self.depth = None
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.label = None
        self.impurity = None
        self.info_gain = None
        self.num_samples = None
        self.num_classes = None

    def split_node(self, sample, target, depth, ini_num_classes):
        self.depth = depth
        self.num_samples = len(target)
        self.num_classes = [len(target[target==i]) for i in ini_num_classes]
        if len(np.unique(target)) == 1:
            self.label = target[0]
            self.impurity = self.criterion_func(target)
            return
        class_count = {i: len(target[target==i]) for i in np.unique(target)}
        self.label = max(class_count.items(), key=lambda x:x[1])[0]
        self.impurity = self.criterion_func(target)
        num_features = sample.shape[1]
        self.info_gain = 0.0
        if self.random_state != None:
            np.random.seed(self.random_state)
        f_loop_order = np.random.permutation(num_features).tolist()
        for f in f_loop_order:
            uniq_feature = np.unique(sample[:, f])
            split_points = (uniq_feature[:-1] + uniq_feature[1:]) / 2.0
            for threshold in split_points:
                target_l = target[sample[:, f] <= threshold]
                target_r= target[sample[:, f] > threshold]
                val = self.calc_info_gain(target, target_l, target_r)
                if self.info_gain < val:
                    self.info_gain = val
                    self.feature = f
                    self.threshold = threshold
        if self.info_gain == 0.0:
            return
        if self.depth == self.max_depth:
            return
        sample_l = sample[sample[:, self.feature] <= self.threshold]
        target_l = target[sample[:, self.feature] <= self.threshold]
        self.left = Node(self.criterion, self.max_depth)
        self.left.split_node(sample_l, target_l, depth + 1, ini_num_classes)
        sample_r = sample[sample[:, self.feature] > self.threshold]
        target_r = target[sample[:, self.feature] > self.threshold]
        self.right = Node(self.criterion, self.max_depth)
        self.right.split_node(sample_r, target_r,  depth + 1, ini_num_classes)

    def criterion_func(self, target):
        classes = np.unique(target)
        numdata = len(target)
        if self.criterion == "gini":
            val = 1
            for c in classes:
                p = float(len(target[target == c])) / numdata
                val -= p ** 2.0
        elif self.criterion == "entropy":
            val = 0
            for c in classes:
                p = float(len(target[target == c])) / numdata
                if p != 0.0:
                    val -= p * np.log2(p)
        return val

    def calc_info_gain(self, target_p, target_cl, target_cr):
        cri_p = self.criterion_func(target_p)
        cri_cl = self.criterion_func(target_cl)
        cri_cr = self.criterion_func(target_cr)
        return cri_p - len(target_cl) / float(len(target_p)) * cri_cl - len(target_cr) / float(len(target_p)) * cri_cr

    def predict(self, sample):
        if self.feature == None or self.depth == self.max_depth:
            return self.label
        else:
            if sample[self.feature] <= self.threshold:
                return self.left.predict(sample)
            else:
                return self.right.predict(sample)

class TreeAnalysis(object):
    def __init__(self):
        self.num_features = None
        self.importances = None

    def compute_feature_importances(self, node):
        if node.feature == None:
            return
        self.importances[node.feature] += node.info_gain * node.num_samples
        self.compute_feature_importances(node.left)
        self.compute_feature_importances(node.right)

    def get_feature_importances(self, node, num_features, normalize=True):
        self.num_features = num_features
        self.importances = np.zeros(num_features)
        self.compute_feature_importances(node)
        self.importances /= node.num_samples
        if normalize:
            normalizer = np.sum(self.importances)
            if normalizer > 0.0:
                self.importances /= normalizer
        return self.importances

class DecisionTreeMH(object):
    def __init__(self, criterion="gini", max_depth=None, random_state=None):
        self.tree = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree_analysis = TreeAnalysis()

    def fit(self, sample, target):
        self.tree = Node(self.criterion, self.max_depth, self.random_state)
        self.tree.split_node(sample, target, 0, np.unique(target))
        self.feature_importances_ = self.tree_analysis.get_feature_importances(self.tree, sample.shape[1])

    def predict(self, sample):
        pred = []
        for s in sample:
            pred.append(self.tree.predict(s))
        return np.array(pred)

    def score(self, sample, target):
        return sum(self.predict(sample) == target) / float(len(target))
