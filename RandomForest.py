import numpy as np


class Node(object):
    def __init__(self, feature, threshold, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class RandomForest(object):
    ''' Construct a random forest for binary classification problem '''

    def __init__(self, ntrees, nsplits=None):
        ''' Initialize.
        :ntrees number of decision trees
        :nsplits number of features used to split tree nodes. Tree will have atmost nsplits height
        '''
        self.trees = [None] * ntrees
        self.nSplits = nsplits
        self.oobSamples = [None] * ntrees

    def _giniNode(self, outputs):
        pos = outputs.sum()
        prob_pos = pos / float(outputs.shape[0])
        prob_neg = 1 - prob_pos
        return prob_pos * (1 - prob_pos) + prob_neg * (1 - prob_neg)

    def _getGiniImpRed(self, inputfeature, outputs, threshold):
        ''' GINI impurity reduction by splitting on feature at threshold '''
        gini = self._giniNode(outputs)
        left = (inputfeature < threshold)
        right = (inputfeature >= threshold)
        gini_left = self._giniNode(outputs[left])
        gini_right = self._giniNode(outputs[right])
        nobs = outputs.shape[0]
        return gini - left.sum() / float(nobs) * gini_left - right.sum() / float(nobs) * gini_right

    def _constructTree(self, inputs, outputs, features, splits):
        ''' Construct a decision tree
        :inputs 2 dimensional numpy ndarray of shape (num observations, num features)
        :outputs 1 dimensional ndarray with output. Shape (num_observations)
        :features list of features to split the node
        :splits threshold on number of splits
        '''
        if splits <= 0:
            return

        sel_feat = None
        reduction = None
        sel_threshold = None
        for feat in features:
            threshold = np.random.choice(inputs[:, feat], size=1)
            gini_red = self._getGiniImpRed(inputs[:, feat], outputs, threshold)
            if (reduction is None) or (reduction < gini_red):
                reduction = gini_red
                sel_feat = feat
                sel_threshold = threshold

        node = Node(sel_feat, sel_threshold)
        left_data = (inputs[:, sel_feat] <= sel_threshold)
        features_rem = [f for f in features if f != sel_feat]
        node.left = self._constructTree(inputs[left_data, :], outputs[left_data], features_rem, splits - 1)
        right_data = np.logical_not(left_data)
        node.right = self._constructTree(inputs[right_data, :], outputs[right_data], features_rem, splits - 1)
        return node

    def construct(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        ''' Construct a random forest for binary classification problem
        :param inputs: 2 dimensional numpy ndarray of shape (num observations, num features)
        :param outputs: 1 dimensional ndarray of shape (num_observations).
        Contains booleans: True or False
        '''

        nfeat = inputs.shape[0]
        if self.nSplits is None:
            self.nSplits = int(np.sqrt(nfeat))
        y_labels = sorted(list(set(outputs)))
        assert len(y_labels) == 2
        y_out = np.where(outputs == y_labels[0], True, False)
        features = np.arange(inputs.shape[1])
        for i in range(len(self.trees)):
            sample_inputs = np.random.choice(inputs.shape[0], inputs.shape[0], replace=True)
            self.trees[i] = self._constructTree(inputs[sample_inputs, :], y_out[sample_inputs], features)
            self.oobSamples[i] = sample_inputs
