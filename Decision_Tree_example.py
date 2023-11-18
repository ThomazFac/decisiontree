import numpy as np
import matplotlib.pyplot as plt

def H(y):
    H_p1 = 0

    m = len(y)
    
    if m != 0:
        n = np.count_nonzero(y==1)
        p1 = n/m

        if p1 != 0 and p1 != 1:       
            p0 = 1-p1
            H_p1 = -p1*np.log(p1)/np.log(2)-p0*np.log(p0)/np.log(2)

    return H_p1


def node_split(X,node_indices, feature):

    left_indices = []
    right_indices = []

    for i in node_indices:
        if X[i,feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)

    return left_indices,right_indices
    

def InfoGain(X, y, node_indices, feature):

    left_indices, right_indices = node_split(X, node_indices, feature)

    X_total, y_total = X[node_indices],  y[node_indices]
    X_left, y_left   = X[left_indices],  y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    G = 0

    m_total = len(y_total)
    
    m_left = len(y_left)
    m_right = len(y_right)
    
    w_l = m_left/m_total
    
    w_r = 1- w_l
    
    G = H(y_total) - (w_l*H(y_left) + w_r*H(y_right))

    return G


def best_split(X, y, node_indices):
    m = X.shape[1] # n features

    best_feature = 2

    best_gain = 0

    for feature in range(m):
        gain = InfoGain(X, y, node_indices, feature)
        #print(gain)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    return best_feature


def train_tree(X, y, node_indices, branch_name, max_depth, current_depth):
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        
        return

    best_feature = best_split(X,y, node_indices)
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    

    left_indices, right_indices = node_split(X, node_indices, best_feature)

    tree.append((left_indices, right_indices, best_feature))

    train_tree(X, y, left_indices, "Left", max_depth, current_depth+1)
    train_tree(X, y, right_indices, "Right", max_depth, current_depth+1)

    
X_train = np.array([[1,1,1],
                    [1,0,1],
                    [1,0,0],
                    [1,0,0],
                    [1,1,1],
                    [0,1,1],
                    [0,0,0],
                    [1,0,1],
                    [0,1,0],
                    [1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])

root_indices = range(y_train.size)
tree = []
train_tree(X_train, y_train, root_indices, 'root', max_depth=2, current_depth=0)

