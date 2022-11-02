
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np


def draw(x_feat_1,x_proto_1,x_support_1,x_feat_2,x_proto_2,x_support_2,x_y_1,n):
    #maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
    maker = ['o', '^', '*']
    colors = ['r','y','g','b','m']
    colors_bg = ['mistyrose','lemonchiffon','lightgreen','lightblue','thistle']
    #colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
    #          'hotpink']
    Label_Com = ['a', 'b', 'c', 'd']
    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 32,
             }
    feat_1 = x_feat_1.data.cpu().numpy()
    feat_1_proto = x_proto_1.data.cpu().numpy()
    feat_2 = x_feat_2.data.cpu().numpy()
    feat_2_proto = x_proto_2.data.cpu().numpy()
    feat_all_1 = np.concatenate((feat_1_proto,feat_1),axis=0)
    feat_all_2 = np.concatenate((feat_all_1,feat_2_proto),axis=0)
    feat_all = np.concatenate((feat_all_2,feat_2),axis=0)
    visual_all = visual(feat_all)
    visual_feat_proto_1 = visual_all[:5]
    visual_feat_1 = visual_all[5:80]
    visual_feat_proto_2 = visual_all[80:85]
    visual_feat_2 = visual_all[85:]

    print(visual_feat_1.shape,visual_feat_proto_1.shape,visual_feat_2.shape,visual_feat_proto_2.shape)
    #feat_2 = x_feat_2.data.cpu().numpy()
    label_test = x_y_1
    label_proto = [[0],[1],[2],[3],[4]]
    label_proto = np.array(label_proto)
    background_x = np.arange(visual_all[:,0].min(),visual_all[:,0].max(),0.001)
    background_y = np.arange(visual_all[:,1].min(),visual_all[:,1].max(),0.001)
    background   = np.meshgrid(background_x,background_y)
    background_1 = torch.tensor(background[0])
    background_2 = torch.tensor(background[1])
    background_v = torch.stack((background_1,background_2),2)
    background_lean = background_v.reshape(-1,2)
    #print(background_v.shape)
    #raw_input()
    
    #background   = np.c_[background_x.reshape(1,background_x.size)[0],background_y.reshape(1,background_x.size)[0]]
    visual_feat_proto_1_p = torch.tensor(visual_feat_proto_1)
    visual_feat_proto_2_p = torch.tensor(visual_feat_proto_2)
    scores_1 = euclidean_dist(background_lean, visual_feat_proto_1_p)
    scores_2 = euclidean_dist(background_lean, visual_feat_proto_2_p)
    _,indices_1 = torch.topk(scores_1,1,largest=True)
    _,indices_2 = torch.topk(scores_2,1,largest=True)
    indices_1 = indices_1.numpy()
    indices_2 = indices_2.numpy()

    #feat = torch.rand(128, 1024)  
    #label_test1 = [0 for index in range(40)]
    #label_test2 = [1 for index in range(40)]
    #label_test3 = [2 for index in range(48)]

    #label_test = np.array(label_test1 + label_test2 + label_test3)
    #print(label_test)
    #print(label_test.shape,feat.shape)

    fig = plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plotlabels(visual_feat_1,visual_feat_proto_1,visual_feat_proto_1, label_test,label_proto,indices_1,background_lean,0, '(a)',maker,colors,Label_Com,font1,colors_bg)
    plt.subplot(122)
    plotlabels(visual_feat_2,visual_feat_proto_2,visual_feat_proto_1, label_test,label_proto,indices_2,background_lean,1, '(b)',maker,colors,Label_Com,font1,colors_bg)
    path_p = '/home/chenxu/mit2023/cx_work/DEEPBDC-new/draw/' + str(n) +'.png'
    plt.savefig(path_p)
    #if n == 0:
    #    plt.savefig(path_p)
    #elif n == 1:
    #    plt.savefig('3_1.png')

    #fig = plt.figure(figsize=(10, 10))

    #plotlabels(visual(feat_2), label_test, '(a)',maker,colors,Label_Com,font1)
    #plt.savefig('1_1.png')
    #plt.show(fig)

def visual(feat):
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)

    x_ts = ts.fit_transform(feat)

    print(x_ts.shape)  # [num, 2]

    x_min, x_max = x_ts.min(0), x_ts.max(0)

    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final

def plotlabels(S_lowDWeights,S_proto,support, Trure_labels,Proto_label,score,background,c, name,maker,colors,Label_Com,font1,colors_bg):
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))
    Proto_label = Proto_label.reshape((-1,1))
    score_label = score.reshape((-1,1))
    #print(S_proto.shape,Proto_label.shape)
    S_data_proto = np.hstack((S_proto, Proto_label))
    background_data = np.hstack((background, score_label))
    support_data = np.hstack((support, Proto_label))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    S_data_proto = pd.DataFrame({'x': S_data_proto[:, 0], 'y': S_data_proto[:, 1], 'label': S_data_proto[:, 2]})
    support_data = pd.DataFrame({'x': support_data[:, 0], 'y': support_data[:, 1], 'label': support_data[:, 2]})
    bg_data = pd.DataFrame({'x': background_data[:, 0], 'y': background_data[:, 1], 'label': background_data[:, 2]})
    #print(S_data)
    #print(S_data.shape)  # [num, 3]

    for index in range(5):
        X_bg = bg_data.loc[bg_data['label'] == index]['x']
        Y_bg = bg_data.loc[bg_data['label'] == index]['y']
        plt.scatter(X_bg, Y_bg, cmap='brg', s=10, marker=maker[1], c=colors_bg[index], edgecolors=colors_bg[index], alpha=0.65)
    for index in range(5):
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=100, marker=maker[0], c=colors[index], edgecolors=colors[index], alpha=0.65)
        X_support = support_data.loc[support_data['label'] == index]['x']
        Y_support = support_data.loc[support_data['label'] == index]['y']
        plt.scatter(X_support, Y_support, cmap='brg', s=300, marker=maker[1], c=colors[index], edgecolors=colors[index], alpha=0.65)
        if c == 1:
            X_proto = S_data_proto.loc[S_data_proto['label'] == index]['x']
            Y_proto = S_data_proto.loc[S_data_proto['label'] == index]['y']
            plt.scatter(X_proto, Y_proto, cmap='brg', s=400, marker=maker[2], c=colors[index], edgecolors=colors[index], alpha=0.65)
        plt.xticks([])
        plt.yticks([])

    plt.title(name, fontsize=32, fontweight='normal', pad=20)


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    #x = torch.tensor(x)
    #y = torch.tensor(y)
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    score = -torch.pow(x - y, 2).sum(2)
    return score