import pandas as pd
import numpy as np

def compute_spatial_entropy(x, y, size, grid_size=100, lim=8):
    """
    计算基于感受野覆盖情况的空间信息熵（视野为 [-lim, lim] × [-lim, lim] 的方形）
    :param x: 感受野中心的 x 坐标
    :param y: 感受野中心的 y 坐标
    :param size: 感受野直径（圆的大小）
    :param grid_size: 网格划分大小（默认 100×100）
    :param lim: 视野范围边界
    :return: 计算出的信息熵, 2D 覆盖矩阵
    """
    # 生成均匀网格
    X, Y = np.meshgrid(np.linspace(-lim, lim, grid_size),
    np.linspace(-lim, lim, grid_size))
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T # (10000, 2)

    # 初始化覆盖计数
    coverage = np.zeros(grid_size**2)

    # 计算每个网格点的覆盖次数
    for xi, yi, si in zip(x, y, size):
        r_i = si / 2 # 直径转换为半径
        distances = np.sqrt((grid_points[:, 0] - xi)**2 + (grid_points[:, 1] - yi)**2)
        coverage += (distances <= r_i).astype(float) # 被覆盖的点+1

    # 归一化为概率分布
    coverage /= coverage.sum()

    # 计算熵（忽略零概率点）
    P_nonzero = coverage[coverage > 0] # 避免 log(0)
    entropy = -np.sum(P_nonzero * np.log2(P_nonzero))

    # 空间覆盖率
    coverage_mask = coverage > 0
    coverage_ratio = np.sum(coverage_mask) / coverage.size

    return entropy, coverage_ratio, coverage.reshape(grid_size, grid_size)

def compute_gradient_metrics(coverage_matrix):
    dx, dy = np.gradient(coverage_matrix)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)

    # 仅使用非零梯度点计算方差
    nonzero_gradients = gradient_magnitude[gradient_magnitude > 0]
    if len(nonzero_gradients) > 0:
        gv = np.mean(nonzero_gradients**2)
    else:
        gv = 0

    return gv, gradient_magnitude

hue_order = ['P7','C5','PO7','P3','Cb1','P1','C1','O1','Pz','POz','Oz','O2','PO4','P2','Cb2','P4','PO8','P6','P8']
df = pd.read_csv('results/RFsize_reliable256.csv')
df19 = df[(df['channel'].isin(hue_order))]

subject = np.array(df['subject'].drop_duplicates())
paradigm = ['WN20','WN15']

frame = []
for sub in subject:
    for p in paradigm:
        x_19 = np.array(df19[(df19['paradigm']==p)&(df19['subject']==sub)]['x'])
        y_19 = np.array(df19[(df19['paradigm']==p)&(df19['subject']==sub)]['y'])
        size_19 = np.array(df19[(df19['paradigm']==p)&(df19['subject']==sub)]['size']) # 直径

        # 示例数据（66 导联）
        x_66 = np.array(df[(df['paradigm']==p)&(df['subject']==sub)]['x'])
        y_66 = np.array(df[(df['paradigm']==p)&(df['subject']==sub)]['y'])
        size_66 = np.array(df[(df['paradigm']==p)&(df['subject']==sub)]['size']) # 直径

        entropy_19, coverage_19, P19 = compute_spatial_entropy(x_19, y_19, size_19)
        entropy_66, coverage_66, P66 = compute_spatial_entropy(x_66, y_66, size_66)

        gv_19, grad_19 = compute_gradient_metrics(P19)
        gv_66, grad_66 = compute_gradient_metrics(P66)
        
        f = pd.DataFrame({
            'subject':[sub],
            'paradigm':[p],
            'entropy':[entropy_19],
            'channel':['Chn19'],
            'delta_coverage':[(coverage_66-coverage_19)/coverage_19*100],
            'delta_gv':[(gv_66-gv_19)/gv_19*100]
        })
        frame.append(f)
        f = pd.DataFrame({
            'subject':[sub],
            'paradigm':[p],
            'entropy':[entropy_66],
            'channel':['Chn66'],
            'delta_coverage':[(coverage_66-coverage_19)/coverage_19*100],
            'delta_gv':[(gv_66-gv_19)/gv_19*100]
        })
        frame.append(f)

df = pd.concat(frame, axis=0, ignore_index=True)
df.to_csv('results/highdensity.csv')


