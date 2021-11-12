import numpy as np

###计算竖直距离
def vertical_distance(left, current, right):
    EPSILON = 1e-06
    INFINITY = float("inf")
    a_x, a_y = left
    b_x, b_y = current
    c_x, c_y = right
    result = 0

    if (abs(a_x - b_x) < EPSILON) or (abs(b_x - c_x) < EPSILON):
        result = 0
    elif (c_x - a_x) == 0:
        # Otherwise we could have a ZeroDivisionError
        result = INFINITY
    else:
        result = np.abs(((a_y + (c_y - a_y) * (b_x - a_x) / (c_x - a_x) - b_y)))

    return result

def get_pip(data, x, y, threshold):
    #print x,' - ', y
    max_dis = 0
    max_id = 0
    ###在[x,y]之间遍历找到一个距离最大的点
    for i in range(x + 1, y):
        dis = vertical_distance(data[x], data[i], data[y])
        if dis > max_dis:
            max_dis = dis
            max_id = i

    ###如果最大距离还是小于阈值，那就不分割了，返回空值
    if max_dis < threshold:
        return []
    else:
        left = get_pip(data, x, max_id, threshold)
        right = get_pip(data, max_id, y, threshold)
        retu = left + [max_id] + right
        return retu

###data的数据形式为(x[i], time_series[i])
def fastpip_threshold(data, threshold):
    ldata = len(data)
    pips = [0] + get_pip(data, 0, ldata-1, threshold) + [ldata-1]
    return pips

def test_yield():
    ret = []
    for x in range(1,10):
        ret.append(int(x))
        if len(ret) > 2:
            yield ret
            ret = []

# if __name__ == '__main__':
#     # xxx = test_yield()
#     # for i in xxx:
#     #     print('i',i)
#     print('fastpip_threshold',fastpip_threshold([(0,0),(1,0),(2,1),(3,3),(4,4),(5,1),(6,0),(7,0)], 1))
#     ###[0,1,4,5,7]