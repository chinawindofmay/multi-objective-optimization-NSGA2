import time, timeit
import numpy as np




def slove_matrix_test():
    A = np.mat('1,2,9; 1,4,10; 3,9,10; 2,3,10; 5,4,8')
    print(A)
    b = np.array([[3, 6,4,8,7]]).T
    print(b)
    # r = np.linalg.solve(A, b)
    r = np.linalg.lstsq(A, b, rcond=None)
    print(r)


def slove_matrix_camera_13():
    origin_point=np.array([119.088314,31.921308])
    xyz = np.array([[119.08954784274101, 31.923024173586704, 25.4], [119.08974163234234, 31.923085071272574, 25.1],
           [119.09100294113159, 31.922237053139384, 22.3], [119.09130401909351, 31.921339511335965, 22.6],
           [119.09147165715694, 31.920691249633293, 22.6], [119.09159436821938, 31.920448221460987, 23.4],
           [119.09130871295929, 31.92015055376141, 23.3], [119.09101366996765, 31.92006916452862, 24.4],
           [119.09069985151291, 31.919992328473526, 25.7], [119.09394800662994, 31.917385556133283, 17.3],
           [119.08967457711697, 31.91927063557891, 30.9], [119.08886052668095, 31.919334950711807, 27.3],
           [119.0853314101696, 31.91988646535915, 17.1], [119.08516645431519, 31.920262108117328, 16.4],
           [119.0942832827568, 31.91764509835025, 16.5], [119.08817857503891, 31.92662162097826, 20.0],
           [119.08366374671459, 31.92131105393831, 11.7], [119.08090308308601, 31.921898981985507, 7.4],
           [119.089095890522, 31.924460096136755, 21.2]])
    xyz[:,0]=(xyz[:,0]-origin_point[0])*111300
    xyz[:,1]=(xyz[:,1]-origin_point[1])*111300
    A=xyz
    # A = np.mat('1,2,9; 1,4,10; 3,9,10; 2,3,10; 5,4,8')
    print(A)
    print(A.shape)

    Pan = np.array([57.000000849366188, 59.900000892579556, 94.500001408159733, 115.90000172704458, 130.3000019416213,
           134.6000020056963, 141.80000211298466, 145.8000021725893, 150.700002245605, 156.80000233650208,
           178.80000266432762, 195.60000291466713, 268.40000399947166, 276.30000411719084, 153.300002284348,
           25.600000381469727, 297.100004427135, 302.4, 38.200000569224358])
    Tilt = np.array([351.19999980926514, 352.09999990463257, 354.40000009536743, 355.19999980926514, 356.29999995231628,
            356.70000004768372, 356.39999985694885, 355.90000009536743, 355.40000009536743, 359.79999999701977,
            355.69999980926514, 353.09999990463257, 350.39999961853027, 349.80000019073486, 359.79999999701977, 355.0,
            350.69999980926514, 352.7, 353.0])
    Tilt=(360-Tilt)*10
    b = np.array(Tilt).T
    print(b)
    print(b.shape)

    A2 = np.full(shape=(A.shape[0], 13), fill_value=0.0)
    A2[:, 0] = np.power(A[:, 0], 3)  # x3
    A2[:, 1] = np.power(A[:, 1], 3)  # y3
    A2[:, 2] = np.power(A[:, 2], 3)  # z3

    A2[:, 3] = np.power(A[:, 0], 2) * A[:, 1]  # x2y
    A2[:, 4] = np.power(A[:, 0], 2) * A[:, 2]  # x2z

    A2[:, 5] = np.power(A[:, 1], 2) * A[:, 2]  # y2z
    A2[:, 6] = np.power(A[:, 1], 2) * A[:, 0]  # y2x

    A2[:, 7] = np.power(A[:, 2], 2) * A[:, 0]  # z2x
    A2[:, 8] = np.power(A[:, 2], 2) * A[:, 1]  # z2y

    A2[:, 9] = A[:, 0] * A[:, 1] * A[:, 2]  # xyz

    A2[:, 10] = A[:, 0]  # x
    A2[:, 11] = A[:, 1]  # y
    A2[:, 12] = A[:, 2]  # z

    r = np.linalg.lstsq(A2, b, rcond=None)
    print("result:")
    print(r)

    test_data=np.array([[119.08519059419632, 31.92133552730082, 17.0],
           [119.08897116780281, 31.920046967452627, 26.0]])

    test_data[:, 0] = (test_data[:, 0]-origin_point[0]) * 111300
    test_data[:, 1] = (test_data[:, 1]-origin_point[1]) * 111300
    predict=r[0][0]*test_data[:,0]**3+r[0][1]*test_data[:,1]**3+r[0][2]*test_data[:,2]**3+\
    r[0][3] * test_data[:, 0] ** 2 * test_data[:, 1]  + r[0][4] * test_data[:, 0] ** 2* test_data[:, 2] +\
    r[0][5] * test_data[:, 1] ** 2 * test_data[:, 2]  + r[0][6] * test_data[:, 1] ** 2* test_data[:, 0] +\
    r[0][7] * test_data[:, 2] ** 2 * test_data[:, 0]  + r[0][8] * test_data[:, 2] ** 2* test_data[:, 1] +\
    r[0][9] * test_data[:, 0]  * test_data[:, 1] * test_data[:, 2] +\
    r[0][10] * test_data[:, 0] +\
    r[0][11] * test_data[:, 1] +\
    r[0][12] * test_data[:, 2]

    print("pridict")
    print(predict)
    test_data_true_result_pan = np.array([297.70000443607569, 185.5000027641654])
    test_data_true_result_tilt = np.array([348.09999942779541, 349.19999980926514])
    print("test_pan",test_data_true_result_pan)
    print("test_tilt",360-test_data_true_result_tilt)







def clock(func):
    def clocked(*args, **kwargs):
        t0 = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed = timeit.default_timer() - t0
        name = func.__name__
        arg_str = ', '.join(repr(arg) for arg in args)
        print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        return result
    return clocked


@clock
def run(seconds):
    time.sleep(seconds)
    return time


if __name__ == '__main__':
    # run(1)
    # slove_matrix_test()
    slove_matrix_camera_13()