import numpy as np
import matplotlib.pyplot as plt


l_default = -0.025
delta_l = 0.005
u_with_line = 1


def create_A(a1, a2):
    a = np.zeros((3, 3))
    a[0][1] = 1
    a[1][2] = 1
    a[2][0] = -1
    a[2][1] = -a1
    a[2][2] = -a2
    return a


def create_b():
    b = np.zeros((3, 1))
    b[0][0] = 0
    b[1][0] = 0
    b[2][0] = 1
    return b


def create_c():
    c = []
    for i in range(3):
        value = np.zeros((1, 3))
        value[0][i] = 1
        c.append(value)
    return c


def calculate_f(a, t, q):
    f = np.eye(3)
    try:
        if isinstance(q, int) == 0:
            raise Exception('Not integer accuracy!')
        elif q < 2 or q > 10:
            raise Exception("Not allowed accuracy!")
    except Exception as e:
        print(e)
        return 0
    for i in range(1, q+1):
        f += np.linalg.matrix_power(a.dot(t), i) / np.math.factorial(i)
    return f


def calculate_g(a, q, t0):
    b = create_b()
    e = np.eye(3)
    tmp = e*t0
    for j in range(2, q+1):
        tmp += (np.linalg.matrix_power(a, j-1).dot(e)*(t0**j))/np.math.factorial(j)

    return tmp.dot(b)


def calculate_equation_1(f, x, g, l):
    u_k = -l.transpose().dot(x) + u_with_line
    res = f.dot(x) + g*u_k
    return res


def calculate_equation_2(x, c, y):
    y.append(c.dot(x))
    return y


def variables():
    a1 = float(input('Input a1 for matrix A: '))
    a2 = float(input('Input a2 for matrix A: '))
    t0 = float(input('Input period kvantuvannya t0: '))
    q = int(input('Input accuracy q: '))
    var = [a1, a2, t0, q]
    return var


def J(x, t0):
    sum = 0.000000000000
    for i in range(len(x)):
        sum += abs(x[i][0][0] - 1)*t0

    return sum


def optimisation_l(f, g, x, t, ll, t0, tmp):
    x_n_j0 = [x]
    for i in range(len(t)):
        x_n_j0.append(calculate_equation_1(f, x_n_j0[i], g, ll))
    j0 = J(x_n_j0, t0)
    flag = False
    print('J0 ', j0)
    ll[tmp][0] += delta_l
    x_n_j_i = [x]
    for i in range(len(t)):
        x_n_j_i.append(calculate_equation_1(f, x_n_j_i[i], g, ll))
    j_i = J(x_n_j_i, t0)
    while j0-j_i > 0:
        x_n_j_i.clear()
        x_n_j_i = [x]
        ll[tmp][0] += delta_l
        for i in range(len(t)):
            x_n_j_i.append(calculate_equation_1(f, x_n_j_i[i], g, ll))
        j_i = J(x_n_j_i, t0)
        print("J(delta) ",j_i)
        print('difference ', j0-j_i)
        flag = True

    print('opti', ll[tmp][0])
    if flag is False:
        ll[tmp][0] = l_default
    while j0-j_i <= 0 and flag is False:
        x_n_j_i.clear()
        x_n_j_i = [x]
        ll[tmp][0] -= delta_l
        for i in range(len(t)):
            x_n_j_i.append(calculate_equation_1(f, x_n_j_i[i], g, ll))
        j_i = J(x_n_j_i, t0)
        print("J(delta) ", j_i)
        print('difference ', j0 - j_i)
    print('opti', ll[tmp][0])
    return x_n_j_i


def interface():
    val = variables()
    x = np.zeros((3, 1))
    t = list(np.arange(0, 30+val[2], val[2]))
    a = create_A(val[0], val[1])
    f = calculate_f(a, val[2], val[3])
    g = calculate_g(a, val[3], val[2])
    l = np.array([[0.0],
                  [0.0],
                  [0.0]])
    tmp = int(input("Input variant 1 or 2: "))
    if tmp != 1 and tmp != 2:
        print("Bad choice! Options for variant are only 1 and 2!")
        return 0
    l[tmp][0] = l_default
    x_n = []
    x_n.append(x)
    for i in range(len(t)):
        x_n.append(calculate_equation_1(f, x_n[i], g, l))
    l[tmp][0] = 0.0
    print(' J default', J(x_n, val[2]))
    x_n_optimazed = optimisation_l(f, g, x, t, l, val[2], tmp)
    c = create_c()
    y1 = []
    y1_opti = []
    for i in x_n:
        y1 = calculate_equation_2(i, c[0], y1)
    for i in x_n_optimazed:
        y1_opti = calculate_equation_2(i, c[0], y1_opti)
    lol1 = []
    for i in y1:
        a = i.tolist()
        lol1.append(a[0])
    list_y1 = []
    for m in range(len(lol1)-1):
        list_y1.append(lol1[m][0])
    lol2 = []
    for i in y1_opti:
        a = i.tolist()
        lol2.append(a[0])
    list_y1_opti = []
    for m in range(len(lol2) - 1):
        list_y1_opti.append(lol2[m][0])

    plt.xlabel('t')
    plt.ylabel('x1(t)')
    plt.grid()
    plt.xticks(np.arange(t[0], t[-1]+1, 5))
    plt.plot(t, list_y1, label='not opti', color='purple')
    plt.plot(t, list_y1_opti, label='opti', color='green')
    plt.legend()
    plt.show()


interface()
