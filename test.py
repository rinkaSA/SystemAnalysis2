
def optimisation_l(f, g, x, t, ll, t0, tmp):
    x_n_j0 = [x]
    for i in range(len(t)):
        x_n_j0.append(calculate_equation_1(f, x_n_j0[i], g, ll))
    sum_x1_0 = J(x_n_j0, t0)
    print("Summ j0", sum_x1_0)
    flag = True
    l_opti = 0.0
    i = 0
    x_n_jd = []
    while i >= 0:
        x_n_jd.append(x)
        ll[tmp][0] += delta_l
        for v in range(len(t)):
            x_n_jd.append(calculate_equation_1(f, x_n_jd[v], g, ll))
        sum_x1_delta = J(x_n_jd, t0)
        print('J(delta)', sum_x1_delta)
        differ = sum_x1_0 - sum_x1_delta
        print("difference ", differ)
        if differ < 0:
            l_opti = ll[tmp][0]
            if i == 0:
                flag = False
                ll[tmp][0] = 0.0
            break
        x_n_jd.clear()
        i += 1
    if flag is True :
        print("L opti ", l_opti)
    else:
        ll[tmp][0] = l_default
    while flag is False:
        x_n_jd.append(x)
        ll[tmp][0] -= delta_l
        for v in range(len(t)):
            x_n_jd.append(calculate_equation_1(f, x_n_jd[v], g, ll))
        sum_x1_delta = J(x_n_jd, t0)
        print('J(delta)', sum_x1_delta)
        differ = sum_x1_0 - sum_x1_delta
        print("difference ", differ)
        if differ > 0:
            print("L opti ", ll[tmp][0])
            flag = True
            break
        x_n_jd.clear()

    return x_n_jd
