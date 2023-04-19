# -*-coding:utf-8 -*-
import time
import numpy as np
import pandas as pd
from docplex.mp.model import Model
import math as mt
import matplotlib.pyplot as plt

class DualASC_Cross:
    def __init__(self):
        self.mDir = 'D:\yzwang\PROBLEM\ASC_Scheduling\TwinASC-Cross\extend1_dynamic scheduling for Dual ASCs\英文Transfer\\'
        self.mDir_input = 'mFiles\\'
        self.mData = {}

    def data_generation(self, N, NV, NB, NAME):
        dataset = pd.read_excel(self.mDir + self.mDir_input + 'TL-GA dynamic scheduling.xlsx',
                                sheet_name=NAME, skiprows=0)

        df = pd.DataFrame(dataset)
        JJ = list(df['No.'].values)
        BAY = list(df['Bay'].values)
        XOB = list(df['Bay_XO'].values)
        XDB = list(df['Bay_XD'].values)
        TSI = df['TS'].values
        TEI = df['TE'].values
        BTI = df['BT'].values

        # 任务数量 1,2,...,N
        Job_set = JJ[1:N+1]
        Bay = BAY[1:N+1]
        XO_B = XOB[1:N+1]
        XD_B = XDB[1:N+1]
        TSO = TSI[1:N+1]
        TEO = TEI[1:N+1]
        BTO = BTI[1:N+1]

        # 构造数据（起始索引从1开始）
        TS = [s for s in TSO]
        TE = [e for e in TEO]
        BT = [b for b in BTO]
        TS.insert(0, 0)
        TS.append(0)
        TE.insert(0, 0)
        TE.append(0)
        BT.insert(0, 0)
        BT.append(0)


        # /***--- define the coordinates of each job ---***/
        Bay_length = 1
        XO = [x * Bay_length for x in XO_B]
        XD = [x * Bay_length for x in XD_B]

        # 任务的起始-目标位置
        OD = np.zeros([N+2, 2])
        for i in range(N):
            OD[i+1][0] = XO[i]
            OD[i+1][1] = XD[i]
        for j in range(2):
            OD[0][j] = 0
            OD[N + 1][j] = 0

        # /***--- define the job set ---***/
        J = Job_set
        N = len(J)                               # the number of job set
        J_Start = [i for i in range(N + 1)]      # the job set contains dummy starting job
        J_End = [i for i in range(1, N + 2)]     # the job set contains dummy finishing job
        J_D = [i for i in range(0, N + 2)]       # the job set contains two dummy jobs
        K = [1, 2]                          # outer ASC and inner ASC
        B = [i for i in range(1, NB + 1)]   # buffer site list
        V = [i for i in range(1, NV + 1)]   # AGVs list

        # /***--- define the cost matrix ---***/
        TII = [(abs(XD[i] - XO[i])) for i in range(N)]
        TII.insert(0, 0)
        TII.append(0)

        C = np.zeros([N+1, N+1])
        for i in range(N):
            for j in range(N):
                C[j+1][i+1] = abs(XD[i] - XO[j])
                C[i][i] = 99999
        C[N][N] = 99999
        C = abs(pd.DataFrame(C))

        T_Cost = np.zeros([N + 2, N + 2])
        for a in range(N+1):
            for b in range(N+1):
                T_Cost[a][b] = C[a][b]
        for p in range(N):
                T_Cost[0][p+1] = abs(XD[p] - XO[p])
        for a in range(N + 2):
            for b in range(N + 2):
                if b == N + 1:
                    T_Cost[a][b] = 0
                elif b == 0:
                    T_Cost[a][b] = 99999
                elif a == N + 1:
                    T_Cost[a][b] = 99999

        T_Cost[0][N+1] = 99999
        T_Cost[N + 1][N + 1] = 99999
        T_Cost = np.round(T_Cost)

        self.mData['job_Num'] = N
        self.mData['jobs'] = Job_set
        self.mData['jobset with dummy tasks'] = J_D
        self.mData['jobset with dummy start task'] = J_Start
        self.mData['jobset with dummy end task'] = J_End
        self.mData['Bay'] = Bay
        self.mData['ASC'] = K
        self.mData['Buffer'] = B
        self.mData['AGV'] = V
        self.mData['original location'] = XO
        self.mData['distination location'] = XD
        self.mData['OD_bay'] = OD
        self.mData['Preparative time'] = TII
        self.mData['Buffer time'] = BT
        self.mData['Cost'] = T_Cost

    def solve_cplex(self):
        ### ***-- the model of Dual-ASC's scheduling by Wang Yao-Zong
        ### ***--- update at 20230301 by HBG
        ### ***--- 交箱序列不确定下堆场起重机动态调度。

        # /***--- the value of dataset delivery ---***/
        J = self.mData['jobs']
        J_D = self.mData['jobset with dummy tasks']
        J_S = self.mData['jobset with dummy start task']
        J_E= self.mData['jobset with dummy end task']
        L = self.mData['Bay']
        K = self.mData['ASC']
        B = self.mData['Buffer']
        V = self.mData['AGV']
        OD = self.mData['OD_bay']
        TII = self.mData['Preparative time']
        T_C = self.mData['Cost']


        N = len(J)
        JS = 0              # dummy start task
        JE = N + 1          # dummy end task
        T_S = 7.5             # the pickup time of each job is handled by ASC
        T_H = 7.5             # the put down time of each job is handled by ASC
        T_I = 7.5             # the delay time setting when the interference occuring

        alpha = 0.1         # the adjustment parameter for M
        beta = 1.0          # the adjustment parameter for M
        M0 = sum(TII) +sum(sum(T_C))
        M1 = alpha * M0
        M2 = beta * M0


        # /***--- the solving for Model starts now ---***/
        begin_t = time.time()

        # define Model
        m = Model("DualASC_Cross")

        # define Variables
        x = m.binary_var_cube(N + 2, N + 2, len(K) + 1, name='x')
        y = m.binary_var_matrix(N + 2, len(K) + 1, name='y')
        z = m.binary_var_matrix(N + 2, len(B) + 1, name='z')
        u = m.binary_var_matrix(N + 2, N + 2, name='u')
        a = m.binary_var_matrix(N + 2, N + 2, name='a')
        b = m.binary_var_matrix(N + 2, N + 2, name='b')

        t_YS = m.continuous_var_matrix(N + 2, len(K) + 1, lb=0, name='t_YS')
        t_YE = m.continuous_var_matrix(N + 2, len(K) + 1, lb=0, name='t_YE')
        t_S = m.continuous_var_matrix(N + 2, 1, lb=0, ub=9000, name='t_S')
        t_E = m.continuous_var_matrix(N + 2, 1, lb=0, ub=9000, name='t_E')

        t = m.continuous_var_matrix(1, 1, lb=0, name='t')

        # Objective
        m.minimize(t[0, 0])

        # Constraints
        m.add_constraints(t[0, 0] >= t_E[i, 0] for i in J_S)
        # m.add_constraints(t[0, 0] >= t_E[i, 0] for i in J)

        m.add_constraints(m.sum(x[JS, j, k] for j in J) == 1 for k in K)
        m.add_constraints(m.sum(x[i, JE, k] for i in J) == 1 for k in K)
        m.add_constraints(m.sum(x[i, j, k] for j in J_E) == y[i, k] for i in J_S for k in K)
        m.add_constraints(m.sum(y[i, k] for k in K) == 1 for i in J)
        m.add_constraints(m.sum(x[i, j, k] for i in J_S) == m.sum(x[j, i, k] for i in J_E)
                          for j in J for k in K)

        m.add_constraints(x[i, j, k] + x[j, i, k] <= 1 for i in J for j in J for k in K)
        m.add_constraints(x[i, i, k] == 0 for i in J for k in K)
        m.add_constraints(t_YE[i, k] >= t_YS[i, k] + TII[i] + T_S + T_H + (y[i, k] - 1) * M1
                          for i in J for k in K)
        # m.add_constraints(t_E[i, 0] >= t_S[i, 0] + TII[i] + T_S + T_H+(y[i, k] - 1) * M1
        #                   for i in J for k in K)
        m.add_constraints(t_YS[j, k] >= t_YE[i, k] + T_C[i][j] + (x[i, j, k] - 1) * M1
                          for i in J_S for j in J_E for k in K)

        # m.add_constraints(t_S[j, 0] >= t_E[i, 0] + T_C[i][j] + (x[i, j, k] - 1) * M1
        #                   for i in J for j in J for k in K)
        #

        m.add_constraints(t_YS[i, k] <= (y[i, k]) * M1 for i in J_S for k in K)
        m.add_constraints(t_YE[i, k] <= (y[i, k]) * M1 for i in J_S for k in K)
        m.add_constraints(t_S[i, 0] == m.sum(t_YS[i, k] for k in K) for i in J_S)
        m.add_constraints(t_E[i, 0] == m.sum(t_YE[i, k] for k in K) for i in J_S)

        m.add_constraints(t_YS[i, k] >= 0 for k in K for i in J_S)
        m.add_constraints(t_YE[i, k] >= 0 for k in K for i in J_S)

        ### 干涉规避处理
        m.add_constraints(a[i, j] + a[j, i] == 1 for i in J for j in J if OD[i][0] ==
                          OD[j][0] and i != j)
        m.add_constraints(b[i, j] + b[j, i] == 1 for i in J for j in J if OD[i][1] ==
                          OD[j][1] and i != j)
        m.add_constraints(t_S[j, 0] >= t_S[i, 0] + T_I + (a[i, j] - 1) * M1 \
                          for i in J for j in J if OD[i][0] == OD[j][0] and i != j)
        m.add_constraints(t_E[j, 0] >= t_E[i, 0] + T_I + (b[i, j] - 1) * M1 \
                          for i in J for j in J if OD[i][1] == OD[j][1] and i != j)

        m.parameters.timelimit = 3600             # 最大求解时间
        # m.parameters.mip.tolerances.mipgap = 0.3    # best integer 与 best bound之间的GAP
        solution = m.solve(log_output=False)

        end_t = time.time()
        time_cost = end_t - begin_t
        print('time consume: %.3f' % time_cost)
        OBJ = solution.get_objective_value()
        print('Obj: ', OBJ)

        Sequence1, Sequence2, TS, TE, tour1, tour2 = [], [], [], [], [], []

        if solution:
            # print("\nSolution:\n")
            print('N = ' + str(N) + ' Objective: ', OBJ)
            stri = 'real-time(Model) solution_N = ' + str(N) + '.txt'
            filename = self.mDir + self.mDir_input + stri
            file_handle = open(filename, 'w')

            file_handle.write("***********------***********\n")
            file_handle.write('The makespan and calculate time are:\n')

            file_handle.write("[%s]\n" % mt.ceil(solution.get_objective_value()))
            file_handle.write('%.4f\n' % time_cost)

            file_handle.write("***********------***********\n")
            file_handle.write('The assignment relation is as follows: \n[job, ASCer]\n')
            for i in range(1, N + 1):
                for k in range(len(K) + 1):
                    if y[i, k].solution_value >= 0.5:
                        if(k==1):
                            Sequence1.append(i)
                        if(k==2):
                            Sequence2.append(i)
                        file_handle.write("[%s,%s]\n" % (i, k))

            file_handle.write("***********------***********\n")
            file_handle.write("The order relation of jobs are as follows:\n[pre-Order post-Order ASCer]\n")
            for i in range(1, N + 1):
                for j in range(1, N + 1):
                    for k in range(len(K) + 1):
                        if x[i, j, k].solution_value >= 0.5:
                            if(k==1):
                                tour1.append((i, j))
                            if(k==2):
                                tour2.append((i, j))
                            file_handle.write("[%s,%s,%s]\n" % (i, j, k))

            file_handle.write("***********------***********\n")
            file_handle.write("The time interval of each job is as follows:\n[job, TS, TE]\n")

            ################## 序列  ##############
            tour = {}
            for k in K:
                list_i = []
                list_j = []
                list_k = [0]
                for i in range(N+2):
                    for j in range(N+2):
                        if x[i, j, k].solution_value > 0.5:
                            list_i.append(i)
                            list_j.append(j)
                for l in range(len(list_i)):
                    ds = list_i.index(list_k[-1])
                    list_k.append(list_j[ds])
                tour[k] = list_k
                print("K=", k, ":", list_k)


            for i in range(N + 2):
                if (t_S[i, 0].solution_value >= 0 and t_E[i, 0].solution_value >= 0):
                    TS.append(mt.ceil(t_S[i, 0].solution_value))
                    TE.append(mt.ceil(t_E[i, 0].solution_value))
                    file_handle.write("[%s,%s,%s]\n" % (i, mt.ceil(t_S[i, 0].solution_value),
                                                        mt.ceil(t_E[i, 0].solution_value)))
            print('TS: ', TS)
            print('TE: ', TE)

            file_handle.close()

        return Sequence1, Sequence2, TS, TE, tour1, tour2, T_H, tour


    def plot_figure(self, Sequence1, Sequence2, TS, TE, tour1, tour2, T_H, tour):
        print("plot for scheduling process...")
        XO1 = self.mData['original location']
        XD1 = self.mData['distination location']
        XO = [o for o in XO1]
        XD = [d for d in XD1]
        XO.insert(0, 999)
        XD.insert(0, 999)

        # print(tour)
        plt.plot([0, XO[tour[1][1]]], [0, abs(XO[tour[1][1]]- XD[tour[1][1]])],  '--', label='ASC1UnloadedPath', color='b', linewidth=1)
        inxd1 = 0
        for s1 in Sequence1:
            if inxd1 == 0:
                plt.plot([XO[s1], XD[s1]],
                         [TS[s1] + T_H, TE[s1] - T_H], label='ASC1LoadPath', color='b', linewidth=1)
            else:
                plt.plot([XO[s1], XD[s1]],
                         [TS[s1] + T_H, TE[s1] - T_H], color='b', linewidth=1)
            plt.plot([XO[s1], XO[s1]],
                     [TS[s1], TS[s1] + T_H], color='b', linewidth=1)
            plt.plot([XD[s1], XD[s1]],
                     [TE[s1] - T_H, TE[s1]], color='b', linewidth=1)
            inxd1 += 1

        for i in range(len(tour1)):
            plt.plot([XD[tour1[i][0]], XO[tour1[i][1]]],
                     [TE[tour1[i][0]], TS[tour1[i][1]]], '--', color='b', linewidth=1)

        plt.plot([0, XO[tour[2][1]]], [0, abs(XO[tour[2][1]] - XD[tour[2][1]])], '--', label='ASC2UnloadedPath', color='r', linewidth=1)
        inxd2 = 0
        for s2 in Sequence2:
            if inxd2 == 0:
                plt.plot([XO[s2], XD[s2]],
                         [TS[s2] + T_H, TE[s2] - T_H], label='ASC2LoadPath', color='r', linewidth=1)
            else:
                plt.plot([XO[s2], XD[s2]],
                         [TS[s2] + T_H, TE[s2] - T_H], color='r', linewidth=1)
            plt.plot([XO[s2], XO[s2]],
                     [TS[s2], TS[s2] + T_H], color='r', linewidth=1)
            plt.plot([XD[s2], XD[s2]],
                     [TE[s2] - T_H, TE[s2]], color='r', linewidth=1)
            inxd2 += 1
        for i in range(len(tour2)):
            plt.plot([XD[tour2[i][0]], XO[tour2[i][1]]],
                     [TE[tour2[i][0]], TS[tour2[i][1]]], '--', color='r', linewidth=1)


        # plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标
        plt.rc('font', family='Times New Roman')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("bay (b)", fontsize=14)
        plt.ylabel("time (u)", fontsize=14)

        # 参数1 用于控制 legend 的左右移动，值越大越向右边移动;
        # 参数2 用于控制 legend 的上下移动，值越大，越向上移动。
        # bbox_to_anchor用于微调图例的位置
        plt.legend(loc='upper center', bbox_to_anchor=(0.8, 0.35), ncol=1, fancybox=True, shadow=True)
        plt.savefig("test.svg", dpi=600, format="svg")
        plt.show()




    def experiment(self):

        # for i in range(1, 3):
        #     N = int(i)
        N = 6    # the number of job
        NV = 10     # the number of AGV
        NB = 5      # the number of buffer site
        # NAME = "input_test"
        NAME = 'dynamic'

        self.data_generation(N, NV, NB, NAME)
        Sequence1, Sequence2, TS, TE, tour1, tour2, T_H, tour = self.solve_cplex()
        self.plot_figure(Sequence1, Sequence2, TS, TE, tour1, tour2, T_H, tour)

if __name__ == '__main__':
    a = DualASC_Cross()
    a.experiment()


