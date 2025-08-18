# This is a sample Python script.
import copy
import random


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x_1 = 0.2
    x_2 = 0.3
    u_j = 0.4 # 交叉的时候举例子用的，是(0,1)之间的额随机数
    r_j = pow(2*u_j, 0.5)
    y_1 = 0.5 * ((1+r_j) * x_1 + (1-r_j)*x_2)
    y_2 = 0.5 * ((1-r_j) * x_1 + (1+r_j)*x_2)
    print(y_1, y_2)

    r = 0.3 #变异的时候举例子用的，是(0,1)之间的额随机数
    print(0.8 * (1 - 0) * r)



