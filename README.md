# simplexCuda
Implementation of Simplex Algorithm in Cuda

The program solves an LP with Simplex algorithm.</br>

#How to run the program?
$nvcc simplex.cu
$./a.out <datafile> <i/n> <max_iteration>
sample run:</br>
$./a.out TestCases/testcase1.txt n 500

#Test case format
Consider the following input

5 5
0 0 -7 -7 -3 
70 2 6 5 4 -1
78 2 5 9 6 1
119 3 9 8 8 -1
34 1 2 6 2 -1


First line means: number of rows = 5, number of columns = 5
second line means: objective function is- maximize z where z = 0.x1 + 7.x2 + 7.x3 + 3.x4
3rd line means: the constraint is  2.x1 + 6.x2 + 5.x3 + 4.x4 <= 70
4th line means: the constraint is 2.x1 + 5.x2 + 9.x3 + 6.x4 >=78
5th line means: the constraint is 3.x1 + 9.x2 + 8.x3 + 9.x4 <=119
6th line means: the constraint is 1.x1 + 2.x2 + 6.x3 + 2.x4 <=34
