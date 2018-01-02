# simplexCuda
Implementation of Simplex Algorithm in Cuda

The program solves an LP with Simplex algorithm.</br>

<h1>How to run the program?</h1>
$nvcc simplex.cu</br>
$./a.out &ltdatafile&gt &lti/n&gt &ltmax_iteration&gt</br>
sample run:</br>
$./a.out TestCases/testcase1.txt n 500</br>

<h2>Test case format</h2>
Consider the following input:</br>

5 5</br>
0 0 -7 -7 -3</br>
70 2 6 5 4 -1</br>
78 2 5 9 6 1</br>
119 3 9 8 8 -1</br>
34 1 2 6 2 -1</br>


First line means: number of rows = 5, number of columns = 5</br>
second line means: objective function is- maximize z where z = 0.x1 + 7.x2 + 7.x3 + 3.x4</br>
3rd line means: the constraint is  2.x1 + 6.x2 + 5.x3 + 4.x4 <= 70</br>
4th line means: the constraint is 2.x1 + 5.x2 + 9.x3 + 6.x4 >=78</br>
5th line means: the constraint is 3.x1 + 9.x2 + 8.x3 + 9.x4 <=119</br>
6th line means: the constraint is 1.x1 + 2.x2 + 6.x3 + 2.x4 <=34</br>
