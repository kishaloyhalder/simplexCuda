/*
  What: Simplex in CUDA
  AUTHOR: kishaloy.halder@gmail.com

  What: Solves LP Problem with Simplex:
    { maximize cx : Ax <= b, x >= 0 }.
  Input: { m, n, Mat[m x n] }, where:
    b = mat[1..m,0] .. column 0 is b >= 0, so x=0 is a basic feasible solution.
    c = mat[0,1..n] .. row 0 is z to maximize, note c is negated in input.
    A = mat[1..m,1..n] .. constraints.
    x = [x1..xm] are the named variables in the problem.
    Slack variables are in columns [m+1..m+n]

  USAGE:
    1. Read the problem data from a file:
      $ cat problem.txt
            m n
            0  -c1 -c2 sign1
            b1 a11 a12 sign2
            b2 a21 a11 sign3
      $ nvcc simplex.cu
      $ ./a.out <datafile> <i/n> <max_iteration>
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

#define NUMBER_OF_ELEMENTS_PER_THREAD 10
#define M 5000
#define N 5000

#define THREADS_PER_BLOCK 512

static const double epsilon   = 0.00001;

int equal(double a, double b) { return fabs(a-b) < epsilon; }
__device__ int equalsTo(double a, double b)
{
	double epsilon = 0.00001;
	return fabs(a-b) < epsilon;
}

typedef struct {
	double * table;
	int numberOfRows;
	int numberOfColumns;
	int numberOfColumnsInOriginalMatrix;
	int firstPivotRow; 
	int pitch;
} Tableau;

typedef struct {
	double value;
	int index;
} Element;

typedef struct {
  int m, n; // m=rows, n=columns, mat[m x n]
  double mat[M][N];
} Tab;

Tab * demoTab;

void nl(int k){ int j; for(j=0;j<k;j++) putchar('-'); putchar('\n'); }

/* // Size of tableau [4 rows x 5 columns ]
	// Max: z = 0.5*x + 3*y + z + 4*w,
	//    x + y + z + w <= 40 .. b1
	//  -2x - y + z + w <= 10 .. b2
	//        y     - w <= 10 .. b3
	Example input file for read_tableau:
     4 5							
      0   -0.5  -3 -1  -4 
     40    1     1  1   1 -1 
     10   -2    -1  1   1 -1
     10    0     1  0  -1 -1 
*/

void updateHostTable(double * table, int pitch, int numberOfRows, int numberOfColumns){
	for(int i=0;i<numberOfRows;i++)
	{
		cudaMemcpy(demoTab->mat[i], (double *)((char *)table + i*pitch), numberOfColumns*sizeof(double), cudaMemcpyDeviceToHost);
	}
	return;
}

void writetofile(FILE * fp,double * tempArray, int numberOfColumns, int direction){
	int i=0;
	double b=tempArray[0];
	for(i=1;i<numberOfColumns;i++)
	{
		if(i==1){
			fprintf(fp, "%lfx%d ",tempArray[i],i);
		}
		else if(tempArray[i]>=0){
		fprintf(fp, "+ %lfx%d ",fabs(tempArray[i]),i);
		}
		else if(tempArray[i]<0){
		fprintf(fp, "- %lfx%d ",fabs(tempArray[i]),i);
		}
	}
	if(direction==1)
		fprintf(fp, " >= %lf", b);
	else if(direction==-1)
		fprintf(fp, " <= %lf", b);
	else	
		fprintf(fp, " = %lf", b);
	fprintf(fp,";\n");
}

void print_tableau(Tab *tab, const char* mes, int verbosityLevel) {
	if(!verbosityLevel)
		return;
	static int counter=0;
	int i, j;
	printf("\n%d. Tableau %s:\n", ++counter, mes);
	nl(70);

	printf("%-6s%5s", "col:", "b[i]");
	for(j=1;j<tab->n; j++) { printf("    x%d,", j); } printf("\n");

	for(i=0;i<tab->m; i++) {
		if (i==0) printf("max:"); else
		printf("b%d: ", i);
		for(j=0;j<tab->n; j++) {
			if (equal((int)tab->mat[i][j], tab->mat[i][j]))
				printf(" %6d", (int)tab->mat[i][j]);
			else
				printf(" %6.2lf", tab->mat[i][j]);
		}
		printf("\n");
	}
	nl(70);
}

void negateRow(double * array, int size)
{
		for(int i=0;i<size;i++)
		{
			array[i] *= -1;
		}
	return;
}

void printRow(double * array, int size)
{
		for(int i=0;i<size;i++)
		{
			printf("%lf ", array[i]);
		}
	printf("\n");
	return;
}

void read_tableau(Tableau *tab, const char * filename, int verbosityLevel) {
	int err, i, j;
	FILE * fp;

	fp  = fopen(filename, "r" );
	if( !fp ) {
		printf("Cannot read %s\n", filename); exit(1);
	}
	  //memset(tab, 0, sizeof(*tab));
	
	err = fscanf(fp, "%d %d", &tab->numberOfRows, &tab->numberOfColumns);
	if (err == 0 || err == EOF) {
		printf("Cannot read m or n\n"); exit(1);
	}
	
	int numberOfColumnsInOriginalMatrix = tab->numberOfColumns;
	tab->numberOfColumnsInOriginalMatrix = tab->numberOfColumns;
	//all the variables are unrestricted..so each xi has to replaced with xi1-xi2
	int numberOfColumnsInExtendedMatrix = (tab->numberOfColumns)*2 - 1;
	//adding slack variables
	tab->numberOfColumns = numberOfColumnsInExtendedMatrix + tab->numberOfRows;
	
	if(verbosityLevel)
	{
		demoTab->m = tab->numberOfRows;
		demoTab->n = tab->numberOfColumns;
	}
	
	//allocating memory in device for the table
	size_t pitch;	
	cudaMallocPitch((void**)&tab->table, &pitch, tab->numberOfColumns * sizeof(double), tab->numberOfRows);
	tab->pitch = pitch;
	
	double tempArray[tab->numberOfColumns];
	
	int directionOfInequality =0;
	int slackVariableIndex = -1;
	
	double temp;
	int tempArrayCounter;
	
	int tabOldNumberOfColumns=numberOfColumnsInOriginalMatrix + tab->numberOfRows -1;
	FILE * opFile=fopen("printEquations.txt","w");
	
	memset(tempArray, 0, sizeof(double)*(tab->numberOfColumns));
	for(i=0;i<tab->numberOfRows; i++) {
		tempArrayCounter = 0;
		if(i==0)
		{
			tempArray[tab->numberOfColumns-1] = 1;
		}
		else
		{
			tempArray[tab->numberOfColumns-1] = 0;
		}
		for(j=0;j<tabOldNumberOfColumns; j++) {
			if(j<numberOfColumnsInOriginalMatrix){
				if(j==0)
				{
					err = fscanf(fp, "%lf", &tempArray[tempArrayCounter]);
					tempArrayCounter++;
					if (err == 0 || err == EOF) {
						printf("Cannot read A[%d][%d]\n", i, j);
						exit(1);
					}
				}
				else
				{
					err = fscanf(fp, "%lf", &temp);
					if (err == 0 || err == EOF) {
						printf("Cannot read A[%d][%d]\n", i, j);
						exit(1);
					}
					//replacing a.xi with a.xi1 - a.xi2
					tempArray[tempArrayCounter] = temp;
					tempArrayCounter++;
					tempArray[tempArrayCounter] = temp*(-1);
					tempArrayCounter++;
				}
			}
			else if(i>=1)
			{
				//slack variables
				tempArray[tempArrayCounter] = (i==(j-numberOfColumnsInOriginalMatrix+1));
				if(tempArray[tempArrayCounter]==1)
					slackVariableIndex = tempArrayCounter;
				tempArrayCounter++;
			}
		}
		
		
		if(i>=1)	//because not needed for 0th row
		{
			//check direction of inequality
			// -1 denotes <=
			// 0 denotes =
			// 1 denotes >=
			directionOfInequality = 0;
			err = fscanf(fp, "%d", &directionOfInequality);
			if (err == 0 || err == EOF || (directionOfInequality*directionOfInequality > 1)) {
				printf("Cannot read A[%d][%d]\n", i, numberOfColumnsInOriginalMatrix);
				exit(1);
			}
			
			writetofile(opFile,tempArray,numberOfColumnsInExtendedMatrix, directionOfInequality);
			/*if(tempArray[0]<0)
			{
				negateRow(tempArray, tab->numberOfColumns);
				if(directionOfInequality==-1)
				{
					tempArray[slackVariableIndex] = 1;
				}
			}
			else if(directionOfInequality==0)
			{
				tempArray[slackVariableIndex] = 0;
			}*/
			if(directionOfInequality == 1)
			{
				tempArray[slackVariableIndex] = -1;
				if(tempArray[0]<=0)
				{
					negateRow(tempArray, tab->numberOfColumns);
				}
				else
				{
					if(tab->firstPivotRow==-1)
					{
						tab->firstPivotRow = i;
					}
				}
			}
		}

		cudaMemcpy((double *)((char *)tab->table + i *pitch), tempArray, ((tab->numberOfColumns) * (sizeof(double))), cudaMemcpyHostToDevice );	
	}
	printf("Read tableau [%d rows x %d columns] from file '%s'.\n", tab->numberOfRows, numberOfColumnsInOriginalMatrix, filename);
	
	if(verbosityLevel)
	{
		updateHostTable(tab->table,tab->pitch, tab->numberOfRows, tab->numberOfColumns);
	}
	
	fclose(fp);
}

__global__ void divideByPivotKernel(double * table, int pitch, int rowIndex, int columnIndex, double pivot, int numberOfColumns)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(index>=numberOfColumns)
		return;

	double * row = (double*)((char*)table +  rowIndex* pitch);
	row[index] /= pivot;
	return;
}

__global__ void computeNewValuesForRowsKernel(double * table, int pitch, int pivotRowIndex, int pivotColumnIndex, int numberOfColumns, int numberOfElements)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	int associatedRow = index/numberOfColumns;
	int associatedColumn = index%numberOfColumns;
	
	//not for elements in pivotColumn
	if(pivotRowIndex==associatedRow || index>=numberOfElements || associatedColumn==pivotColumnIndex)
		return;
		
	double * row = (double*)((char*)table +  associatedRow* pitch);
	double * pivotRow = (double*)((char*)table +  pivotRowIndex* pitch);
	
	double multiplier = row[pivotColumnIndex];
	row[associatedColumn] = row[associatedColumn]*pivotRow[pivotColumnIndex] - multiplier * pivotRow[associatedColumn];
	
	return;
}

__global__ void computeNewValuesForRowsInPivotColumn(double * table, int pitch, int pivotColumn, int pivotRow, int numberOfRows)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(index>=numberOfRows || index==pivotRow)
		return;
	
	double * row = (double*)((char*)table +  index* pitch);
	row[pivotColumn] = 0.;
	/*double *pivotRowArray = (double*)((char*)table +  pivotRow* pitch);
	double multiplier = row[pivotColumn];
	row[pivotColumn] -= multiplier * pivotRowArray[pivotColumn];*/
	return;
}

void pivot_on(Tableau *tab, int row, int col) {
	double pivot;

  	cudaMemcpy(&pivot, (double*)((char*)tab->table + row*(tab->pitch)) + col, sizeof(double),cudaMemcpyDeviceToHost);
	int numberOfElements = tab->numberOfColumns * tab->numberOfRows;
  	assert(pivot>0);
  
  	//launch kernel function to do the division
  	divideByPivotKernel<<< (tab->numberOfColumns + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( tab->table, tab->pitch, row, col, pivot, tab->numberOfColumns );
  
  	//launch kernel to compute the new rows
  	computeNewValuesForRowsKernel<<< (numberOfElements + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( tab->table, tab->pitch, row, col, tab->numberOfColumns, numberOfElements );

	//launch kernel to compute the pivot column
	computeNewValuesForRowsInPivotColumn<<<(tab->numberOfRows + (THREADS_PER_BLOCK-1))/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(tab->table, tab->pitch, col, row, tab->numberOfRows);
}

__global__ void findPivotIndex(double * array, Element * elements, int size, int numberOfThreadsRequired, int maxNumberOfElementsPerThread, int maxOrMin, int offset)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(index>=numberOfThreadsRequired)
		return;
	
	int start = index * maxNumberOfElementsPerThread;
	int end = (start + maxNumberOfElementsPerThread )>=size?(size-1):(start+maxNumberOfElementsPerThread -1);
	
	double solution = array[start];
	int indexToSolution = start;
	
	for(int i=start+1;i<=end;i++)
	{
		if(maxOrMin==0)	//need to find max
		{
			if(array[i]>solution && !equalsTo( array[i],solution))
			{
				solution = array[i];
				indexToSolution = i;
			}
		}
		else
		{
			if(array[i]<solution && !equalsTo( array[i],solution))
			{
				solution = array[i];
				indexToSolution = i;
			}
		}
	}
	
	elements[index].value = solution;
	elements[index].index = indexToSolution+offset;
	
	return;
}

// Find pivot_col = most negative column in mat[0][1..n]
int find_pivot_column(Tableau *tab, int verbosityLevel) {
  	int j, pivot_col;
	int offset = 0;		
	int maxOrMin = 1;	//denotes whether we need to find max element or min element for pivoting. max = 0. min = 1.
	int length = tab->numberOfColumns;
	double * pivotRow = tab->table;
	
	if(tab->firstPivotRow!=-1)
	{
		pivotRow = (double *)((char *)tab->table + (tab->firstPivotRow)*(tab->pitch));
		pivotRow++;		//eliminating the 0th element from the find max operation
		tab->firstPivotRow = -1;
		maxOrMin = 0;
		offset = 1;		//because we don't need to compare the objective value
		length -= 1;
	}
	
	
	int numberOfElementsToCompareInCPU = length/NUMBER_OF_ELEMENTS_PER_THREAD +
	(length%NUMBER_OF_ELEMENTS_PER_THREAD>0?1:0);
	Element elements[numberOfElementsToCompareInCPU];
	Element *elementsInDevice;
	int size = sizeof(Element)*numberOfElementsToCompareInCPU;
	
	cudaMalloc( (void **) &elementsInDevice, size);
	
	
	//launch kernel
	findPivotIndex<<< (tab->numberOfColumns + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>
	( pivotRow, elementsInDevice, length, numberOfElementsToCompareInCPU, NUMBER_OF_ELEMENTS_PER_THREAD, maxOrMin, offset);
	cudaMemcpy(elements, elementsInDevice, size, cudaMemcpyDeviceToHost);
  	
	//double lowest = tab->mat[0][pivot_col];
	pivot_col = elements[0].index;
	double solution = elements[0].value;
	
  	for(j=1; j<numberOfElementsToCompareInCPU; j++) {
  		if(maxOrMin==0)		//need to find max
		{
			if (elements[j].value > solution) {
      			solution = elements[j].value;
      			pivot_col = elements[j].index;
			}
		}
		else
		{
			if (elements[j].value < solution) {
      			solution = elements[j].value;
      			pivot_col = elements[j].index;
			}
		}
  	}
	if(verbosityLevel)
		printf("Most negative column in row[0] is col %d = %g.\n", pivot_col, solution);
  	if( (solution >= 0 && maxOrMin ==1) || (maxOrMin==1 && equal(solution, 0.))) {
    		return -1; // All positive columns in row[0], this is optimal.
  	}
	
	cudaFree(elementsInDevice);
  	return pivot_col;
}

__global__ void find_pivot_kernel( )
{
	return;
}

__global__ void computeRatioKernel(double * table, int pitch, int pivotColumn, double * ratioArray, int numOfRows )
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(index==0 || index>=numOfRows)
		return;
	
	double* row = (double*)((char*)table + index * pitch);
	ratioArray[index] = -1;
	if(row[pivotColumn]>0 && !equalsTo(row[pivotColumn],0.))
		ratioArray[index] = row[0]/row[pivotColumn];
	return;
}


// Find the pivot_row, with smallest positive ratio = col[0] / col[pivot]
int find_pivot_row(Tableau *tab, int pivot_col, int verbosityLevel) {
	int i, pivot_row = 0;
	double min_ratio = -1;
  
	double * ratioArrayInDevice;
	double * ratioArrayInHost;
	int size = (tab->numberOfRows)*sizeof(double);
  
	cudaMalloc( (void **) &ratioArrayInDevice, size);
	ratioArrayInHost = (double *)malloc(size);
  
	if(verbosityLevel)
		printf("Ratios A[row_i,0]/A[row_i,%d] = [",pivot_col);
  
	//launch kernel to compute the ratios
	computeRatioKernel<<< (tab->numberOfRows + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( tab->table, tab->pitch, pivot_col, ratioArrayInDevice, tab->numberOfRows );
  
	cudaMemcpy( ratioArrayInHost, ratioArrayInDevice, size, cudaMemcpyDeviceToHost );
  
	for(i=1;i<tab->numberOfRows;i++){
		double ratio = ratioArrayInHost[i];
		if(verbosityLevel)
			printf("%3.2lf, ", ratio);
		if ( (ratio >= 0  && ratio < min_ratio ) || min_ratio < 0 || (equal(ratio, 0.) && ratio<min_ratio)) {
			min_ratio = ratio;
			if(equal(ratio, 0.))
				min_ratio = 0.0;
			pivot_row = i;
		}
	}  
	if(verbosityLevel)
		printf("].\n");
  
	cudaFree(ratioArrayInDevice);
	free(ratioArrayInHost);
  
	if (min_ratio == -1)
		return -1; // Unbounded.
	if(verbosityLevel)
		printf("Found pivot A[%d,%d], min positive ratio=%g in row=%d.\n", pivot_row, pivot_col, min_ratio, pivot_row);
	
	return pivot_row;
}

// Given a column of identity matrix, find the row containing 1.
// return -1, if the column as not from an identity matrix.
 
__global__ void findBasisVariableKernel(double * deviceTable, int pitch, Element * indicesToBasisVariableArray, int numberOfRows, int numberOfColumns)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index>=numberOfColumns || index ==0)
		return;
	int xi = -1;
	double value;
	int flag = 0;
	
	for(int i=0;i<numberOfRows;i++)
	{
		double * row_i = (double *) ((char *) deviceTable + i*pitch);
		if (row_i[index]>0 && !equalsTo(row_i[index],0.) ) {
      			if (xi == -1)
        		{
					xi=i;   // found first +ve, save this row number.
					value = row_i[index];
				}
      			else
        		{
					flag = 1; // found second '1', not an identity matrix.
					break;
				}

    	} 
		else if (!equalsTo( row_i[index],0) ) {
			flag = 1;
      			break; // not an identity matrix column.
    		}
	}	
	
	if(flag==1)
		indicesToBasisVariableArray[index].index = -1;
	else
	{
		indicesToBasisVariableArray[index].index = xi;
		indicesToBasisVariableArray[index].value = value;
	}
	return;
}

void print_optimal_vector(Tableau *tab, char *message) {
	int j, xi;
	printf("%s at ", message);
	
	Element * indicesToBasisVariablesInDevice;
	Element * indicesToBasisVariablesInHost;
	
	double basisVariable;
	
	int size = sizeof(Element)*(tab->numberOfColumns);

	cudaMalloc( (void **) &indicesToBasisVariablesInDevice, size);
	indicesToBasisVariablesInHost = (Element *)malloc(size);

	//launch kernel to compute basis variable for each column
	findBasisVariableKernel<<<(tab->numberOfColumns + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, 
	THREADS_PER_BLOCK >>>(tab->table, tab->pitch, indicesToBasisVariablesInDevice, tab->numberOfRows, tab->numberOfColumns);

	cudaMemcpy(indicesToBasisVariablesInHost, indicesToBasisVariablesInDevice, size, cudaMemcpyDeviceToHost);
	
	int flag = 0;
	double variableValue = 0;
	double transformedVariableValue = 0;
	
  	for(j=1;j<tab->numberOfColumns;j++) { // for each column.
		
		xi = indicesToBasisVariablesInHost[j].index;
    	if (xi != -1)
		{
			cudaMemcpy(&basisVariable, (double *)((char *)tab->table + xi * tab->pitch), sizeof(double),cudaMemcpyDeviceToHost);
			//printf("x%d=%3.2lf, ", j, basisVariable/indicesToBasisVariablesInHost[j].value );
			transformedVariableValue = (basisVariable/indicesToBasisVariablesInHost[j].value );
		}
    	else
		{
			transformedVariableValue = 0;
			//printf("x%d=0, ", j);
		}
		variableValue += flag==0?transformedVariableValue:(-1*transformedVariableValue);
		flag ^= 1;
		if(!flag)
		{
			if(((j+1)/2)<tab->numberOfColumnsInOriginalMatrix)
				printf("x%d=%3.2lf, ", (j+1)/2,variableValue);
			variableValue = 0;
		}
  	}
  	printf("\n");
	cudaFree(indicesToBasisVariablesInDevice);
	free(indicesToBasisVariablesInHost);
} 

void simplex(Tableau *tab, int maxIteration, int verbosityLevel) {
	int loop=0;
	double optimalValue;
	double pValue;

	while( ++loop ) {
		int pivot_col, pivot_row = -1;

		pivot_col = find_pivot_column(tab, verbosityLevel);
		if( pivot_col < 0 ) {
			cudaMemcpy(&optimalValue, tab->table, sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(&pValue, tab->table + (tab->numberOfColumns) - 1, sizeof(double), cudaMemcpyDeviceToHost);
			printf("Found optimal value=A[0,0]=%3.2lf (no negatives in row 0).\n", optimalValue/pValue);
			print_optimal_vector(tab, "Optimal vector");
			break;
		}
		if(verbosityLevel)
			printf("Entering variable x%d to be made basic, so pivot_col=%d.\n", pivot_col, pivot_col);

		pivot_row = find_pivot_row(tab, pivot_col, verbosityLevel);
		
		if (pivot_row < 0) {
			//if(verbosityLevel)
				printf("unbounded (no pivot_row).\n");
			break;
		}
		if(verbosityLevel)
			printf("Leaving variable x%d, so pivot_row=%d\n", pivot_row, pivot_row);

		pivot_on(tab, pivot_row, pivot_col);
		if(verbosityLevel)
		{
			updateHostTable(tab->table,tab->pitch, tab->numberOfRows, tab->numberOfColumns);
			print_tableau(demoTab,"After pivoting", verbosityLevel);
			print_optimal_vector(tab, "Basic feasible solution");
		}

		if(loop > maxIteration) {
			printf("Too many iterations > %d.\n", loop);
			break;
		}
	}
	printf("number of iterations: %d\n", loop);
	cudaFree(tab->table);
}

int main(int argc, char *argv[]){
	Tableau * tab;
	tab = (Tableau *) malloc(sizeof(Tableau));
	tab->firstPivotRow = -1;
	
	int verbosityLevel = 0;
	int maxIteration = 0;
	
	struct timeval  tv1, tv2;

	
	/* here, do your time-consuming job */
	
	if (argc > 3) { // usage: cmd datafile i/n max_iteration_number
		if(strcasecmp(argv[2],"i")==0)
		{
			verbosityLevel = 1;
			demoTab = (Tab *) malloc(sizeof(Tab));	//allocate memory in RAM for display purpose
		}
		maxIteration = atoi(argv[3]);
		
		read_tableau(tab, argv[1], verbosityLevel);
	}
	else
	{
		printf("usage: cmd <datafile> <i/n> <max)iteration_number> \n");
		free(tab);
		exit(1);
	}
	
	print_tableau(demoTab,"Initial", verbosityLevel);
	gettimeofday(&tv1, NULL);
	/* stuff to do! */
	simplex(tab, maxIteration, verbosityLevel);
	gettimeofday(&tv2, NULL);

	printf ("Total time = %f seconds\n",
         (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
         (double) (tv2.tv_sec - tv1.tv_sec));
	
	
	free(tab);
	if(verbosityLevel)
		free(demoTab);
	return 0;
} 

