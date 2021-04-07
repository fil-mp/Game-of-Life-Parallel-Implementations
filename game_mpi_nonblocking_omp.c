#define generations 1000

#define ALIVE 1
#define DEAD 0
#define TAG 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

void create_datatype(MPI_Datatype *derivedtype, int start1, int start2, int subsize1, int subsize2,int width,int height)
{
	 const int array_of_sizes[2] = {height + 2,  width+ 2};
	 const int array_of_subsizes[2] = {subsize1, subsize2};
	 const int array_of_starts[2] = {start1, start2};

	 MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_CHAR, derivedtype);
	 MPI_Type_commit(derivedtype);
}

void find_neighbours(MPI_Comm newcomm, int my_rank, int NPROWS, int NPCOLS, int *west, int *east, int *north, int *south, int *northwest, int *northeast, int *southwest, int *southeast)
{

	//int source, dest;
	int nw[2], ne[2], sw[2], se[2];
	int coords[2];

	/*Finding top/bottom neighbours*/
	MPI_Cart_shift(newcomm, 0, 1, north, south);

	/*Finding left/right neighbours*/
	MPI_Cart_shift(newcomm, 1, 1, west, east);
	MPI_Cart_coords(newcomm, my_rank, 2, coords);

	nw[0] = coords[0] - 1;
	ne[0] = coords[0] - 1;
	sw[0] = coords[0] + 1;
	se[0] = coords[0] + 1;

	nw[1] = coords[1] - 1;
	ne[1] = coords[1] + 1;
	sw[1] = coords[1] - 1;
	se[1] = coords[1] + 1;

	MPI_Cart_rank(newcomm, nw, northwest);
	MPI_Cart_rank(newcomm, ne, northeast);
	MPI_Cart_rank(newcomm, sw, southwest);
	MPI_Cart_rank(newcomm, se, southeast);
}

int main(int argc, char **argv)
{
	int size, i, j, psb, rank, dim_size[2], periods[2];
	int width = 320, height = 320;
	MPI_Comm newcomm;

	MPI_Init(&argc, &argv); //Initialization
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//old_comm = MPI_COMM_WORLD;
	int dims[2] = { 0,0 };
	int ndims = 2; // 2D matrix/grid
	int rows_columns = (int)sqrt(size);
	dim_size[0] = rows_columns; // number of rows
	dim_size[1] = rows_columns; // number of columns
	periods[0] = 1;				// rows periodic (each column forms a ring)
	periods[1] = 1;				// columns periodic (each row forms a ring)
	int reorder = 1;				// allows processes reordered for efficiency
	const int NPROWS = dims[0]; /* Number of 'block' rows */
	const int NPCOLS = dims[1]; /* Number of 'block' cols */

	MPI_Cart_create(MPI_COMM_WORLD, ndims, dim_size, periods, reorder, &newcomm);
	int newrank, coords[2];

	MPI_Comm_rank(newcomm, &newrank);
	MPI_Cart_coords(newcomm, newrank, ndims, coords);

	int width_local, height_local;

	// Calculate the local dimensions for the local subarrays
	height_local = height / rows_columns;
	width_local = width / rows_columns;

	// Allocate space in each instance for the local array(s)
	char **s_grid = malloc((width_local + 2) * sizeof(char *));
	char *b = malloc((width_local + 2) * (height_local + 2) * sizeof(char));

	for (int i = 0; i < (width_local + 2); i++)
		s_grid[i] = &b[i * (height_local + 2)];

	for (i = 1; i <= width_local; ++i)
		for (j = 1; j <= height_local; ++j)
			//s_grid[i][j] = (rand() % 2 == 0) ? DEAD : ALIVE;
	
			psb = rand() % 100 + 1;
			if (psb<=40)
				s_grid[i][j] = '1';
			else
				s_grid[i][j] = '0';
				
	// Allocate space for the new array which holds the next generation of the local grid
	char **newGrid = malloc((width_local + 2) * sizeof(char *));
	char *a = malloc((width_local + 2) * (height_local + 2) * sizeof(char));
	for (int i = 0; i < (width_local + 2); i++)
		newGrid[i] = &a[i * (height_local + 2)];

	/*16 requests , 16 statuses */
	MPI_Request array_of_srequests[8];
	MPI_Request array_of_rrequests[8];
	MPI_Status array_of_sstatuses[8];
	MPI_Status array_of_rstatuses[8];
	/*Create 4 datatypes for sending*/
	MPI_Datatype firstcolumn_send, firstrow_send, lastcolumn_send, lastrow_send;
	create_datatype(&firstcolumn_send, 1, 1, width_local, 1, width_local, height_local);
	create_datatype(&firstrow_send, 1, 1, 1, height_local, width_local, height_local);
	create_datatype(&lastcolumn_send, 1, height_local, width_local, 1, width_local, height_local);
	create_datatype(&lastrow_send, width_local, 1, 1, height_local, width_local, height_local);

	/*Create 4 datatypes for receiving*/
	MPI_Datatype firstcolumn_recv, firstrow_recv, lastcolumn_recv, lastrow_recv;
	create_datatype(&firstcolumn_recv, 1, 0, width_local, 1,width_local,height_local);
	create_datatype(&firstrow_recv, 0, 1, 1, height_local,width_local, height_local);
	create_datatype(&lastcolumn_recv, 1, height_local + 1, width_local, 1, width_local, height_local);
	create_datatype(&lastrow_recv, width_local + 1, 1, 1, height_local, width_local, height_local);

	int west, east, north, south, northwest, northeast, southwest, southeast;
	find_neighbours(newcomm, newrank, NPROWS, NPCOLS, &west, &east, &north, &south, &northwest, &northeast, &southwest, &southeast);

	

	MPI_Barrier(MPI_COMM_WORLD);
	double t_start = MPI_Wtime();
	int counter = 0;
	//int threadsnum = 64;
	for (int iter = 0; iter < generations; iter++)
	{

		MPI_Irecv(&(s_grid[0][0]), 1, firstcolumn_recv, west, TAG, newcomm, &array_of_rrequests[0]);
	    MPI_Irecv(&(s_grid[0][0]), 1, firstrow_recv, north, TAG, newcomm, &array_of_rrequests[1]);
	    MPI_Irecv(&(s_grid[0][0]), 1, lastcolumn_recv, east, TAG, newcomm, &array_of_rrequests[2]);
	    MPI_Irecv(&(s_grid[0][0]), 1, lastrow_recv, south, TAG, newcomm, &array_of_rrequests[3]);
	    MPI_Irecv(&(s_grid[0][0]), 1, MPI_CHAR, northwest, TAG, newcomm, &array_of_rrequests[4]);
	    MPI_Irecv(&(s_grid[0][height_local + 1]), 1, MPI_CHAR, northeast, TAG, newcomm, &array_of_rrequests[5]);
	    MPI_Irecv(&(s_grid[width_local + 1][height_local + 1]), 1, MPI_CHAR, southeast, TAG, newcomm, &array_of_rrequests[6]);
	    MPI_Irecv(&(s_grid[width_local + 1][0]), 1, MPI_CHAR, southwest, TAG, newcomm, &array_of_rrequests[7]);

        MPI_Isend(&(s_grid[0][0]), 1, firstcolumn_send, west, TAG, newcomm, &array_of_srequests[0]);
	    MPI_Isend(&(s_grid[0][0]), 1, firstrow_send, north, TAG, newcomm, &array_of_srequests[1]);
	    MPI_Isend(&(s_grid[0][0]), 1, lastcolumn_send, east, TAG, newcomm, &array_of_srequests[2]);
	    MPI_Isend(&(s_grid[0][0]), 1, lastrow_send, south, TAG, newcomm, &array_of_srequests[3]);
	    MPI_Isend(&(s_grid[1][1]), 1, MPI_CHAR, northwest, TAG, newcomm, &array_of_srequests[4]);
	    MPI_Isend(&(s_grid[1][height_local]), 1, MPI_CHAR, northeast, TAG, newcomm, &array_of_srequests[5]);
	    MPI_Isend(&(s_grid[width_local][height_local]), 1, MPI_CHAR, southeast, TAG, newcomm, &array_of_srequests[6]);
	    MPI_Isend(&(s_grid[width_local][1]), 1, MPI_CHAR, southwest, TAG, newcomm, &array_of_srequests[7]);

		int empty_or_diff = 0;
		int diff = 0;
		
		int i, j;
		int sum=0;
		
 #pragma omp parallel for num_threads(threadsnum) shared(s_grid, newGrid, height_local, width_local) schedule(static, 1) private(i,j) reduction(+:sum) collapse(2)
		for ( i = 2; i <= height_local - 1; i++)
		{
			for ( j = 2; j <= width_local - 1; j++)
			{
				sum = 0;

				sum += s_grid[i + 1][j]; //north
				sum += s_grid[i - 1][j]; //south
				sum += s_grid[i][j + 1];//east
				sum += s_grid[i][j - 1]; //west
				sum += s_grid[i + 1][j + 1];//northeast
				sum += s_grid[i - 1][j - 1]; //northwest
				sum += s_grid[i - 1][j + 1];//southeast
				sum += s_grid[i + 1][j - 1];//southwest

				if (s_grid[i][j] == ALIVE && sum < 2) {

					newGrid[i][j] = DEAD;
					diff++;
				}
				else if (s_grid[i][j] == ALIVE && (sum == 2 || sum == 3))
				{
					newGrid[i][j] = ALIVE;
				}
				else if (s_grid[i][j] == ALIVE && sum > 3)
				{
					newGrid[i][j] = DEAD;
					diff++;
				}

				else if (s_grid[i][j] == DEAD && sum == 3) {
					{
						newGrid[i][j] = ALIVE;
						diff++;
					}
				}
				else {
					newGrid[i][j] = s_grid[i][j];
				}
			}
		}

		MPI_Waitall(8, array_of_rrequests, array_of_rstatuses);
		
#pragma omp parallel for num_threads(threadsnum) shared(s_grid, newGrid, height_local, width_local) schedule(static, 1) private(i,j) reduction(+:sum) collapse(2)	
		for (i = 1; i <= width_local; i += width_local - 1)
		{
			for ( j = 1; j <= height_local; j++)
			{
				sum = 0;

				sum += s_grid[j + 1][i];	 //north
				sum += s_grid[j - 1][i];	 //south
				sum += s_grid[j][i + 1];	 //east
				sum += s_grid[j][i - 1];	 //west
				sum += s_grid[j + 1][i + 1]; //northeast
				sum += s_grid[j - 1][i - 1]; //northwest
				sum += s_grid[j - 1][i + 1]; //southeast
				sum += s_grid[j + 1][i - 1]; //southwest

				if (s_grid[j][i] == ALIVE && sum < 2)
				{

					newGrid[j][i] = DEAD;
					diff++;
				}
				else if (s_grid[j][i] == ALIVE && (sum == 2 || sum == 3))
				{
					newGrid[j][i] = ALIVE;
				}
				else if (s_grid[j][i] == ALIVE && sum > 3)
				{
					newGrid[j][i] = DEAD;
					diff++;
				}

				else if (s_grid[j][i] == DEAD && sum == 3) {
					newGrid[j][i] = ALIVE;
					diff++;

				}
				else {
					newGrid[j][i] = s_grid[j][i];
				}
			}
		}
		
#pragma omp parallel for num_threads(threadsnum) shared(s_grid, newGrid, height_local, width_local) schedule(static, 1) private(i,j) reduction(+:sum) collapse(2)	
		for ( i = 1; i <= height_local; i += height_local - 1)
		{
			for ( j = 1; j <= width_local; j++)
			{
				sum = 0;

				sum += s_grid[i + 1][j];	 //north
				sum += s_grid[i - 1][j];	 //south
				sum += s_grid[i][j + 1];	 //east
				sum += s_grid[i][j - 1];	 //west
				sum += s_grid[i + 1][j + 1]; //northeast
				sum += s_grid[i - 1][j - 1]; //northwest
				sum += s_grid[i - 1][j + 1]; //southeast
				sum += s_grid[i + 1][j - 1]; //southwest

				if (s_grid[i][j] == ALIVE && sum < 2)
				{
					newGrid[i][j] = DEAD;
					diff++;
				}
				else if (s_grid[i][j] == ALIVE && (sum == 2 || sum == 3))
				{
					newGrid[i][j] = ALIVE;
				}
				else if (s_grid[i][j] == ALIVE && sum > 3)
				{
					newGrid[i][j] = DEAD;
					diff++;
				}

				else if (s_grid[i][j] == DEAD && sum == 3) {
					newGrid[i][j] = ALIVE;
					diff++;

				}
				else {
					newGrid[i][j] = s_grid[i][j];
				}
			}
		}

		counter++;
		if (counter == 10) {
			MPI_Allreduce(&diff, &empty_or_diff, 1, MPI_INT, MPI_SUM, newcomm);
			if (newrank == 0) {
				printf("diffs made : %d\n", empty_or_diff);
				if (empty_or_diff == 0) {
					printf("Nothing different on this iteration: %d\n",iter+1);
				}
			}
			if (empty_or_diff == 0) {
				break;
			}
			counter = 0;
		}

		char **temp;
		temp = s_grid;
		s_grid = newGrid;
		newGrid = temp;

	MPI_Waitall(8, array_of_srequests, array_of_sstatuses);
}

	double t_finish = MPI_Wtime();
	double time_elapsed;
	time_elapsed = t_finish - t_start;
	printf("(%2d proc)Time elapsed for %5d loops:%.4f\n", newrank, generations, time_elapsed);
	MPI_Type_free(&firstcolumn_send);
	MPI_Type_free(&firstrow_send);
	MPI_Type_free(&lastcolumn_send);
	MPI_Type_free(&lastrow_send);

	MPI_Type_free(&firstcolumn_recv);
	MPI_Type_free(&firstrow_recv);
	MPI_Type_free(&lastcolumn_recv);
	MPI_Type_free(&lastrow_recv);

	free(newGrid[0]);
	free(newGrid);
	MPI_Finalize();
	return 0;
}