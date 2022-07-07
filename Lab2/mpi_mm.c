/******************************************************************************
 *FILE: mpi_mm.c
 *DESCRIPTION:  
 *  MPI Matrix Multiply - C Version
 *  In this code, the master task distributes a matrix multiply
 *  operation to numtasks-1 worker tasks.
 *  NOTE:  C and Fortran versions of this code differ because of the way
 *  arrays are stored/passed.  C arrays are row-major order but Fortran
 *  arrays are column-major order.
 *AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
 *  Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
 *LAST REVISED: 09/29/2021
 ******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <limits.h>
#define MASTER 0 /*taskid of first task */ 
#define FROM_MASTER 1 /*setting a message type */ 
#define FROM_WORKER 2 /*setting a message type */


//PART-1
double rmax = DBL_MIN; // DOUBLE MIN OR -999;
double smax = DBL_MIN;
double cmax = DBL_MIN;

double rmin = DBL_MAX; //09999 OR DOUBLE_MAX;
double smin = DBL_MAX;
double cmin = DBL_MAX;

double ravg = 0.00;
double savg = 0.00;
double cavg = 0.00;


//PART-2
double inimin = DBL_MAX;
double inimax = DBL_MIN;
double iniavg = 0.00;

double senrecmin = DBL_MAX;
double senrecmax = DBL_MIN;
double senrecavg = 0.00;

double t1;



int main(int argc, char *argv[])
{
	int sizeOfMatrix;
	if (argc == 2)
	{
		sizeOfMatrix = atoi(argv[1]);
	}
	else
	{
		printf("\n Please provide the size of the matrix");
		return 0;
	}

	int numtasks,                             /*number of tasks in partition */
	   taskid,                                /*a task identifier */
	   numworkers,                            /*number of worker tasks */
	   source,                                /*task id of message source */
	   dest,                                  /*task id of message destination */
	   mtype,                                 /*message type */
	   rows,                                  /*rows of matrix A sent to each worker */
	   averow, extra, offset,                 /*used to determine rows sent to each worker */
	   i, j, k, rc;                           /*misc */

	double a[sizeOfMatrix][sizeOfMatrix],  /*matrix A to be multiplied */
		b[sizeOfMatrix][sizeOfMatrix],      /*matrix B to be multiplied */
		c[sizeOfMatrix][sizeOfMatrix];      /*result matrix C */
	
   MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	if (numtasks < 2)
	{
		printf("Need at least two MPI tasks. Quitting...\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
		exit(1);
	}

	numworkers = numtasks - 1;

   double ini1, ini2 = 0.0;
   double senrec1, senrec2 = 0.0;
	/****************************master task ************************************/
	if (taskid == MASTER)
	{
		// INITIALIZATION PART FOR THE MASTER PROCESS STARTS HERE
      t1 = MPI_Wtime();
      ini1 = MPI_Wtime();
		printf("mpi_mm has started with %d tasks.\n", numtasks);
		printf("Initializing arrays...\n");
		for (i = 0; i < sizeOfMatrix; i++)
			for (j = 0; j < sizeOfMatrix; j++)
				a[i][j] = i + j;
		for (i = 0; i < sizeOfMatrix; i++)
			for (j = 0; j < sizeOfMatrix; j++)
				b[i][j] = i * j;
      ini2 = MPI_Wtime()-ini1;
		//INITIALIZATION PART FOR THE MASTER PROCESS ENDS HERE
      
		//SEND-RECEIVE PART FOR THE MASTER PROCESS STARTS HERE
      senrec1 = MPI_Wtime();
		/*Send matrix data to the worker tasks */
      
		averow = sizeOfMatrix / numworkers;
		extra = sizeOfMatrix % numworkers;
		offset = 0;
		mtype = FROM_MASTER;
		for (dest = 1; dest <= numworkers; dest++)
		{
			rows = (dest <= extra) ? averow + 1 : averow;
			printf("Sending %d rows to task %d offset=%d\n", rows, dest, offset);
			MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
			MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
			MPI_Send(&a[offset][0], rows *sizeOfMatrix, MPI_DOUBLE, dest, mtype,
				MPI_COMM_WORLD);
			MPI_Send(&b, sizeOfMatrix *sizeOfMatrix, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
			offset = offset + rows;
		}
      

		/*Receive results from worker tasks */
      
		mtype = FROM_WORKER;
		for (i = 1; i <= numworkers; i++)
		{
			source = i;
			MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&c[offset][0], rows *sizeOfMatrix, MPI_DOUBLE, source, mtype,
				MPI_COMM_WORLD, &status);
			printf("Received results from task %d\n", source);
		}
      

		//SEND-RECEIVE PART FOR THE MASTER PROCESS ENDS HERE
      senrec2 = MPI_Wtime() - senrec1;
		/*Print results - you can uncomment the following lines to print the result matrix */

		/* printf("******************************************************\n");
		printf("Result Matrix:\n");
		for (i=0; i < sizeOfMatrix; i++)
		{
		   printf("\n"); 
		   for (j=0; j < sizeOfMatrix; j++) 
		      printf("%6.2f   ", c[i][j]);
		}

		printf("\n******************************************************\n");
		printf ("Done.\n"); */


	}

   double rec1, rec2 = 0.0;
   double cal1, cal2 = 0.0;
   double sen1, sen2 = 0.0;
	/****************************worker task ************************************/
	if (taskid > MASTER)
	{
		//RECEIVING PART FOR WORKER PROCESS STARTS HERE
      // rec1 = MPI_Wtime();    //***** Do the time for the RECEIVING PART here
		mtype = FROM_MASTER;
		MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(&a, rows *sizeOfMatrix, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(&b, sizeOfMatrix *sizeOfMatrix, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
      // rec2 = MPI_Wtime()-rec1;    //***** Do the time for the RECEIVING PART here
		//RECEIVING PART FOR WORKER PROCESS ENDS HERE

		//CALCULATION PART FOR WORKER PROCESS STARTS HERE
      // cal1 = MPI_Wtime();//***** Do the time for the RECEIVING PART here
		for (k = 0; k < sizeOfMatrix; k++)
			for (i = 0; i < rows; i++)
			{
				c[i][k] = 0.0;
				for (j = 0; j < sizeOfMatrix; j++)
					c[i][k] = c[i][k] + a[i][j] *b[j][k];
			}
      // cal2 = MPI_Wtime()-cal1;//***** Do the time for the RECEIVING PART here
		//CALCULATION PART FOR WORKER PROCESS ENDS HERE

		//SENDING PART FOR WORKER PROCESS STARTS HERE
      // sen1 = MPI_Wtime();//***** Do the time for the SENDING PART here
		mtype = FROM_WORKER;
		MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
		MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
		MPI_Send(&c, rows *sizeOfMatrix, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      // sen2 = MPI_Wtime() - sen1;//***** Do the time for the RECEIVING PART here
		//SENDING PART FOR WORKER PROCESS ENDS HERE
	}



   //USE MPI_Reduce here to calculate the minimum, maximum and the average times for the worker processes.

   /* 
   //PART-1 
   MPI_Reduce(&rec2,&rmax, 1, MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
   MPI_Reduce(&rec2,&rmin, 1, MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
   MPI_Reduce(&rec2,&ravg, 1, MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

   MPI_Reduce(&cal2,&cmax, 1, MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
   MPI_Reduce(&cal2,&cmin, 1, MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
   MPI_Reduce(&cal2,&cavg, 1, MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

   MPI_Reduce(&sen2,&smax, 1, MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
   MPI_Reduce(&sen2,&smin, 1, MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
   MPI_Reduce(&sen2,&savg, 1, MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
   */

   
   //PART-2
   MPI_Reduce(&senrec2,&senrecmax, 1, MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
   MPI_Reduce(&senrec2,&senrecmin, 1, MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
   MPI_Reduce(&senrec2,&senrecavg, 1, MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
   
   MPI_Reduce(&ini2,&inimax, 1, MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
   MPI_Reduce(&ini2,&inimin, 1, MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
   MPI_Reduce(&ini2,&iniavg, 1, MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
   
	MPI_Barrier(MPI_COMM_WORLD);
   double t2 = MPI_Wtime() - t1;
   
	if (taskid == 0)
	{
      //PART-1
      // printf("\n\n\n****** MASTER ***********\n");
      // printf("RMAX TIME  :  %lf\n",rmax);
      // printf("RMIN TIME  :  %lf\n",rmin);
      // printf("RAVG TIME  :  %lf\n\n\n",ravg/(numtasks-1));

      // printf("SMAX TIME  :  %lf\n",smax);
      // printf("SMIN TIME  :  %lf\n",smin);
      // printf("SAVG TIME  :  %lf\n\n\n",savg/(numtasks-1));

      // printf("CMAX TIME  :  %lf\n",cmax);
      // printf("CMIN TIME  :  %lf\n",cmin);
      // printf("CAVG TIME  :  %lf\n\n\n",cavg/(numtasks-1)); 
      

      
      //PART-2
      printf("INIT-MAX TIME  :  %lf\n",inimax);
      printf("INIT-MIN TIME  :  %lf\n",inimin);
      printf("INIT-AVG TIME  :  %lf\n\n\n",iniavg/(numtasks-1)); 

      printf("SENREC-MAX TIME  :  %lf\n",senrecmax);
      printf("SENREC-MIN TIME  :  %lf\n",senrecmin);
      printf("SENREC-AVG TIME  :  %lf\n\n\n",senrecavg/(numtasks-1)); 

      printf("FULL-TIME for Master Thread to complete : %lf \n",t2);
      
   }

	MPI_Finalize();
}