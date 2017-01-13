#include <stdio.h>
#ifndef NOSIGNAL
#include <signal.h>
#endif
#include <setjmp.h>

extern double fabs(), floor(), log(), pow(), sqrt();

#ifdef Single
#define FLOAT float
#define FABS(x) (float)fabs((double)(x))
#define FLOOR(x) (float)floor((double)(x))
#define LOG(x) (float)log((double)(x))
#define POW(x,y) (float)pow((double)(x),(double)(y))
#define SQRT(x) (float)sqrt((double)(x))
#else
#define FLOAT double
#define FABS(x) fabs(x)
#define FLOOR(x) floor(x)
#define LOG(x) log(x)
#define POW(x,y) pow(x,y)
#define SQRT(x) sqrt(x)
#endif

extern jmp_buf ovfl_buf;
typedef void (*Sig_type)();
extern Sig_type sigsave;
extern void sigfpe();

#define KEYBOARD 0

extern FLOAT Radix, BInvrse, RadixD2, BMinusU2;
extern FLOAT Sign(), Random();

/*Small floating point constants.*/
extern FLOAT Zero;
extern FLOAT Half;
extern FLOAT One;
extern FLOAT Two;
extern FLOAT Three;
extern FLOAT Four;
extern FLOAT Five;
extern FLOAT Eight;
extern FLOAT Nine;
extern FLOAT TwentySeven;
extern FLOAT ThirtyTwo;
extern FLOAT TwoForty;
extern FLOAT MinusOne;
extern FLOAT OneAndHalf;
/*Integer constants*/
extern int NoTrials;
#define False 0
#define True 1

/* Definitions for declared types
	Guard == (Yes, No);
	Rounding == (Chopped, Rounded, Other);
	Message == packed array [1..40] of char;
	Class == (Flaw, Defect, Serious, Failure);
	  */
#define Yes 1
#define No  0
#define Chopped 2
#define Rounded 1
#define Other   0
#define Flaw    3
#define Defect  2
#define Serious 1
#define Failure 0
typedef int Guard, Rounding, Class;
typedef char Message;

/* Declarations of Variables */
extern int Indx;
extern char ch[8];
extern FLOAT AInvrse, A1;
extern FLOAT C, CInvrse;
extern FLOAT D, FourD;
extern FLOAT E0, E1, Exp2, E3, MinSqEr;
extern FLOAT SqEr, MaxSqEr, E9;
extern FLOAT Third;
extern FLOAT F6, F9;
extern FLOAT H, HInvrse;
extern int I;
extern FLOAT StickyBit, J;
extern FLOAT MyZero;
extern FLOAT Precision;
extern FLOAT Q, Q9;
extern FLOAT R, Random9;
extern FLOAT T, Underflow, S;
extern FLOAT OneUlp, UfThold, U1, U2;
extern FLOAT V, V0, V9;
extern FLOAT W;
extern FLOAT X, X1, X2, X8, Random1;
extern FLOAT Y, Y1, Y2, Random2;
extern FLOAT Z, PseudoZero, Z1, Z2, Z9;
extern int ErrCnt[4];
extern int fpecount;
extern int Milestone;
extern int PageNo;
extern int M, N, N1;
extern Guard GMult, GDiv, GAddSub;
extern Rounding RMult, RDiv, RAddSub, RSqrt;
extern int Break, Done, NotMonot, Monot, Anomaly, IEEE,
		SqRWrng, UfNGrad;
