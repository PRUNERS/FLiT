#undef V9
/*  A C/C++ version of Kahan's Floating Point Test "Paranoia"

  Added to FLiT with minor modifications by

      Michael Bentley, University of Utah, Jan. 2017

  Taken from

      Thos Sumner, UCSF, Feb. 1985
      David Gay, BTL, Jan. 1986

  This is a rewrite from the Pascal version by

      B. A. Wichmann, 18 Jan. 1985

  (and does NOT exhibit good C programming style).

(C) Apr 19 1983 in BASIC version by:
  Professor W. M. Kahan,
  567 Evans Hall
  Electrical Engineering & Computer Science Dept.
  University of California
  Berkeley, California 94720
  USA

converted to Pascal by:
  B. A. Wichmann
  National Physical Laboratory
  Teddington Middx
  TW11 OLW
  UK

converted to C by:

  David M. Gay and Thos Sumner
  AT&T Bell Labs Computer Center, Rm. U-76
  600 Mountain Avenue University of California
  Murray Hill, NJ 07974 San Francisco, CA 94143
  USA USA

converted for K&R, Standard C, and Standard C++ compilation [with loss
of support for the split script below] by:

   Nelson H. F. Beebe
   Center for Scientific Computing
   University of Utah
   Department of Mathematics, 322 INSCC
   155 S 1400 E RM 233
   Salt Lake City, UT 84112-0090
   USA
   Email: beebe@math.utah.edu, beebe@acm.org, beebe@computer.org, beebe@ieee.org (Internet)
   WWW URL: http://www.math.utah.edu/~beebe
   Telephone: +1 801 581 5254
   FAX: +1 801 585 1640, +1 801 581 4148

with simultaneous corrections to the Pascal source (reflected
in the Pascal source available over netlib).
[A couple of bug fixes from dgh = sun!dhough incorporated 31 July 1986.]

Reports of results on various systems from all the versions
of Paranoia are being collected by Richard Karpinski at the
same address as Thos Sumner.  This includes sample outputs,
bug reports, and criticisms.

You may copy this program freely if you acknowledge its source.
Comments on the Pascal version to NPL, please.


The C version catches signals from floating-point exceptions.
If signal(SIGFPE,...) is unavailable in your environment, you may
#define NOSIGNAL to comment out the invocations of signal.

This source file is too big for some C compilers, but may be split
into pieces.  Comments containing "SPLIT" suggest convenient places
for this splitting.  At the end of these comments is an "ed script"
(for the UNIX(tm) editor ed) that will do this splitting.

By #defining Single when you compile this source, you may obtain
a single-precision C version of Paranoia.


The following is from the introductory commentary from Wichmann's work:

The BASIC program of Kahan is written in Microsoft BASIC using many
facilities which have no exact analogy in Pascal.  The Pascal
version below cannot therefore be exactly the same.  Rather than be
a minimal transcription of the BASIC program, the Pascal coding
follows the conventional style of block-structured languages.  Hence
the Pascal version could be useful in producing versions in other
structured languages.

Rather than use identifiers of minimal length (which therefore have
little mnemonic significance), the Pascal version uses meaningful
identifiers as follows [Note: A few changes have been made for C]:


BASIC   C               BASIC   C               BASIC   C

   A                       J                       S    StickyBit
   a1   AInverse           J0   NoErrors           T
   B    radix                    [Failure]         T0   Underflow
   B1   BInverse           J1   NoErrors           T2   thirtyTwo
   B2   radixD2                  [SeriousDefect]   T5   oneAndHalf
   B9   bMinusU2           J2   NoErrors           T7   twentySeven
   C                             [Defect]          T8   twoForty
   C1   CInverse           J3   NoErrors           U    OneUlp
   D                             [Flaw]            U0   UnderflowThreshold
   D4   FourD              K    PageNo             U1
   E0                      L    Milestone          U2
   E1                      M                       V
   E2   Exp2               N                       V0
   E3                      N1                      V8
   E5   MinSqEr            O    zero               V9
   E6   SqEr               O1   one                W
   E7   MaxSqEr            O2   two                X
   E8                      O3   three              X1
   E9                      O4   four               X8
   F1   minusOne           O5   five               X9   Random1
   F2   half               O8   eight              Y
   F3   Third              O9   nine               Y1
   F6                      P    Precision          Y2
   F9                      Q                       Y9   Random2
   G1   GMult              Q8                      Z
   G2   GDiv               Q9                      Z0   PseudoZero
   G3   GAddSub            R                       Z1
   H                       R1   RMult              Z2
   H1   HInverse           R2   RDiv               Z9
   I                       R3   RAddSub
   IO   noTrials           R4   RSqrt
   I3   IEEE               R9   Random9

   SqRWrng

All the variables in BASIC are true variables and in consequence,
the program is more difficult to follow since the "constants" must
be determined (the glossary is very helpful).  The Pascal version
uses Real constants, but checks are added to ensure that the values
are correctly converted by the compiler.

The major textual change to the Pascal version apart from the
identifiersis that named procedures are used, inserting parameters
wherehelpful.  New procedures are also introduced.  The
correspondence is as follows:


BASIC       Pascal
lines

  90- 140   pause
 170- 250   instructions
 380- 460   heading
 480- 670   characteristics
 690- 870   history
2940-2950   random
3710-3740   newD
4040-4080   DoesYequalX
4090-4110   printIfNPositive
4640-4850   TestPartialUnderflow

*/

#include "testBase.hpp"
#include "QFPHelpers.hpp"
#include "CUHelpers.hpp"

#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <csetjmp>
#include <cmath>

#define KEYBOARD 0
#define False    0
#define True     1
#define Yes      1
#define No       0
#define Chopped  2
#define Rounded  1
#define Other    0
#define Flaw     3
#define Defect   2
#define Serious  1
#define Failure  0

extern "C" void sigfpe(int i);
extern "C" typedef void (*SigType)(int);

typedef int Guard, Rounding, Class;
typedef char Message;

template <typename F>
class Paranoia : public QFPTest::TestBase<F> {
public:
  Paranoia(std::string id) : QFPTest::TestBase<F>(std::move(id)) {}

  virtual size_t getInputsPerRun() { return 1; }
  virtual QFPTest::TestInput<F> getDefaultInput()
  { QFPTest::TestInput<F> ti; ti.vals = { 1.0 }; return ti; }

protected:
  virtual QFPTest::ResultType::mapped_type run_impl(const QFPTest::TestInput<F>& ti);

  void   badCond(int K_, const char *T_);
  void   characteristics(void);
  void   heading(void);
  void   history(void);
  void   instructions(void);
  void   isYEqX(void);
  void   newD(void);
  void   pause(void);
  void   printIfNPositive(void);
  F      random(void);
  void   sr3750(void);
  void   sr3980(void);
  F      sign(F X_);
  void   sqXMinX(int ErrKind_);
  void   tstCond(int K_, int Valid_, const char *T_);
  void   tstPtUf(void);
  void   msglist(const char **s_);
  void   notify(const char *s_);
  F      pow(F x_, F y_);

protected:
  using QFPTest::TestBase<F>::id;


  F radix, bInverse, radixD2, bMinusU2;

  /*Small floating point constants.*/
  F zero = 0.0;
  F half = 0.5;
  F one = 1.0;
  F two = 2.0;
  F three = 3.0;
  F four = 4.0;
  F five = 5.0;
  F eight = 8.0;
  F nine = 9.0;
  F twentySeven = 27.0;
  F thirtyTwo = 32.0;
  F twoForty = 240.0;
  F minusOne = -1.0;
  F oneAndHalf = 1.5;

  /*Integer constants*/
  int noTrials = 20; /*Number of tests for commutativity. */

  /* Declarations of Variables */
  int indx;
  char ch[8];
  F aInverse, a1;
  F C, CInvrse;
  F D, FourD;
  F E0, E1, Exp2, E3, MinSqEr;
  F SqEr, MaxSqEr, E9;
  F Third;
  F F6, F9;
  F H, HInvrse;
  int I;
  F StickyBit, J;
  F MyZero;
  F Precision;
  F Q, Q9;
  F R, Random9;
  F T, Underflow, S;
  F OneUlp, UfThold, U1, U2;
  F V, V0, V9;
  F W;
  F X, X1, X2, X8, Random1;
  F Y, Y1, Y2, Random2;
  F Z, PseudoZero, Z1, Z2, Z9;
  int ErrCnt[4];
  int Milestone;
  int PageNo;
  int M, N, N1;
  Guard GMult, GDiv, GAddSub;
  Rounding RMult, RDiv, RAddSub, RSqrt;
  int Break, Done, NotMonot, Monot, Anomaly, IEEE,
      SqRWrng, UfNGrad;
};
REGISTER_TYPE(Paranoia)

namespace {
  int fpecount = 0;
  jmp_buf ovfl_buf;
  SigType sigsave = nullptr;
}

/* floating point exception receiver */
void sigfpe(int i)
{
  Q_UNUSED(i);
  fpecount++;
  printf("\n* * * FLOATING-POINT ERROR * * *\n");
  (void)fflush(stdout);
  if (sigsave) {
    (void)signal(SIGFPE, sigsave);
    sigsave = nullptr;
    longjmp(ovfl_buf, 1);
  }
  abort();
}

template <typename F>
QFPTest::ResultType::mapped_type Paranoia<F>::run_impl(const QFPTest::TestInput<F>& ti)
{
  /* First two assignments use integer right-hand sides. */
  zero = 0;
  one = 1;
  two = one + one;
  three = two + one;
  four = three + one;
  five = four + one;
  eight = four + four;
  nine = three * three;
  twentySeven = nine * three;
  thirtyTwo = four * eight;
  twoForty = four * five * three * four;
  minusOne = -one;
  half = one / two;
  oneAndHalf = one + half;
  ErrCnt[Failure] = 0;
  ErrCnt[Serious] = 0;
  ErrCnt[Defect] = 0;
  ErrCnt[Flaw] = 0;
  PageNo = 1;
  /*=============================================*/
  Milestone = 0;
  /*=============================================*/
  (void)signal(SIGFPE, sigfpe);
  instructions();
  pause();
  heading();
  pause();
  characteristics();
  pause();
  history();
  pause();
  /*=============================================*/
  Milestone = 7;
  /*=============================================*/
  printf("Program is now RUNNING tests on small integers:\n");

  tstCond (Failure, (zero + zero == zero) && (one - one == zero)
       && (one > zero) && (one + one == two),
      "0+0 != 0, 1-1 != 0, 1 <= 0, or 1+1 != 2");
  Z = - zero;
  if (Z != 0.0) {
    ErrCnt[Failure] = ErrCnt[Failure] + 1;
    printf("Comparison alleges that -0.0 is Non-zero!\n");
    U1 = 0.001;
    radix = 1;
    tstPtUf();
    }
  tstCond (Failure, (three == two + one) && (four == three + one)
       && (four + two * (- two) == zero)
       && (four - three - one == zero),
       "3 != 2+1, 4 != 3+1, 4+2*(-2) != 0, or 4-3-1 != 0");
  tstCond (Failure, (minusOne == (0 - one))
       && (minusOne + one == zero ) && (one + minusOne == zero)
       && (minusOne + std::abs(one) == zero)
       && (minusOne + minusOne * minusOne == zero),
       "-1+1 != 0, (-1)+abs(1) != 0, or -1+(-1)*(-1) != 0");
  tstCond (Failure, half + minusOne + half == zero,
      "1/2 + (-1) + 1/2 != 0");
  /*=============================================*/
  Milestone = 10;
  /*=============================================*/
  tstCond (Failure, (nine == three * three)
       && (twentySeven == nine * three) && (eight == four + four)
       && (thirtyTwo == eight * four)
       && (thirtyTwo - twentySeven - four - one == zero),
       "9 != 3*3, 27 != 9*3, 32 != 8*4, or 32-27-4-1 != 0");
  tstCond (Failure, (five == four + one) &&
      (twoForty == four * five * three * four)
       && (twoForty / three - four * four * five == zero)
       && ( twoForty / four - five * three * four == zero)
       && ( twoForty / five - four * three * four == zero),
      "5 != 4+1, 240/3 != 80, 240/4 != 60, or 240/5 != 48");
  if (ErrCnt[Failure] == 0) {
    printf("-1, 0, 1/2, 1, 2, 3, 4, 5, 9, 27, 32 & 240 are O.K.\n");
    printf("\n");
    }
  printf("Searching for radix and Precision.\n");
  W = one;
  do  {
    W = W + W;
    Y = W + one;
    Z = Y - W;
    Y = Z - one;
    } while (minusOne + std::abs(Y) < zero);
  /*.. now W is just big enough that |((W+1)-W)-1| >= 1 ...*/
  Precision = zero;
  Y = one;
  do  {
    radix = W + Y;
    Y = Y + Y;
    radix = radix - W;
    } while ( radix == zero);
  if (radix < two) radix = one;
  printf("radix = %f .\n", radix);
  if (radix != 1) {
    W = one;
    do  {
      Precision = Precision + one;
      W = W * radix;
      Y = W + one;
      } while ((Y - W) == one);
    }
  /*... now W == radix^Precision is barely too big to satisfy (W+1)-W == 1
                                    ...*/
  U1 = one / W;
  U2 = radix * U1;
  printf("Closest relative separation found is U1 = %.7e .\n\n", U1);
  printf("Recalculating radix and precision\n ");

  /*save old values*/
  E0 = radix;
  E1 = U1;
  E9 = U2;
  E3 = Precision;

  X = four / three;
  Third = X - one;
  F6 = half - Third;
  X = F6 + F6;
  X = std::abs(X - Third);
  if (X < U2) X = U2;

  /*... now X = (unknown no.) ulps of 1+...*/
  do  {
    U2 = X;
    Y = half * U2 + thirtyTwo * U2 * U2;
    Y = one + Y;
    X = Y - one;
    } while ( ! ((U2 <= X) || (X <= zero)));

  /*... now U2 == 1 ulp of 1 + ... */
  X = two / three;
  F6 = X - half;
  Third = F6 + F6;
  X = Third - half;
  X = std::abs(X + F6);
  if (X < U1) X = U1;

  /*... now  X == (unknown no.) ulps of 1 -... */
  do  {
    U1 = X;
    Y = half * U1 + thirtyTwo * U1 * U1;
    Y = half - Y;
    X = half + Y;
    Y = half - X;
    X = half + Y;
    } while ( ! ((U1 <= X) || (X <= zero)));
  /*... now U1 == 1 ulp of 1 - ... */
  if (U1 == E1) printf("confirms closest relative separation U1 .\n");
  else printf("gets better closest relative separation U1 = %.7e .\n", U1);
  W = one / U1;
  F9 = (half - U1) + half;
  radix = std::floor(0.01 + U2 / U1);
  if (radix == E0) printf("radix confirmed.\n");
  else printf("MYSTERY: recalculated radix = %.7e .\n", radix);
  tstCond (Defect, radix <= eight + eight,
       "radix is too big: roundoff problems");
  tstCond (Flaw, (radix == two) || (radix == 10)
       || (radix == one), "radix is not as good as 2 or 10");
  /*=============================================*/
  Milestone = 20;
  /*=============================================*/
  tstCond (Failure, F9 - half < half,
       "(1-U1)-1/2 < 1/2 is FALSE, prog. fails?");
  X = F9;
  I = 1;
  Y = X - half;
  Z = Y - half;
  tstCond (Failure, (X != one)
       || (Z == zero), "Comparison is fuzzy,X=1 but X-1/2-1/2 != 0");
  X = one + U2;
  I = 0;
  /*=============================================*/
  Milestone = 25;
  /*=============================================*/
  /*... bMinusU2 = nextafter(radix, 0) */
  bMinusU2 = radix - one;
  bMinusU2 = (bMinusU2 - U2) + one;
  /* Purify Integers */
  if (radix != one)  {
    X = - twoForty * std::log(U1) / std::log(radix);
    Y = std::floor(half + X);
    if (std::abs(X - Y) * four < one) X = Y;
    Precision = X / twoForty;
    Y = std::floor(half + Precision);
    if (std::abs(Precision - Y) * twoForty < half) Precision = Y;
    }
  if ((Precision != std::floor(Precision)) || (radix == one)) {
    printf("Precision cannot be characterized by an Integer number\n");
    printf("of significant digits but, by itself, this is a minor flaw.\n");
    }
  if (radix == one)
    printf("logarithmic encoding has precision characterized solely by U1.\n");
  else printf("The number of significant digits of the radix is %f .\n",
      Precision);
  tstCond (Serious, U2 * nine * nine * twoForty < one,
       "Precision worse than 5 decimal figures  ");
  /*=============================================*/
  Milestone = 30;
  /*=============================================*/
  /* Test for extra-precise subepressions */
  X = std::abs(((four / three - one) - one / four) * three - one / four);
  do  {
    Z2 = X;
    X = (one + (half * Z2 + thirtyTwo * Z2 * Z2)) - one;
    } while ( ! ((Z2 <= X) || (X <= zero)));
  X = Y = Z = std::abs((three / four - two / three) * three - one / four);
  do  {
    Z1 = Z;
    Z = (one / two - ((one / two - (half * Z1 + thirtyTwo * Z1 * Z1))
      + one / two)) + one / two;
    } while ( ! ((Z1 <= Z) || (Z <= zero)));
  do  {
    do  {
      Y1 = Y;
      Y = (half - ((half - (half * Y1 + thirtyTwo * Y1 * Y1)) + half
        )) + half;
      } while ( ! ((Y1 <= Y) || (Y <= zero)));
    X1 = X;
    X = ((half * X1 + thirtyTwo * X1 * X1) - F9) + F9;
    } while ( ! ((X1 <= X) || (X <= zero)));
  if ((X1 != Y1) || (X1 != Z1)) {
    badCond(Serious, "Disagreements among the values X1, Y1, Z1,\n");
    printf("respectively  %.7e,  %.7e,  %.7e,\n", X1, Y1, Z1);
    printf("are symptoms of inconsistencies introduced\n");
    printf("by extra-precise evaluation of arithmetic subexpressions.\n");
    notify("Possibly some part of this");
    if ((X1 == U1) || (Y1 == U1) || (Z1 == U1))  printf(
      "That feature is not tested further by this program.\n") ;
    }
  else  {
    if ((Z1 != U1) || (Z2 != U2)) {
      if ((Z1 >= U1) || (Z2 >= U2)) {
        badCond(Failure, "");
        notify("Precision");
        printf("\tU1 = %.7e, Z1 - U1 = %.7e\n",U1,Z1-U1);
        printf("\tU2 = %.7e, Z2 - U2 = %.7e\n",U2,Z2-U2);
        }
      else {
        if ((Z1 <= zero) || (Z2 <= zero)) {
          printf("Because of unusual radix = %f", radix);
          printf(", or exact rational arithmetic a result\n");
          printf("Z1 = %.7e, or Z2 = %.7e ", Z1, Z2);
          notify("of an\nextra-precision");
          }
        if (Z1 != Z2 || Z1 > zero) {
          X = Z1 / U1;
          Y = Z2 / U2;
          if (Y > X) X = Y;
          Q = - std::log(X);
          printf("Some subexpressions appear to be calculated extra\n");
          printf("precisely with about %g extra B-digits, i.e.\n",
            (Q / std::log(radix)));
          printf("roughly %g extra significant decimals.\n",
            Q / std::log(10.));
          }
        printf("That feature is not tested further by this program.\n");
        }
      }
    }
  pause();
  /*=============================================*/
  Milestone = 35;
  /*=============================================*/
  if (radix >= two) {
    X = W / (radix * radix);
    Y = X + one;
    Z = Y - X;
    T = Z + U2;
    X = T - Z;
    tstCond (Failure, X == U2,
      "Subtraction is not normalized X=Y,X+Z != Y+Z!");
    if (X == U2) printf(
      "Subtraction appears to be normalized, as it should be.");
    }
  printf("\nChecking for guard digit in *, /, and -.\n");
  Y = F9 * one;
  Z = one * F9;
  X = F9 - half;
  Y = (Y - half) - X;
  Z = (Z - half) - X;
  X = one + U2;
  T = X * radix;
  R = radix * X;
  X = T - radix;
  X = X - radix * U2;
  T = R - radix;
  T = T - radix * U2;
  X = X * (radix - one);
  T = T * (radix - one);
  if ((X == zero) && (Y == zero) && (Z == zero) && (T == zero)) GMult = Yes;
  else {
    GMult = No;
    tstCond (Serious, False,
      "* lacks a Guard Digit, so 1*X != X");
    }
  Z = radix * U2;
  X = one + Z;
  Y = std::abs((X + Z) - X * X) - U2;
  X = one - U2;
  Z = std::abs((X - U2) - X * X) - U1;
  tstCond (Failure, (Y <= zero)
       && (Z <= zero), "* gets too many final digits wrong.\n");
  Y = one - U2;
  X = one + U2;
  Z = one / Y;
  Y = Z - X;
  X = one / three;
  Z = three / nine;
  X = X - Z;
  T = nine / twentySeven;
  Z = Z - T;
  tstCond(Defect, X == zero && Y == zero && Z == zero,
    "Division lacks a Guard Digit, so error can exceed 1 ulp\nor  1/3  and  3/9  and  9/27 may disagree");
  Y = F9 / one;
  X = F9 - half;
  Y = (Y - half) - X;
  X = one + U2;
  T = X / one;
  X = T - X;
  if ((X == zero) && (Y == zero) && (Z == zero)) GDiv = Yes;
  else {
    GDiv = No;
    tstCond (Serious, False,
      "Division lacks a Guard Digit, so X/1 != X");
    }
  X = one / (one + U2);
  Y = X - half - half;
  tstCond (Serious, Y < zero,
       "Computed value of 1/1.000..1 >= 1");
  X = one - U2;
  Y = one + radix * U2;
  Z = X * radix;
  T = Y * radix;
  R = Z / radix;
  StickyBit = T / radix;
  X = R - X;
  Y = StickyBit - Y;
  tstCond (Failure, X == zero && Y == zero,
      "* and/or / gets too many last digits wrong");
  Y = one - U1;
  X = one - F9;
  Y = one - Y;
  T = radix - U2;
  Z = radix - bMinusU2;
  T = radix - T;
  if ((X == U1) && (Y == U1) && (Z == U2) && (T == U2)) GAddSub = Yes;
  else {
    GAddSub = No;
    tstCond (Serious, False,
      "- lacks Guard Digit, so cancellation is obscured");
    }
  if (F9 != one && F9 - one >= zero) {
    badCond(Serious, "comparison alleges  (1-U1) < 1  although\n");
    printf("  subtraction yields  (1-U1) - 1 = 0 , thereby vitiating\n");
    printf("  such precautions against division by zero as\n");
    printf("  ...  if (X == 1.0) {.....} else {.../(X-1.0)...}\n");
    }
  if (GMult == Yes && GDiv == Yes && GAddSub == Yes) printf(
    "     *, /, and - appear to have guard digits, as they should.\n");
  /*=============================================*/
  Milestone = 40;
  /*=============================================*/
  pause();
  printf("Checking rounding on multiply, divide and add/subtract.\n");
  RMult = Other;
  RDiv = Other;
  RAddSub = Other;
  radixD2 = radix / two;
  a1 = two;
  Done = False;
  do  {
    aInverse = radix;
    do  {
      X = aInverse;
      aInverse = aInverse / a1;
      } while ( ! (std::floor(aInverse) != aInverse));
    Done = (X == one) || (a1 > three);
    if (! Done) a1 = nine + one;
    } while ( ! (Done));
  if (X == one) a1 = radix;
  aInverse = one / a1;
  X = a1;
  Y = aInverse;
  Done = False;
  do  {
    Z = X * Y - half;
    tstCond (Failure, Z == half,
      "X * (1/X) differs from 1");
    Done = X == radix;
    X = radix;
    Y = one / X;
    } while ( ! (Done));
  Y2 = one + U2;
  Y1 = one - U2;
  X = oneAndHalf - U2;
  Y = oneAndHalf + U2;
  Z = (X - U2) * Y2;
  T = Y * Y1;
  Z = Z - X;
  T = T - X;
  X = X * Y2;
  Y = (Y + U2) * Y1;
  X = X - oneAndHalf;
  Y = Y - oneAndHalf;
  if ((X == zero) && (Y == zero) && (Z == zero) && (T <= zero)) {
    X = (oneAndHalf + U2) * Y2;
    Y = oneAndHalf - U2 - U2;
    Z = oneAndHalf + U2 + U2;
    T = (oneAndHalf - U2) * Y1;
    X = X - (Z + U2);
    StickyBit = Y * Y1;
    S = Z * Y2;
    T = T - Y;
    Y = (U2 - Y) + StickyBit;
    Z = S - (Z + U2 + U2);
    StickyBit = (Y2 + U2) * Y1;
    Y1 = Y2 * Y1;
    StickyBit = StickyBit - Y2;
    Y1 = Y1 - half;
    if ((X == zero) && (Y == zero) && (Z == zero) && (T == zero)
      && ( StickyBit == zero) && (Y1 == half)) {
      RMult = Rounded;
      printf("Multiplication appears to round correctly.\n");
      }
    else  if ((X + U2 == zero) && (Y < zero) && (Z + U2 == zero)
        && (T < zero) && (StickyBit + U2 == zero)
        && (Y1 < half)) {
        RMult = Chopped;
        printf("Multiplication appears to chop.\n");
        }
      else printf("* is neither chopped nor correctly rounded.\n");
    if ((RMult == Rounded) && (GMult == No)) notify("Multiplication");
    }
  else printf("* is neither chopped nor correctly rounded.\n");
  /*=============================================*/
  Milestone = 45;
  /*=============================================*/
  Y2 = one + U2;
  Y1 = one - U2;
  Z = oneAndHalf + U2 + U2;
  X = Z / Y2;
  T = oneAndHalf - U2 - U2;
  Y = (T - U2) / Y1;
  Z = (Z + U2) / Y2;
  X = X - oneAndHalf;
  Y = Y - T;
  T = T / Y1;
  Z = Z - (oneAndHalf + U2);
  T = (U2 - oneAndHalf) + T;
  if (! ((X > zero) || (Y > zero) || (Z > zero) || (T > zero))) {
    X = oneAndHalf / Y2;
    Y = oneAndHalf - U2;
    Z = oneAndHalf + U2;
    X = X - Y;
    T = oneAndHalf / Y1;
    Y = Y / Y1;
    T = T - (Z + U2);
    Y = Y - Z;
    Z = Z / Y2;
    Y1 = (Y2 + U2) / Y2;
    Z = Z - oneAndHalf;
    Y2 = Y1 - Y2;
    Y1 = (F9 - U1) / F9;
    if ((X == zero) && (Y == zero) && (Z == zero) && (T == zero)
      && (Y2 == zero) && (Y2 == zero)
      && (Y1 - half == F9 - half )) {
      RDiv = Rounded;
      printf("Division appears to round correctly.\n");
      if (GDiv == No) notify("Division");
      }
    else if ((X < zero) && (Y < zero) && (Z < zero) && (T < zero)
      && (Y2 < zero) && (Y1 - half < F9 - half)) {
      RDiv = Chopped;
      printf("Division appears to chop.\n");
      }
    }
  if (RDiv == Other) printf("/ is neither chopped nor correctly rounded.\n");
  bInverse = one / radix;
  tstCond (Failure, (bInverse * radix - half == half),
       "radix * ( 1 / radix ) differs from 1");
  /*=============================================*/
  Milestone = 50;
  /*=============================================*/
  tstCond (Failure, ((F9 + U1) - half == half)
       && ((bMinusU2 + U2 ) - one == radix - one),
       "Incomplete carry-propagation in Addition");
  X = one - U1 * U1;
  Y = one + U2 * (one - U2);
  Z = F9 - half;
  X = (X - half) - Z;
  Y = Y - one;
  if ((X == zero) && (Y == zero)) {
    RAddSub = Chopped;
    printf("Add/Subtract appears to be chopped.\n");
    }
  if (GAddSub == Yes) {
    X = (half + U2) * U2;
    Y = (half - U2) * U2;
    X = one + X;
    Y = one + Y;
    X = (one + U2) - X;
    Y = one - Y;
    if ((X == zero) && (Y == zero)) {
      X = (half + U2) * U1;
      Y = (half - U2) * U1;
      X = one - X;
      Y = one - Y;
      X = F9 - X;
      Y = one - Y;
      if ((X == zero) && (Y == zero)) {
        RAddSub = Rounded;
        printf("Addition/Subtraction appears to round correctly.\n");
        if (GAddSub == No) notify("Add/Subtract");
        }
      else printf("Addition/Subtraction neither rounds nor chops.\n");
      }
    else printf("Addition/Subtraction neither rounds nor chops.\n");
    }
  else printf("Addition/Subtraction neither rounds nor chops.\n");
  S = one;
  X = one + half * (one + half);
  Y = (one + U2) * half;
  Z = X - Y;
  T = Y - X;
  StickyBit = Z + T;
  if (StickyBit != zero) {
    S = zero;
    badCond(Flaw, "(X - Y) + (Y - X) is non zero!\n");
    }
  StickyBit = zero;
  if ((GMult == Yes) && (GDiv == Yes) && (GAddSub == Yes)
    && (RMult == Rounded) && (RDiv == Rounded)
    && (RAddSub == Rounded) && (std::floor(radixD2) == radixD2)) {
    printf("Checking for sticky bit.\n");
    X = (half + U1) * U2;
    Y = half * U2;
    Z = one + Y;
    T = one + X;
    if ((Z - one <= zero) && (T - one >= U2)) {
      Z = T + Y;
      Y = Z - X;
      if ((Z - T >= U2) && (Y - T == zero)) {
        X = (half + U1) * U1;
        Y = half * U1;
        Z = one - Y;
        T = one - X;
        if ((Z - one == zero) && (T - F9 == zero)) {
          Z = (half - U1) * U1;
          T = F9 - Z;
          Q = F9 - Y;
          if ((T - F9 == zero) && (F9 - U1 - Q == zero)) {
            Z = (one + U2) * oneAndHalf;
            T = (oneAndHalf + U2) - Z + U2;
            X = one + half / radix;
            Y = one + radix * U2;
            Z = X * Y;
            if (T == zero && X + radix * U2 - Z == zero) {
              if (radix != two) {
                X = two + U2;
                Y = X / two;
                if ((Y - one == zero)) StickyBit = S;
                }
              else StickyBit = S;
              }
            }
          }
        }
      }
    }
  if (StickyBit == one) printf("Sticky bit apparently used correctly.\n");
  else printf("Sticky bit used incorrectly or not at all.\n");
  tstCond (Flaw, !(GMult == No || GDiv == No || GAddSub == No ||
      RMult == Other || RDiv == Other || RAddSub == Other),
    "lack(s) of guard digits or failure(s) to correctly round or chop\n(noted above) count as one flaw in the final tally below");
  /*=============================================*/
  Milestone = 60;
  /*=============================================*/
  printf("\n");
  printf("Does Multiplication commute?  ");
  printf("Testing on %d random pairs.\n", noTrials);
  Random9 = std::sqrt(3.0);
  Random1 = Third;
  I = 1;
  do  {
    X = random();
    Y = random();
    Z9 = Y * X;
    Z = X * Y;
    Z9 = Z - Z9;
    I = I + 1;
    } while ( ! ((I > noTrials) || (Z9 != zero)));
  if (I == noTrials) {
    Random1 = one + half / three;
    Random2 = (U2 + U1) + one;
    Z = Random1 * Random2;
    Y = Random2 * Random1;
    Z9 = (one + half / three) * ((U2 + U1) + one) - (one + half /
      three) * ((U2 + U1) + one);
    }
  if (! ((I == noTrials) || (Z9 == zero)))
    badCond(Defect, "X * Y == Y * X trial fails.\n");
  else printf("     No failures found in %d integer pairs.\n", noTrials);
  /*=============================================*/
  Milestone = 70;
  /*=============================================*/
  printf("\nRunning test of square root(x).\n");
  tstCond (Failure, (zero == std::sqrt(zero))
       && (- zero == std::sqrt(- zero))
       && (one == std::sqrt(one)), "Square root of 0.0, -0.0 or 1.0 wrong");
  MinSqEr = zero;
  MaxSqEr = zero;
  J = zero;
  X = radix;
  OneUlp = U2;
  sqXMinX (Serious);
  X = bInverse;
  OneUlp = bInverse * U1;
  sqXMinX (Serious);
  X = U1;
  OneUlp = U1 * U1;
  sqXMinX (Serious);
  if (J != zero) pause();
  printf("Testing if sqrt(X * X) == X for %d Integers X.\n", noTrials);
  J = zero;
  X = two;
  Y = radix;
  if ((radix != one)) do  {
    X = Y;
    Y = radix * Y;
    } while ( ! ((Y - X >= noTrials)));
  OneUlp = X * U2;
  I = 1;
  while (I <= noTrials) {
    X = X + one;
    sqXMinX (Defect);
    if (J > zero) break;
    I = I + 1;
    }
  printf("Test for sqrt monotonicity.\n");
  I = - 1;
  X = bMinusU2;
  Y = radix;
  Z = radix + radix * U2;
  NotMonot = False;
  Monot = False;
  while ( ! (NotMonot || Monot)) {
    I = I + 1;
    X = std::sqrt(X);
    Q = std::sqrt(Y);
    Z = std::sqrt(Z);
    if ((X > Q) || (Q > Z)) NotMonot = True;
    else {
      Q = std::floor(Q + half);
      if ((I > 0) || (radix == Q * Q)) Monot = True;
      else if (I > 0) {
      if (I > 1) Monot = True;
      else {
        Y = Y * bInverse;
        X = Y - U1;
        Z = Y + U1;
        }
      }
      else {
        Y = Q;
        X = Y - U2;
        Z = Y + U2;
        }
      }
    }
  if (Monot) printf("sqrt has passed a test for Monotonicity.\n");
  else {
    badCond(Defect, "");
    printf("sqrt(X) is non-monotonic for X near %.7e .\n", Y);
    }
  /*=============================================*/
  Milestone = 80;
  /*=============================================*/
  MinSqEr = MinSqEr + half;
  MaxSqEr = MaxSqEr - half;
  Y = (std::sqrt(one + U2) - one) / U2;
  SqEr = (Y - one) + U2 / eight;
  if (SqEr > MaxSqEr) MaxSqEr = SqEr;
  SqEr = Y + U2 / eight;
  if (SqEr < MinSqEr) MinSqEr = SqEr;
  Y = ((std::sqrt(F9) - U2) - (one - U2)) / U1;
  SqEr = Y + U1 / eight;
  if (SqEr > MaxSqEr) MaxSqEr = SqEr;
  SqEr = (Y + one) + U1 / eight;
  if (SqEr < MinSqEr) MinSqEr = SqEr;
  OneUlp = U2;
  X = OneUlp;
  for( indx = 1; indx <= 3; ++indx) {
    Y = std::sqrt((X + U1 + X) + F9);
    Y = ((Y - U2) - ((one - U2) + X)) / OneUlp;
    Z = ((U1 - X) + F9) * half * X * X / OneUlp;
    SqEr = (Y + half) + Z;
    if (SqEr < MinSqEr) MinSqEr = SqEr;
    SqEr = (Y - half) + Z;
    if (SqEr > MaxSqEr) MaxSqEr = SqEr;
    if (((indx == 1) || (indx == 3)))
      X = OneUlp * sign (X) * std::floor(eight / (nine * std::sqrt(OneUlp)));
    else {
      OneUlp = U1;
      X = - OneUlp;
      }
    }
  /*=============================================*/
  Milestone = 85;
  /*=============================================*/
  SqRWrng = False;
  Anomaly = False;
  RSqrt = Other; /* ~dgh */
  if (radix != one) {
    printf("Testing whether sqrt is rounded or chopped.\n");
    D = std::floor(half + pow(radix, one + Precision - std::floor(Precision)));
  /* ... == radix^(1 + fract) if (Precision == Integer + fract. */
    X = D / radix;
    Y = D / a1;
    if ((X != std::floor(X)) || (Y != std::floor(Y))) {
      Anomaly = True;
      }
    else {
      X = zero;
      Z2 = X;
      Y = one;
      Y2 = Y;
      Z1 = radix - one;
      FourD = four * D;
      do  {
        if (Y2 > Z2) {
          Q = radix;
          Y1 = Y;
          do  {
            X1 = std::abs(Q + std::floor(half - Q / Y1) * Y1);
            Q = Y1;
            Y1 = X1;
            } while ( ! (X1 <= zero));
          if (Q <= one) {
            Z2 = Y2;
            Z = Y;
            }
          }
        Y = Y + two;
        X = X + eight;
        Y2 = Y2 + X;
        if (Y2 >= FourD) Y2 = Y2 - FourD;
        } while ( ! (Y >= D));
      X8 = FourD - Z2;
      Q = (X8 + Z * Z) / FourD;
      X8 = X8 / eight;
      if (Q != std::floor(Q)) Anomaly = True;
      else {
        Break = False;
        do  {
          X = Z1 * Z;
          X = X - std::floor(X / radix) * radix;
          if (X == one)
            Break = True;
          else
            Z1 = Z1 - one;
          } while ( ! (Break || (Z1 <= zero)));
        if ((Z1 <= zero) && (! Break)) Anomaly = True;
        else {
          if (Z1 > radixD2) Z1 = Z1 - radix;
          do  {
            newD();
            } while ( ! (U2 * D >= F9));
          if (D * radix - D != W - D) Anomaly = True;
          else {
            Z2 = D;
            I = 0;
            Y = D + (one + Z) * half;
            X = D + Z + Q;
            sr3750();
            Y = D + (one - Z) * half + D;
            X = D - Z + D;
            X = X + Q + X;
            sr3750();
            newD();
            if (D - Z2 != W - Z2) Anomaly = True;
            else {
              Y = (D - Z2) + (Z2 + (one - Z) * half);
              X = (D - Z2) + (Z2 - Z + Q);
              sr3750();
              Y = (one + Z) * half;
              X = Q;
              sr3750();
              if (I == 0) Anomaly = True;
              }
            }
          }
        }
      }
    if ((I == 0) || Anomaly) {
      badCond(Failure, "Anomalous arithmetic with Integer < ");
      printf("radix^Precision = %.7e\n", W);
      printf(" fails test whether sqrt rounds or chops.\n");
      SqRWrng = True;
      }
    }
  if (! Anomaly) {
    if (! ((MinSqEr < zero) || (MaxSqEr > zero))) {
      RSqrt = Rounded;
      printf("Square root appears to be correctly rounded.\n");
      }
    else  {
      if ((MaxSqEr + U2 > U2 - half) || (MinSqEr > half)
        || (MinSqEr + radix < half)) SqRWrng = True;
      else {
        RSqrt = Chopped;
        printf("Square root appears to be chopped.\n");
        }
      }
    }
  if (SqRWrng) {
    printf("Square root is neither chopped nor correctly rounded.\n");
    printf("Observed errors run from %.7e ", MinSqEr - half);
    printf("to %.7e ulps.\n", half + MaxSqEr);
    tstCond (Serious, MaxSqEr - MinSqEr < radix * radix,
      "sqrt gets too many last digits wrong");
    }
  /*=============================================*/
  Milestone = 90;
  /*=============================================*/
  pause();
  printf("Testing powers Z^i for small Integers Z and i.\n");
  N = 0;
  /* ... test powers of zero. */
  I = 0;
  Z = -zero;
  M = 3;
  Break = False;
  do  {
    X = one;
    sr3980();
    if (I <= 10) {
      I = 1023;
      sr3980();
      }
    if (Z == minusOne) Break = True;
    else {
      Z = minusOne;
      printIfNPositive();
      N = 0;
      /* .. if(-1)^N is invalid, replace minusOne by one. */
      I = - 4;
      }
    } while ( ! Break);
  printIfNPositive();
  N1 = N;
  N = 0;
  Z = a1;
  M = (int)(std::floor(two * std::log(W) / std::log(a1)));
  Break = False;
  do  {
    X = Z;
    I = 1;
    sr3980();
    if (Z == aInverse) Break = True;
    else Z = aInverse;
    } while ( ! (Break));
  /*=============================================*/
    Milestone = 100;
  /*=============================================*/
  /*  Powers of radix have been tested, */
  /*         next try a few primes     */
  M = noTrials;
  Z = three;
  do  {
    X = Z;
    I = 1;
    sr3980();
    do  {
      Z = Z + two;
      } while ( three * std::floor(Z / three) == Z );
    } while ( Z < eight * three );
  if (N > 0) {
    printf("Errors like this may invalidate financial calculations\n");
    printf("\tinvolving interest rates.\n");
    }
  printIfNPositive();
  N += N1;
  if (N == 0) printf("... no discrepancis found.\n");
  if (N > 0) pause();
  else printf("\n");
  /*=============================================*/
  Milestone = 110;
  /*=============================================*/
  printf("Seeking Underflow thresholds UfThold and E0.\n");
  D = U1;
  if (Precision != std::floor(Precision)) {
    D = bInverse;
    X = Precision;
    do  {
      D = D * bInverse;
      X = X - one;
      } while ( X > zero);
    }
  Y = one;
  Z = D;
  /* ... D is power of 1/radix < 1. */
  do  {
    C = Y;
    Y = Z;
    Z = Y * Y;
    } while ((Y > Z) && (Z + Z > Z));
  Y = C;
  Z = Y * D;
  do  {
    C = Y;
    Y = Z;
    Z = Y * D;
    } while ((Y > Z) && (Z + Z > Z));
  if (radix < two) HInvrse = two;
  else HInvrse = radix;
  H = one / HInvrse;
  /* ... 1/HInvrse == H == Min(1/radix, 1/2) */
  CInvrse = one / C;
  E0 = C;
  Z = E0 * H;
  /* ...1/radix^(BIG Integer) << 1 << CInvrse == 1/C */
  do  {
    Y = E0;
    E0 = Z;
    Z = E0 * H;
    } while ((E0 > Z) && (Z + Z > Z));
  UfThold = E0;
  E1 = zero;
  Q = zero;
  E9 = U2;
  S = one + E9;
  D = C * S;
  if (D <= C) {
    E9 = radix * U2;
    S = one + E9;
    D = C * S;
    if (D <= C) {
      badCond(Failure, "multiplication gets too many last digits wrong.\n");
      Underflow = E0;
      Y1 = zero;
      PseudoZero = Z;
      pause();
      }
    }
  else {
    Underflow = D;
    PseudoZero = Underflow * H;
    UfThold = zero;
    do  {
      Y1 = Underflow;
      Underflow = PseudoZero;
      if (E1 + E1 <= E1) {
        Y2 = Underflow * HInvrse;
        E1 = std::abs(Y1 - Y2);
        Q = Y1;
        if ((UfThold == zero) && (Y1 != Y2)) UfThold = Y1;
        }
      PseudoZero = PseudoZero * H;
      } while ((Underflow > PseudoZero)
        && (PseudoZero + PseudoZero > PseudoZero));
    }
  /* Comment line 4530 .. 4560 */
  if (PseudoZero != zero) {
    printf("\n");
    Z = PseudoZero;
  /* ... Test PseudoZero for "phoney- zero" violates */
  /* ... PseudoZero < Underflow or PseudoZero < PseudoZero + PseudoZero
       ... */
    if (PseudoZero <= zero) {
      badCond(Failure, "Positive expressions can underflow to an\n");
      printf("allegedly negative value\n");
      printf("PseudoZero that prints out as: %g .\n", PseudoZero);
      X = - PseudoZero;
      if (X <= zero) {
        printf("But -PseudoZero, which should be\n");
        printf("positive, isn't; it prints out as  %g .\n", X);
        }
      }
    else {
      badCond(Flaw, "Underflow can stick at an allegedly positive\n");
      printf("value PseudoZero that prints out as %g .\n", PseudoZero);
      }
    tstPtUf();
    }
  /*=============================================*/
  Milestone = 120;
  /*=============================================*/
  if (CInvrse * Y > CInvrse * Y1) {
    S = H * S;
    E0 = Underflow;
    }
  if (! ((E1 == zero) || (E1 == E0))) {
    badCond(Defect, "");
    if (E1 < E0) {
      printf("Products underflow at a higher");
      printf(" threshold than differences.\n");
      if (PseudoZero == zero)
      E0 = E1;
      }
    else {
      printf("Difference underflows at a higher");
      printf(" threshold than products.\n");
      }
    }
  printf("Smallest strictly positive number found is E0 = %g .\n", E0);
  Z = E0;
  tstPtUf();
  Underflow = E0;
  if (N == 1) Underflow = Y;
  I = 4;
  if (E1 == zero) I = 3;
  if (UfThold == zero) I = I - 2;
  UfNGrad = True;
  switch (I)  {
    case  1:
    UfThold = Underflow;
    if ((CInvrse * Q) != ((CInvrse * Y) * S)) {
      UfThold = Y;
      badCond(Failure, "Either accuracy deteriorates as numbers\n");
      printf("approach a threshold = %.17e\n", UfThold);;
      printf(" coming down from %.17e\n", C);
      printf(" or else multiplication gets too many last digits wrong.\n");
      }
    pause();
    break;

    case  2:
    badCond(Failure, "Underflow confuses Comparison, which alleges that\n");
    printf("Q == Y while denying that |Q - Y| == 0; these values\n");
    printf("print out as Q = %.17e, Y = %.17e .\n", Q, Y2);
    printf ("|Q - Y| = %.17e .\n" , std::abs(Q - Y2));
    UfThold = Q;
    break;

    case  3:
    X = X;
    break;

    case  4:
    if ((Q == UfThold) && (E1 == E0)
      && (std::abs( UfThold - E1 / E9) <= E1)) {
      UfNGrad = False;
      printf("Underflow is gradual; it incurs Absolute Error =\n");
      printf("(roundoff in UfThold) < E0.\n");
      Y = E0 * CInvrse;
      Y = Y * (oneAndHalf + U2);
      X = CInvrse * (one + U2);
      Y = Y / X;
      IEEE = (Y == E0);
      }
    }
  if (UfNGrad) {
    printf("\n");
    sigsave = sigfpe;
    if (setjmp(ovfl_buf)) {
      printf("Underflow / UfThold failed!\n");
      R = H + H;
      }
    else R = std::sqrt(Underflow / UfThold);
    sigsave = 0;
    if (R <= H) {
      Z = R * UfThold;
      X = Z * (one + R * H * (one + H));
      }
    else {
      Z = UfThold;
      X = Z * (one + H * H * (one + H));
      }
    if (! ((X == Z) || (X - Z != zero))) {
      badCond(Flaw, "");
      printf("X = %.17e\n\tis not equal to Z = %.17e .\n", X, Z);
      Z9 = X - Z;
      printf("yet X - Z yields %.17e .\n", Z9);
      printf("    Should this NOT signal Underflow, ");
      printf("this is a SERIOUS DEFECT\nthat causes ");
      printf("confusion when innocent statements like\n");;
      printf("    if (X == Z)  ...  else");
      printf("  ... (f(X) - f(Z)) / (X - Z) ...\n");
      printf("encounter Division by zero although actually\n");
      sigsave = sigfpe;
      if (setjmp(ovfl_buf)) printf("X / Z fails!\n");
      else printf("X / Z = 1 + %g .\n", (X / Z - half) - half);
      sigsave = 0;
      }
    }
  printf("The Underflow threshold is %.17e, %s\n", UfThold,
       " below which");
  printf("calculation may suffer larger Relative error than ");
  printf("merely roundoff.\n");
  Y2 = U1 * U1;
  Y = Y2 * Y2;
  Y2 = Y * U1;
  if (Y2 <= UfThold) {
    if (Y > E0) {
      badCond(Defect, "");
      I = 5;
      }
    else {
      badCond(Serious, "");
      I = 4;
      }
    printf("Range is too narrow; U1^%d Underflows.\n", I);
    }
  /*=============================================*/
  Milestone = 130;
  /*=============================================*/
  Y = - std::floor(half - twoForty * std::log(UfThold) / std::log(HInvrse)) / twoForty;
  Y2 = Y + Y;
  printf("Since underflow occurs below the threshold\n");
  printf("UfThold = (%.17e) ^ (%.17e)\nonly underflow ", HInvrse, Y);
  printf("should afflict the expression\n\t(%.17e) ^ (%.17e);\n", HInvrse, Y);
  V9 = pow(HInvrse, Y2);
  printf("actually calculating yields: %.17e .\n", V9);
  if (! ((V9 >= zero) && (V9 <= (radix + radix + E9) * UfThold))) {
    badCond(Serious, "this is not between 0 and underflow\n");
    printf("   threshold = %.17e .\n", UfThold);
    }
  else if (! (V9 > UfThold * (one + E9)))
    printf("This computed value is O.K.\n");
  else {
    badCond(Defect, "this is not between 0 and underflow\n");
    printf("   threshold = %.17e .\n", UfThold);
    }
  /*=============================================*/
  Milestone = 140;
  /*=============================================*/
  printf("\n");
  /* ...calculate Exp2 == exp(2) == 7.389056099... */
  X = zero;
  I = 2;
  Y = two * three;
  Q = zero;
  N = 0;
  do  {
    Z = X;
    I = I + 1;
    Y = Y / (I + I);
    R = Y + Q;
    X = Z + R;
    Q = (Z - X) + R;
    } while(X > Z);
  Z = (oneAndHalf + one / eight) + X / (oneAndHalf * thirtyTwo);
  X = Z * Z;
  Exp2 = X * X;
  X = F9;
  Y = X - U1;
  printf("Testing X^((X + 1) / (X - 1)) vs. exp(2) = %.17e as X -> 1.\n",
    Exp2);
  for(I = 1;;) {
    Z = X - bInverse;
    Z = (X + one) / (Z - (one - bInverse));
    Q = pow(X, Z) - Exp2;
    if (std::abs(Q) > twoForty * U2) {
      N = 1;
       V9 = (X - bInverse) - (one - bInverse);
      badCond(Defect, "Calculated");
      printf(" %.17e for\n", pow(X,Z));
      printf("\t(1 + (%.17e) ^ (%.17e);\n", V9, Z);
      printf("\tdiffers from correct value by %.17e .\n", Q);
      printf("\tThis much error may spoil financial\n");
      printf("\tcalculations involving tiny interest rates.\n");
      break;
      }
    else {
      Z = (Y - X) * two + Y;
      X = Y;
      Y = Z;
      Z = one + (X - F9)*(X - F9);
      if (Z > one && I < noTrials) I++;
      else  {
        if (X > one) {
          if (N == 0)
             printf("Accuracy seems adequate.\n");
          break;
          }
        else {
          X = one + U2;
          Y = U2 + U2;
          Y += X;
          I = 1;
          }
        }
      }
    }
  /*=============================================*/
  Milestone = 150;
  /*=============================================*/
  printf("Testing powers Z^Q at four nearly extreme values.\n");
  N = 0;
  Z = a1;
  Q = std::floor(half - std::log(C) / std::log(a1));
  Break = False;
  do  {
    X = CInvrse;
    Y = pow(Z, Q);
    isYEqX();
    Q = - Q;
    X = C;
    Y = pow(Z, Q);
    isYEqX();
    if (Z < one) Break = True;
    else Z = aInverse;
    } while ( ! (Break));
  printIfNPositive();
  if (N == 0) printf(" ... no discrepancies found.\n");
  printf("\n");

  /*=============================================*/
  Milestone = 160;
  /*=============================================*/
  pause();
  printf("Searching for Overflow threshold:\n");
  printf("This may generate an error.\n");
  Y = - CInvrse;
  V9 = HInvrse * Y;
  sigsave = sigfpe;
  if (setjmp(ovfl_buf)) { I = 0; V9 = Y; goto overflow; }
  do {
    V = Y;
    Y = V9;
    V9 = HInvrse * Y;
    } while(V9 < Y);
  I = 1;
overflow:
  sigsave = 0;
  Z = V9;
  printf("Can `Z = -Y' overflow?\n");
  printf("Trying it on Y = %.17e .\n", Y);
  V9 = - Y;
  V0 = V9;
  if (V - Y == V + V0) printf("Seems O.K.\n");
  else {
    printf("finds a ");
    badCond(Flaw, "-(-Y) differs from Y.\n");
    }
  if (Z != Y) {
    badCond(Serious, "");
    printf("overflow past %.17e\n\tshrinks to %.17e .\n", Y, Z);
    }
  if (I) {
    Y = V * (HInvrse * U2 - HInvrse);
    Z = Y + ((one - HInvrse) * U2) * V;
    if (Z < V0) Y = Z;
    if (Y < V0) V = Y;
    if (V0 - V < V0) V = V0;
    }
  else {
    V = Y * (HInvrse * U2 - HInvrse);
    V = V + ((one - HInvrse) * U2) * Y;
    }
  printf("Overflow threshold is V  = %.17e .\n", V);
  if (I) printf("Overflow saturates at V0 = %.17e .\n", V0);
  else printf("There is no saturation value because the system traps on overflow.\n");
  V9 = V * one;
  printf("No Overflow should be signaled for V * 1 = %.17e\n", V9);
  V9 = V / one;
  printf("                           nor for V / 1 = %.17e .\n", V9);
  printf("Any overflow signal separating this * from the one\n");
  printf("above is a DEFECT.\n");
  /*=============================================*/
  Milestone = 170;
  /*=============================================*/
  if (!(-V < V && -V0 < V0 && -UfThold < V && UfThold < V)) {
    badCond(Failure, "Comparisons involving ");
    printf("+-%g, +-%g\nand +-%g are confused by Overflow.",
      V, V0, UfThold);
    }
  /*=============================================*/
  Milestone = 175;
  /*=============================================*/
  printf("\n");
  for(indx = 1; indx <= 3; ++indx) {
    switch (indx)  {
      case 1: Z = UfThold; break;
      case 2: Z = E0; break;
      case 3: Z = PseudoZero; break;
      }
    if (Z != zero) {
      V9 = std::sqrt(Z);
      Y = V9 * V9;
      if (Y / (one - radix * E9) < Z
         || Y > (one + radix * E9) * Z) { /* dgh: + E9 --> * E9 */
        if (V9 > U1) badCond(Serious, "");
        else badCond(Defect, "");
        printf("Comparison alleges that what prints as Z = %.17e\n", Z);
        printf(" is too far from sqrt(Z) ^ 2 = %.17e .\n", Y);
        }
      }
    }
  /*=============================================*/
  Milestone = 180;
  /*=============================================*/
  for(indx = 1; indx <= 2; ++indx) {
    if (indx == 1) Z = V;
    else Z = V0;
    V9 = std::sqrt(Z);
    X = (one - radix * E9) * V9;
    V9 = V9 * X;
    if (((V9 < (one - two * radix * E9) * Z) || (V9 > Z))) {
      Y = V9;
      if (X < W) badCond(Serious, "");
      else badCond(Defect, "");
      printf("Comparison alleges that Z = %17e\n", Z);
      printf(" is too far from sqrt(Z) ^ 2 (%.17e) .\n", Y);
      }
    }
  /*=============================================*/
  Milestone = 190;
  /*=============================================*/
  pause();
  X = UfThold * V;
  Y = radix * radix;
  if (X*Y < one || X > Y) {
    if (X * Y < U1 || X > Y/U1) badCond(Defect, "Badly");
    else badCond(Flaw, "");

    printf(" unbalanced range; UfThold * V = %.17e\n\t%s\n",
      X, "is too far from 1.\n");
    }
  /*=============================================*/
  Milestone = 200;
  /*=============================================*/
  for (indx = 1; indx <= 5; ++indx)  {
    X = F9;
    switch (indx)  {
      case 2: X = one + U2; break;
      case 3: X = V; break;
      case 4: X = UfThold; break;
      case 5: X = radix;
      }
    Y = X;
    sigsave = sigfpe;
    if (setjmp(ovfl_buf))
      printf("  X / X  traps when X = %g\n", X);
    else {
      V9 = (Y / X - half) - half;
      if (V9 == zero) continue;
      if (V9 == - U1 && indx < 5) badCond(Flaw, "");
      else badCond(Serious, "");
      printf("  X / X differs from 1 when X = %.17e\n", X);
      printf("  instead, X / X - 1/2 - 1/2 = %.17e .\n", V9);
      }
    sigsave = 0;
    }
  /*=============================================*/
  Milestone = 210;
  /*=============================================*/
  MyZero = zero;
  printf("\n");
  printf("What message and/or values does Division by zero produce?\n") ;
  sigsave = sigfpe;
  printf("    Trying to compute 1 / 0 produces ...");
  if (!setjmp(ovfl_buf)) printf("  %.7e .\n", one / MyZero);
  sigsave = 0;
  sigsave = sigfpe;
  printf("\n    Trying to compute 0 / 0 produces ...");
  if (!setjmp(ovfl_buf)) printf("  %.7e .\n", zero / MyZero);
  sigsave = 0;
  /*=============================================*/
  Milestone = 220;
  /*=============================================*/
  pause();
  printf("\n");
  {
    static const char *msg[] = {
      "FAILUREs  encountered =",
      "SERIOUS DEFECTs  discovered =",
      "DEFECTs  discovered =",
      "FLAWs  discovered =" };
    int i;
    for(i = 0; i < 4; i++) if (ErrCnt[i])
      printf("The number of  %-29s %d.\n",
        msg[i], ErrCnt[i]);
    }
  printf("\n");
  if ((ErrCnt[Failure] + ErrCnt[Serious] + ErrCnt[Defect]
      + ErrCnt[Flaw]) > 0) {
    if ((ErrCnt[Failure] + ErrCnt[Serious] + ErrCnt[
      Defect] == 0) && (ErrCnt[Flaw] > 0)) {
      printf("The arithmetic diagnosed seems ");
      printf("Satisfactory though flawed.\n");
      }
    if ((ErrCnt[Failure] + ErrCnt[Serious] == 0)
      && ( ErrCnt[Defect] > 0)) {
      printf("The arithmetic diagnosed may be Acceptable\n");
      printf("despite inconvenient Defects.\n");
      }
    if ((ErrCnt[Failure] + ErrCnt[Serious]) > 0) {
      printf("The arithmetic diagnosed has ");
      printf("unacceptable Serious Defects.\n");
      }
    if (ErrCnt[Failure] > 0) {
      printf("Potentially fatal FAILURE may have spoiled this");
      printf(" program's subsequent diagnoses.\n");
      }
    }
  else {
    printf("No failures, defects nor flaws have been discovered.\n");
    if (! ((RMult == Rounded) && (RDiv == Rounded)
      && (RAddSub == Rounded) && (RSqrt == Rounded)))
      printf("The arithmetic diagnosed seems Satisfactory.\n");
    else {
      if (StickyBit >= one &&
        (radix - two) * (radix - nine - one) == zero) {
        printf("Rounding appears to conform to ");
        printf("the proposed IEEE standard P");
        if ((radix == two) &&
           ((Precision - four * three * two) *
            ( Precision - twentySeven -
             twentySeven + one) == zero))
          printf("754");
        else printf("854");
        if (IEEE) printf(".\n");
        else {
          printf(",\nexcept for possibly Double Rounding");
          printf(" during Gradual Underflow.\n");
          }
        }
      printf("The arithmetic diagnosed appears to be Excellent!\n");
      }
    }
  if (fpecount)
    printf("\nA total of %d floating point exceptions were registered.\n",
      fpecount);
  printf("END OF TEST.\n");
  return { Milestone, fpecount + ErrCnt[Failure] + ErrCnt[Serious] + ErrCnt[Defect] + ErrCnt[Flaw] };
  }

/* sign */

template <typename F>
F Paranoia<F>::sign (F X)
{ return X >= 0. ? 1.0 : -1.0; }

/* pause */

template <typename F>
void Paranoia<F>::pause(void)
{
  printf("\nDiagnosis resumes after milestone Number %d", Milestone);
  printf("          Page: %d\n\n", PageNo);
  ++Milestone;
  ++PageNo;
  }

 /* tstCond */

template <typename F>
void Paranoia<F>::tstCond (int K, int Valid, const char *T)
{ if (! Valid) { badCond(K,T); printf(".\n"); } }

template <typename F>
void Paranoia<F>::badCond(int K, const char *T)
{
  static const char *msg[] = { "FAILURE", "SERIOUS DEFECT", "DEFECT", "FLAW" };

  ErrCnt [K] = ErrCnt [K] + 1;
  printf("%s:  %s", msg[K], T);
  }

/* random */
/*  random computes
     X = (Random1 + Random9)^5
     Random1 = X - std::floor(X) + 0.000005 * X;
   and returns the new value of Random1
*/

template <typename F>
F Paranoia<F>::random(void)
{
  F X, Y;

  X = Random1 + Random9;
  Y = X * X;
  Y = Y * Y;
  X = X * Y;
  Y = X - std::floor(X);
  Random1 = Y + X * 0.000005;
  return(Random1);
  }

/* sqXMinX */

template <typename F>
void Paranoia<F>::sqXMinX (int ErrKind)
{
  F XA, XB;

  XB = X * bInverse;
  XA = X - XB;
  SqEr = ((std::sqrt(X * X) - XB) - XA) / OneUlp;
  if (SqEr != zero) {
    if (SqEr < MinSqEr) MinSqEr = SqEr;
    if (SqEr > MaxSqEr) MaxSqEr = SqEr;
    J = J + 1.0;
    badCond(ErrKind, "\n");
    printf("sqrt( %.17e) - %.17e  = %.17e\n", X * X, X, OneUlp * SqEr);
    printf("\tinstead of correct value 0 .\n");
    }
  }

/* newD */

template <typename F>
void Paranoia<F>::newD(void)
{
  X = Z1 * Q;
  X = std::floor(half - X / radix) * radix + X;
  Q = (Q - X * Z) / radix + X * X * (D / radix);
  Z = Z - two * X * D;
  if (Z <= zero) {
    Z = - Z;
    Z1 = - Z1;
    }
  D = radix * D;
  }

/* sr3750 */

template <typename F>
void Paranoia<F>::sr3750(void)
{
  if (! ((X - radix < Z2 - radix) || (X - Z2 > W - Z2))) {
    I = I + 1;
    X2 = std::sqrt(X * D);
    Y2 = (X2 - Z2) - (Y - Z2);
    X2 = X8 / (Y - half);
    X2 = X2 - half * X2 * X2;
    SqEr = (Y2 + half) + (half - X2);
    if (SqEr < MinSqEr) MinSqEr = SqEr;
    SqEr = Y2 - X2;
    if (SqEr > MaxSqEr) MaxSqEr = SqEr;
    }
  }

/* isYEqX */

template <typename F>
void Paranoia<F>::isYEqX(void)
{
  if (Y != X) {
    if (N <= 0) {
      if (Z == zero && Q <= zero)
        printf("WARNING:  computing\n");
      else badCond(Defect, "computing\n");
      printf("\t(%.17e) ^ (%.17e)\n", Z, Q);
      printf("\tyielded %.17e;\n", Y);
      printf("\twhich compared unequal to correct %.17e ;\n",
        X);
      printf("\t\tthey differ by %.17e .\n", Y - X);
      }
    N = N + 1; /* ... count discrepancies. */
    }
  }

/* sr3980 */

template <typename F>
void Paranoia<F>::sr3980(void)
{
  do {
    Q = static_cast<F>(I);
    Y = pow(Z, Q);
    isYEqX();
    if (++I > M) break;
    X = Z * X;
    } while ( X < W );
  }

/* printIfNPositive */

template <typename F>
void Paranoia<F>::printIfNPositive(void)
{
  if (N > 0) printf("Similar discrepancies have occurred %d times.\n", N);
  }

/* tstPtUf */

template <typename F>
void Paranoia<F>::tstPtUf(void)
{
  N = 0;
  if (Z != zero) {
    printf("Since comparison denies Z = 0, evaluating ");
    printf("(Z + Z) / Z should be safe.\n");
    sigsave = sigfpe;
    if (setjmp(ovfl_buf)) goto very_serious;
    Q9 = (Z + Z) / Z;
    printf("What the machine gets for (Z + Z) / Z is  %.17e .\n",
      Q9);
    if (std::abs(Q9 - two) < radix * U2) {
      printf("This is O.K., provided Over/Underflow");
      printf(" has NOT just been signaled.\n");
      }
    else {
      if ((Q9 < one) || (Q9 > two)) {
very_serious:
        N = 1;
        ErrCnt [Serious] = ErrCnt [Serious] + 1;
        printf("This is a VERY SERIOUS DEFECT!\n");
        }
      else {
        N = 1;
        ErrCnt [Defect] = ErrCnt [Defect] + 1;
        printf("This is a DEFECT!\n");
        }
      }
    sigsave = 0;
    V9 = Z * one;
    Random1 = V9;
    V9 = one * Z;
    Random2 = V9;
    V9 = Z / one;
    if ((Z == Random1) && (Z == Random2) && (Z == V9)) {
      if (N > 0) pause();
      }
    else {
      N = 1;
      badCond(Defect, "What prints as Z = ");
      printf("%.17e\n\tcompares different from  ", Z);
      if (Z != Random1) printf("Z * 1 = %.17e ", Random1);
      if (! ((Z == Random2)
        || (Random2 == Random1)))
        printf("1 * Z == %g\n", Random2);
      if (! (Z == V9)) printf("Z / 1 = %.17e\n", V9);
      if (Random2 != Random1) {
        ErrCnt [Defect] = ErrCnt [Defect] + 1;
        badCond(Defect, "Multiplication does not commute!\n");
        printf("\tComparison alleges that 1 * Z = %.17e\n",
          Random2);
        printf("\tdiffers from Z * 1 = %.17e\n", Random1);
        }
      pause();
      }
    }
  }

template <typename F>
void Paranoia<F>::notify(const char *s)
{
  printf("%s test appears to be inconsistent...\n", s);
  printf("   PLEASE NOTIFY KARPINKSI!\n");
  }


/* msglist */

template <typename F>
void Paranoia<F>::msglist(const char **s)
{ while(*s) printf("%s\n", *s++); }

/* instructions */

template <typename F>
void Paranoia<F>::instructions(void)
{
  static const char *instr[] = {
  "Lest this program stop prematurely, i.e. before displaying\n",
  "    `END OF TEST',\n",
  "try to persuade the computer NOT to terminate execution when an",
  "error like Over/Underflow or Division by zero occurs, but rather",
  "to persevere with a surrogate value after, perhaps, displaying some",
  "warning.  If persuasion avails naught, don't despair but run this",
  "program anyway to see how many milestones it passes, and then",
  "amend it to make further progress.\n",
  "Answer questions with Y, y, N or n (unless otherwise indicated).\n",
  0};

  msglist(instr);
  }

/* heading */

template<typename F>
void Paranoia<F>::heading(void)
{
  static const char *head[] = {
  "Users are invited to help debug and augment this program so it will",
  "cope with unanticipated and newly uncovered arithmetic pathologies.\n",
  "Please send suggestions and interesting results to",
  "\tRichard Karpinski",
  "\tComputer Center U-76",
  "\tUniversity of California",
  "\tSan Francisco, CA 94143-0704, USA\n",
  "In doing so, please include the following information:",
#ifdef Single
  "\tPrecision:\tsingle;",
#else
  "\tPrecision:\tdouble;",
#endif
  "\tVersion:\t10 February 1989;",
  "\tComputer:\n",
  "\tCompiler:\n",
  "\tOptimization level:\n",
  "\tOther relevant compiler options:",
  0};

  msglist(head);
  }

/* characteristics */

template <typename F>
void Paranoia<F>::characteristics(void)
{
  static const char *chars[] = {
   "Running this program should reveal these characteristics:",
  "     radix = 1, 2, 4, 8, 10, 16, 100, 256 ...",
  "     Precision = number of significant digits carried.",
  "     U2 = radix/radix^Precision = one Ulp",
  "\t(OneUlpnit in the Last Place) of 1.000xxx .",
  "     U1 = 1/radix^Precision = one Ulp of numbers a little less than 1.0 .",
  "     Adequacy of guard digits for Mult., Div. and Subt.",
  "     Whether arithmetic is chopped, correctly rounded, or something else",
  "\tfor Mult., Div., Add/Subt. and Sqrt.",
  "     Whether a Sticky Bit used correctly for rounding.",
  "     UnderflowThreshold = an underflow threshold.",
  "     E0 and PseudoZero tell whether underflow is abrupt, gradual, or fuzzy.",
  "     V = an overflow threshold, roughly.",
  "     V0  tells, roughly, whether  Infinity  is represented.",
  "     Comparisions are checked for consistency with subtraction",
  "\tand for contamination with pseudo-zeros.",
  "     Sqrt is tested.  Y^X is not tested.",
  "     Extra-precise subexpressions are revealed but NOT YET tested.",
  "     Decimal-Binary conversion is NOT YET tested for accuracy.",
  0};

  msglist(chars);
  }

template <typename F>
void Paranoia<F>::history(void)
{ /* history */
 /* Converted from Brian Wichmann's Pascal version to C by Thos Sumner,
  with further massaging by David M. Gay. */

  static const char *hist[] = {
  "The program attempts to discriminate among",
  "   FLAWs, like lack of a sticky bit,",
  "   Serious DEFECTs, like lack of a guard digit, and",
  "   FAILUREs, like 2+2 == 5 .",
  "Failures may confound subsequent diagnoses.\n",
  "The diagnostic capabilities of this program go beyond an earlier",
  "program called `MACHAR', which can be found at the end of the",
  "book  `Software Manual for the Elementary Functions' (1980) by",
  "W. J. Cody and W. Waite. Although both programs try to discover",
  "the radix, Precision and range (over/underflow thresholds)",
  "of the arithmetic, this program tries to cope with a wider variety",
  "of pathologies, and to say how well the arithmetic is implemented.",
  "\nThe program is based upon a conventional radix representation for",
  "floating-point numbers, but also allows logarithmic encoding",
  "as used by certain early WANG machines.\n",
  "BASIC version of this program (C) 1983 by Prof. W. M. Kahan;",
  "see source comments for more history.",
  0};

  msglist(hist);
  }

template <typename F>
F Paranoia<F>::pow(F x, F y) /* return x ^ y (exponentiation) */
{
  F xy, ye;
  long i;
  int ex, ey = 0, flip = 0;

  if (!y) return 1.0;

  if ((y < -1100. || y > 1100.) && x != -1.) return std::exp(y * std::log(x));

  if (y < 0.) { y = -y; flip = 1; }
  y = std::modf(y, &ye);
  if (y) xy = std::exp(y * std::log(x));
  else xy = 1.0;
  /* next several lines assume >= 32 bit integers */
  x = std::frexp(x, &ex);
  if ((i = static_cast<long>(ye), i)) for(;;) {
    if (i & 1) { xy *= x; ey += ex; }
    if (!(i >>= 1)) break;
    x *= x;
    ex *= 2;
    if (x < .5) { x *= 2.; ex -= 1; }
    }
  if (flip) { xy = 1. / xy; ey = -ey; }
  return std::ldexp(xy, ey);
}
