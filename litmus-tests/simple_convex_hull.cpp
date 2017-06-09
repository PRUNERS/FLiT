#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include <vector>
extern "C" {
#include "quadmath.h"
}
#include "s3fp_utils.h"

#include "simple_convex_hull.h"

using namespace std;

#ifndef IFT
#define IFT float
#endif

#ifndef OFT
#if defined(__CUDA__) || defined(__aarch64__)
#define OFT double
#else
#define OFT __float128
#endif
#endif


long N = 0;
bool RAISE_ASSERTION = false;
int CHMETHOD = -1;
int CANO_FORM = 0;
int SAMPLE_PHASE = 0;
int VERIFY_METHOD = 0;

vector<Point> PointList;
vector<Edge> CHullEdges;

size_t getEdgeCount() {
  return CHullEdges.size();
}

vector<PrimitiveCall> Decisions;

/*
  Some variables for deciding the "localness"
*/
long curr_sample = -1;
FILE *localness_file = NULL;

bool IsOnPQ (Point p, Point q, Point r) {
  struct Point rp;
  rp.id = 0;
  rp.x = p.x - r.x;
  rp.y = p.y - r.y;
  struct Point rq;
  rq.id = 0;
  rq.x = q.x - r.x;
  rq.y = q.y - r.y;

  if (((rp.x * rq.x) + (rp.y * rq.y)) < 0) return true;
  else return false;
}

void Canonical_Unordered_Points (vector<Point>& pv, int cano_selection) {
  assert(1 <= cano_selection && cano_selection <= 2);

  if (pv.size() == 0 || pv.size() == 1) return;

  vector<Point> temp_pv;
  int *new_inds = (int*) malloc(sizeof(int) * pv.size());

  for (size_t i = 0 ; i < pv.size() ; i++)
    new_inds[i] = i;

  for (size_t i = 0 ; i < pv.size() ; i++) {
    for (size_t j = 0 ; j < (pv.size() - 1) ; j++) {
      int this_ind = new_inds[j];
      int next_ind = new_inds[j+1];
      assert(pv[this_ind].id != pv[next_ind].id);
      bool go_swap = false;

      if (cano_selection == 1) {
	if (pv[this_ind].id > pv[next_ind].id) go_swap = true;
      }
      else if (cano_selection == 2) {
	if (pv[this_ind].x < pv[next_ind].x) go_swap = false;
	else if (pv[this_ind].x == pv[next_ind].x) {
	  if (pv[this_ind].y <= pv[next_ind].y) go_swap = false;
	  else go_swap = true;
	}
	else go_swap = true;
      }
      else assert(false);

      if (go_swap) {
	int tempi = new_inds[j];
	new_inds[j] = new_inds[j+1];
	new_inds[j+1] = tempi;
      }
    }
  }

  for (size_t i = 0 ; i < pv.size() ; i++)
    temp_pv.push_back(pv[new_inds[i]]);
  pv.clear();
  pv.insert(pv.begin(), temp_pv.begin(), temp_pv.end());
}

void PrintPoint(Point p) {
  cout << "[" << p.id << "](" << (long double) p.x << ", " << (long double) p.y << ")";
}

void PrintPV (vector<Point> pv) {
  vector<Point>::iterator pi = pv.begin();
  for ( ; pi != pv.end() ; pi++) {
    PrintPoint((*pi));
    cout << endl;
  }
}

void WRONG_CONVEX_HULL (FILE *outfile) {
  if (outfile == NULL) return ;
  OFT odata = 1;
  fwrite(&odata, sizeof(OFT), 1, outfile);
  fwrite(&odata, sizeof(OFT), 1, outfile);

  fwrite(&odata, sizeof(OFT), 1, outfile);
  odata = -1;
  fwrite(&odata, sizeof(OFT), 1, outfile);

  RAISE_ASSERTION = true;
}

void ReadInputs (FILE *infile) {
  assert(infile != NULL);
  IFT idatax;
  IFT idatay;

  fseek(infile, 0, SEEK_END);
  long nbytes = ftell(infile);
  assert(nbytes % (sizeof(IFT) * 2) == 0);
  N = nbytes / (sizeof(IFT) * 2);
  assert(N >= 3);

  fseek(infile, 0, SEEK_SET);
  for (long i = 0 ; i < N ; i++) {
    fread(&idatax, sizeof(IFT), 1, infile);
    fread(&idatay, sizeof(IFT), 1, infile);
    struct Point newPoint;
    newPoint.id = i;
    newPoint.x = (IFT) idatax;
    newPoint.y = (IFT) idatay;
    PointList.push_back(newPoint);
  }
}

void GiftWrappingComputeConvexhull (FILE *outfile) {

  assert(outfile != NULL);

  // Find the right most point
  assert(PointList.size() >= 3);
  struct Point rm_point = PointList[0];
  for (size_t p = 1 ; p < PointList.size() ; p++) {
    if (rm_point.x < PointList[p].x)
      rm_point = PointList[p];
  }

  // gift wrapping
  struct Point this_point = rm_point;
  struct Point next_point;
  while (true) {
    // find the initial candidate
    for (size_t p = 0 ; p < PointList.size() ; p++) {
      if (this_point.id != PointList[p].id) {
	next_point = PointList[p];
	break;
      }
    }

    // iterate through all points
    for (size_t p = 0 ; p < PointList.size() ; p++) {
      assert(this_point.id != next_point.id);
      if (this_point.id == PointList[p].id ||
	  next_point.id == PointList[p].id) continue;

      WFT ret_det;
      int this_ori = Orientation<WFT>(this_point, next_point, PointList[p],
				      ret_det);
      assert(this_ori == 1 || this_ori == 0 || this_ori == -1);
      bool decision = (this_ori == -1 || this_ori == 0);

      struct PrimitiveCall acall;
      acall.idp = this_point.id;
      acall.idq = next_point.id;
      acall.idr = PointList[p].id;
      acall.result = decision;
      Decisions.push_back(acall);

      // replace next_point
      if (decision)
	next_point = PointList[p];
    }

    // add a new edge
    Edge new_edge;
    new_edge.first = this_point;
    new_edge.second = next_point;
    CHullEdges.push_back(new_edge);

    // check if we got the whole convex hull
    if (next_point.id == CHullEdges[0].first.id)
      break;

    // check if we are trapped
    if (CHullEdges.size() > PointList.size())
      break;

    // next iteration
    this_point = CHullEdges[CHullEdges.size()-1].second;
  }

#ifdef __VERBOSE
  vector<Edge>::iterator ei;
  for (ei = CHullEdges.begin() ; ei != CHullEdges.end() ; ei++) {
    cout << "CHulldge: " << endl;
    PrintPoint(ei->first); cout << endl;
    PrintPoint(ei->second); cout << endl;
  }
#endif
}

void VerifyHull(FILE *outfile) {

  assert(outfile != NULL);

  if (VERIFY_METHOD == 0) return ;
  else if (VERIFY_METHOD == 1) {
    WFT ret_det;
    if (CHullEdges.size() < 3) {
      if (VERIFY_METHOD == 1) {
	WRONG_CONVEX_HULL(outfile); return ; }
      else assert(false);
    }

    if (VERIFY_METHOD == 1 &&
	CHullEdges[0].second.id != CHullEdges[1].first.id) {
      WRONG_CONVEX_HULL(outfile); return ;
    }
    int decision = Orientation<WFT>(CHullEdges[0].first,
				    CHullEdges[0].second,
				    CHullEdges[1].second,
				    ret_det);
    size_t i;
    for (i = 1 ; i < (CHullEdges.size()-1) ; i++) {
      if (VERIFY_METHOD == 1 &&
	  CHullEdges[i].second.id != CHullEdges[i+1].first.id) {
	WRONG_CONVEX_HULL(outfile); return ;
      }
      if (decision ==
	  Orientation<WFT>(CHullEdges[i].first,
			   CHullEdges[i].second,
			   CHullEdges[i+1].second,
			   ret_det)) {
	if (VERIFY_METHOD == 1) {
	  WRONG_CONVEX_HULL(outfile); return ; }
      }
    }
    if (VERIFY_METHOD == 1 &&
	CHullEdges[i].second.id != CHullEdges[0].first.id) {
      WRONG_CONVEX_HULL(outfile); return ;
    }
    if (decision ==
	Orientation<WFT>(CHullEdges[i].first,
			 CHullEdges[i].second,
			 CHullEdges[0].second,
			 ret_det)) {
      if (VERIFY_METHOD == 1) {
	WRONG_CONVEX_HULL(outfile); return ; }
    }
  }
  else assert(false && "Error: Invalid verification method");
}

OFT CheckConsistency () {
  cout << "Decision size: " << Decisions.size() << endl;
  for (size_t i = 0 ; i < Decisions.size() ; i++) {
    PrimitiveCall LTxyz;
    if ( Decisions[i].result ) {
      LTxyz.idp = Decisions[i].idp;
      LTxyz.idq = Decisions[i].idq;
      LTxyz.idr = Decisions[i].idr;
    }
    else {
      LTxyz.idp = Decisions[i].idp;
      LTxyz.idq = Decisions[i].idr;
      LTxyz.idr = Decisions[i].idq;
    }
    LTxyz.result = true;

    for (size_t j = 0 ; j < Decisions.size() ; j++) {
      // cyclic symmetry
      if (LTxyz.idp == Decisions[j].idp &&
	  LTxyz.idq == Decisions[j].idq &&
	  LTxyz.idr == Decisions[j].idr &&
	  Decisions[j].result == false) {
#ifdef __VERBOSE
	cout << "Violation of cyclic symmetry: " << endl;
	cout << LTxyz.idp << " " << LTxyz.idq << " " << LTxyz.idr << " " << LTxyz.result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << endl;
#endif
	return -1; }
      if (LTxyz.idp == Decisions[j].idq &&
	  LTxyz.idq == Decisions[j].idr &&
	  LTxyz.idr == Decisions[j].idp &&
	  Decisions[j].result == false) {
#ifdef __VERBOSE
	cout << "Violation of cyclic symmetry: " << endl;
	cout << LTxyz.idp << " " << LTxyz.idq << " " << LTxyz.idr << " " << LTxyz.result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << endl;
#endif
	return -1; }
      if (LTxyz.idp == Decisions[j].idr &&
	  LTxyz.idq == Decisions[j].idp &&
	  LTxyz.idr == Decisions[j].idq &&
	  Decisions[j].result == false) {
#ifdef __VERBOSE
	cout << "Violation of cyclic symmetry: " << endl;
	cout << LTxyz.idp << " " << LTxyz.idq << " " << LTxyz.idr << " " << LTxyz.result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << endl;
#endif
	return -1; }

      // antisymmetry
      if (LTxyz.idp == Decisions[j].idp &&
	  LTxyz.idq == Decisions[j].idr &&
	  LTxyz.idr == Decisions[j].idq &&
	  Decisions[j].result == true) {
#ifdef __VERBOSE
	cout << "Violation of antisymmetry: " << endl;
	cout << LTxyz.idp << " " << LTxyz.idq << " " << LTxyz.idr << " " << LTxyz.result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << endl;
#endif
	return -1; }
      if (LTxyz.idp == Decisions[j].idr &&
	  LTxyz.idq == Decisions[j].idq &&
	  LTxyz.idr == Decisions[j].idp &&
	  Decisions[j].result == true) {
#ifdef __VERBOSE
	cout << "Violation of antisymmetry: " << endl;
	cout << LTxyz.idp << " " << LTxyz.idq << " " << LTxyz.idr << " " << LTxyz.result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << endl;
#endif
	return -1; }
      if (LTxyz.idp == Decisions[j].idq &&
	  LTxyz.idq == Decisions[j].idp &&
	  LTxyz.idr == Decisions[j].idr &&
	  Decisions[j].result == true) {
#ifdef __VERBOSE
	cout << "Violation of antisymmetry: " << endl;
	cout << LTxyz.idp << " " << LTxyz.idq << " " << LTxyz.idr << " " << LTxyz.result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << endl;
#endif
	return -1; }

      // nondegeneracy
      if (Decisions[i].idp == Decisions[j].idp &&
	  Decisions[i].idq == Decisions[j].idr &&
	  Decisions[i].idr == Decisions[j].idq &&
	  Decisions[i].result == false &&
	  Decisions[j].result == false) {
#ifdef __VERBOSE
	cout << "Violation of nondegeneracy: " << endl;
	cout << Decisions[i].idp << " " << Decisions[i].idq << " " << Decisions[i].idr << " " << Decisions[i].result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << endl;
#endif
	return -1; }
      if (Decisions[i].idp == Decisions[j].idr &&
	  Decisions[i].idq == Decisions[j].idq &&
	  Decisions[i].idr == Decisions[j].idp &&
	  Decisions[i].result == false &&
	  Decisions[j].result == false) {
#ifdef __VERBOSE
	cout << "Violation of nondegeneracy: " << endl;
	cout << Decisions[i].idp << " " << Decisions[i].idq << " " << Decisions[i].idr << " " << Decisions[i].result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << endl;
#endif
	return -1; }
      if (Decisions[i].idp == Decisions[j].idq &&
	  Decisions[i].idq == Decisions[j].idp &&
	  Decisions[i].idr == Decisions[j].idr &&
	  Decisions[i].result == false &&
	  Decisions[j].result == false) {
#ifdef __VERBOSE
	cout << "Violation of nondegeneracy: " << endl;
	cout << Decisions[i].idp << " " << Decisions[i].idq << " " << Decisions[i].idr << " " << Decisions[i].result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << endl;
#endif
	return -1; }
    }
  }

  return 1;
}

#ifndef SCH_LIB

int main (int argc, char *argv[]) {
  assert(argc == 7);

  CHMETHOD = atoi(argv[1]);
  assert(CHMETHOD == 0 ||
	 CHMETHOD == 1);

  CANO_FORM = atoi(argv[2]);
  // assert(0 <= CANO_FORM && CANO_FORM <= 2);
  assert(CANO_FORM == 0);

  SAMPLE_PHASE = atoi(argv[3]);
  assert(0 <= SAMPLE_PHASE && SAMPLE_PHASE <= 1);

  VERIFY_METHOD = atoi(argv[4]);
  assert(0 <= VERIFY_METHOD && VERIFY_METHOD <= 1);

  FILE *infile = s3fpGetInFile(argc, argv);
  FILE *outfile = s3fpGetOutFile(argc, argv);

  ReadInputs(infile);

  switch (CHMETHOD) {
  case 0:
    SimpleComputeConvexhull(); //outfile);
    break;
  case 1:
    GiftWrappingComputeConvexhull(outfile);
    break;
  default:
    assert(false && "Error: Invalid convex hull method");
    break;
  }

  VerifyHull(outfile);

  if (VERIFY_METHOD == 0 ||
      VERIFY_METHOD == 1) {
    if (RAISE_ASSERTION == false) {
      OFT odata = 1;
      fwrite(&odata, sizeof(OFT), 1, outfile);
      fwrite(&odata, sizeof(OFT), 1, outfile);

      fwrite(&odata, sizeof(OFT), 1, outfile);
      //printf("hull edge count: %u\n", CHullEdges.size());
      odata = CHullEdges.size();
      fwrite(&odata, sizeof(OFT), 1, outfile);
    }
  }
  else assert(false && "Error: Invalid verification method");

  fclose(infile);
  fclose(outfile);

  return 0;
}

#endif // #ifndef SCH_LIB
