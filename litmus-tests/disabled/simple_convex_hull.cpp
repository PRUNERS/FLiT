/* -- LICENSE BEGIN --
 *
 * Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * Written by
 *   Michael Bentley (mikebentley15@gmail.com),
 *   Geof Sawaya (fredricflinstone@gmail.com),
 *   and Ian Briggs (ian.briggs@utah.edu)
 * under the direction of
 *   Ganesh Gopalakrishnan
 *   and Dong H. Ahn.
 *
 * LLNL-CODE-743137
 *
 * All rights reserved.
 *
 * This file is part of FLiT. For details, see
 *   https://pruners.github.io/flit
 * Please also read
 *   https://github.com/PRUNERS/FLiT/blob/master/LICENSE
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the disclaimer below.
 *
 * - Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the disclaimer
 *   (as noted below) in the documentation and/or other materials
 *   provided with the distribution.
 *
 * - Neither the name of the LLNS/LLNL nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL
 * SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Additional BSD Notice
 *
 * 1. This notice is required to be provided under our contract
 *    with the U.S. Department of Energy (DOE). This work was
 *    produced at Lawrence Livermore National Laboratory under
 *    Contract No. DE-AC52-07NA27344 with the DOE.
 *
 * 2. Neither the United States Government nor Lawrence Livermore
 *    National Security, LLC nor any of their employees, makes any
 *    warranty, express or implied, or assumes any liability or
 *    responsibility for the accuracy, completeness, or usefulness of
 *    any information, apparatus, product, or process disclosed, or
 *    represents that its use would not infringe privately-owned
 *    rights.
 *
 * 3. Also, reference herein to any specific commercial products,
 *    process, or services by trade name, trademark, manufacturer or
 *    otherwise does not necessarily constitute or imply its
 *    endorsement, recommendation, or favoring by the United States
 *    Government or Lawrence Livermore National Security, LLC. The
 *    views and opinions of authors expressed herein do not
 *    necessarily state or reflect those of the United States
 *    Government or Lawrence Livermore National Security, LLC, and
 *    shall not be used for advertising or product endorsement
 *    purposes.
 *
 * -- LICENSE END --
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include <vector>
#include "s3fp_utils.h"

#include "simple_convex_hull.h"

#ifndef IFT
#define IFT float
#endif

#ifndef OFT
#define OFT long double
#endif


long N = 0;
bool RAISE_ASSERTION = false;
int CHMETHOD = -1;
int CANO_FORM = 0;
int SAMPLE_PHASE = 0;
int VERIFY_METHOD = 0;

std::vector<Point> PointList;
std::vector<Edge> CHullEdges;

size_t getEdgeCount() {
  return CHullEdges.size();
}

std::vector<PrimitiveCall> Decisions;

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

void Canonical_Unordered_Points (std::vector<Point>& pv, int cano_selection) {
  assert(1 <= cano_selection && cano_selection <= 2);

  if (pv.size() == 0 || pv.size() == 1) return;

  std::vector<Point> temp_pv;
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
  std::cout << "[" << p.id << "](" << (long double) p.x << ", " << (long double) p.y << ")";
}

void PrintPV (std::vector<Point> pv) {
  std::vector<Point>::iterator pi = pv.begin();
  for ( ; pi != pv.end() ; pi++) {
    PrintPoint((*pi));
    std::cout << std::endl;
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
  std::vector<Edge>::iterator ei;
  for (ei = CHullEdges.begin() ; ei != CHullEdges.end() ; ei++) {
    std::cout << "CHulldge: " << std::endl;
    PrintPoint(ei->first); std::cout << std::endl;
    PrintPoint(ei->second); std::cout << std::endl;
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
  std::cout << "Decision size: " << Decisions.size() << std::endl;
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
	std::cout << "Violation of cyclic symmetry: " << std::endl;
	std::cout << LTxyz.idp << " " << LTxyz.idq << " " << LTxyz.idr << " " << LTxyz.result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << std::endl;
#endif
	return -1; }
      if (LTxyz.idp == Decisions[j].idq &&
	  LTxyz.idq == Decisions[j].idr &&
	  LTxyz.idr == Decisions[j].idp &&
	  Decisions[j].result == false) {
#ifdef __VERBOSE
	std::cout << "Violation of cyclic symmetry: " << std::endl;
	std::cout << LTxyz.idp << " " << LTxyz.idq << " " << LTxyz.idr << " " << LTxyz.result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << std::endl;
#endif
	return -1; }
      if (LTxyz.idp == Decisions[j].idr &&
	  LTxyz.idq == Decisions[j].idp &&
	  LTxyz.idr == Decisions[j].idq &&
	  Decisions[j].result == false) {
#ifdef __VERBOSE
	std::cout << "Violation of cyclic symmetry: " << std::endl;
	std::cout << LTxyz.idp << " " << LTxyz.idq << " " << LTxyz.idr << " " << LTxyz.result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << std::endl;
#endif
	return -1; }

      // antisymmetry
      if (LTxyz.idp == Decisions[j].idp &&
	  LTxyz.idq == Decisions[j].idr &&
	  LTxyz.idr == Decisions[j].idq &&
	  Decisions[j].result == true) {
#ifdef __VERBOSE
	std::cout << "Violation of antisymmetry: " << std::endl;
	std::cout << LTxyz.idp << " " << LTxyz.idq << " " << LTxyz.idr << " " << LTxyz.result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << std::endl;
#endif
	return -1; }
      if (LTxyz.idp == Decisions[j].idr &&
	  LTxyz.idq == Decisions[j].idq &&
	  LTxyz.idr == Decisions[j].idp &&
	  Decisions[j].result == true) {
#ifdef __VERBOSE
	std::cout << "Violation of antisymmetry: " << std::endl;
	std::cout << LTxyz.idp << " " << LTxyz.idq << " " << LTxyz.idr << " " << LTxyz.result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << std::endl;
#endif
	return -1; }
      if (LTxyz.idp == Decisions[j].idq &&
	  LTxyz.idq == Decisions[j].idp &&
	  LTxyz.idr == Decisions[j].idr &&
	  Decisions[j].result == true) {
#ifdef __VERBOSE
	std::cout << "Violation of antisymmetry: " << std::endl;
	std::cout << LTxyz.idp << " " << LTxyz.idq << " " << LTxyz.idr << " " << LTxyz.result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << std::endl;
#endif
	return -1; }

      // nondegeneracy
      if (Decisions[i].idp == Decisions[j].idp &&
	  Decisions[i].idq == Decisions[j].idr &&
	  Decisions[i].idr == Decisions[j].idq &&
	  Decisions[i].result == false &&
	  Decisions[j].result == false) {
#ifdef __VERBOSE
	std::cout << "Violation of nondegeneracy: " << std::endl;
	std::cout << Decisions[i].idp << " " << Decisions[i].idq << " " << Decisions[i].idr << " " << Decisions[i].result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << std::endl;
#endif
	return -1; }
      if (Decisions[i].idp == Decisions[j].idr &&
	  Decisions[i].idq == Decisions[j].idq &&
	  Decisions[i].idr == Decisions[j].idp &&
	  Decisions[i].result == false &&
	  Decisions[j].result == false) {
#ifdef __VERBOSE
	std::cout << "Violation of nondegeneracy: " << std::endl;
	std::cout << Decisions[i].idp << " " << Decisions[i].idq << " " << Decisions[i].idr << " " << Decisions[i].result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << std::endl;
#endif
	return -1; }
      if (Decisions[i].idp == Decisions[j].idq &&
	  Decisions[i].idq == Decisions[j].idp &&
	  Decisions[i].idr == Decisions[j].idr &&
	  Decisions[i].result == false &&
	  Decisions[j].result == false) {
#ifdef __VERBOSE
	std::cout << "Violation of nondegeneracy: " << std::endl;
	std::cout << Decisions[i].idp << " " << Decisions[i].idq << " " << Decisions[i].idr << " " << Decisions[i].result << "  vs  " << Decisions[j].idp << " " << Decisions[j].idq << " " << Decisions[j].idr << " " << Decisions[j].result << std::endl;
#endif
	return -1; }
    }
  }

  return 1;
}
