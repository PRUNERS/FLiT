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

#ifndef SIMPLE_CONVEX_HULL_H
#define SIMPLE_CONVEX_HULL_H 0

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include <vector>
#include "s3fp_utils.h"

#ifndef IFT
#define IFT float
#endif

#ifndef OFT
#define OFT long double
#endif

#define WFT float

struct Point {
  long id;
  IFT x;
  IFT y;
};

typedef std::pair<Point, Point> Edge;

size_t getEdgeCount();

struct PrimitiveCall {
  long idp;
  long idq;
  long idr;
  bool result;
};

/*
   test the orientation of point r wrt to line p -> q
   return 1 for p -> q -> r is counter clock-wise
   return 0 for p -> q -> r is collinear
   return -1 for p -> q -> r is clock-wise

   if outfile == NULL, the determinant will not be recorded
*/
template<typename OWFT>
int Orientation (Point p, Point q, Point r, OWFT &ret_det) {
  OWFT px = (OWFT) p.x;
  OWFT py = (OWFT) p.y;
  OWFT qx = (OWFT) q.x;
  OWFT qy = (OWFT) q.y;
  OWFT rx = (OWFT) r.x;
  OWFT ry = (OWFT) r.y;

  OWFT det = ((qx - px) * (ry - py)) - ((qy - py) * (rx - px));

  ret_det = det;

  if (det > 0) return 1;
  else if (det < 0) return -1;
  else return 0;
}

bool IsOnPQ (Point p, Point q, Point r);

/*
  case cano_selection = 1:

  sort by point id

  ----

  case cano_selection = 2:

  sort by point's geometric position
*/
void Canonical_Unordered_Points (std::vector<Point>& pv, int cano_selection);
/*
  case: cano_selection = 1

  Let the new head point has id h_id. The two adjacent points has ids, left_id and right_id.

  h_id is the smallest id among all points.
  Also, the list is rotated by direction h_id -> left_id if left_id < right_id.
  Otherwise, the list is rotated by direction h_id -> right_id if right_id < left_id.

  ----

  case cano_selection = 2

  Let the new head point has the hightest y coordinate and is the left-most.

  The new list is in an counter clock-wise order

*/
template <typename OWFT>
void Canonical_Ordered_Points (std::vector<Point>& pv, int cano_selection) {
  assert(1 <= cano_selection && cano_selection <= 2);
  assert(pv.size() >= 3);

  int nh_index = 0;
  std::vector<Point> temp_pv;

  // find the new head
  for (int i = 1 ; i < pv.size() ; i++) {
    assert(pv[nh_index].id != pv[i].id);

    if (cano_selection == 1) {
      if (pv[nh_index].id > pv[i].id) nh_index = i;
    }
    else if (cano_selection == 2) {
      if (pv[nh_index].y < pv[i].y) nh_index = i;
      else if (pv[nh_index].y == pv[i].y) {
	if (pv[nh_index].x > pv[i].x) nh_index = i;
	else {;}
      }
      else {;}
    }
    else assert(false);
  }

  // find left and right indexes
  int left_index = ((nh_index == 0) ? (pv.size() - 1) : (nh_index - 1));
  int right_index = (nh_index + 1) % pv.size();
  assert(0 <= left_index && left_index < pv.size());
  assert(0 <= right_index && right_index < pv.size());

  // decide rotation direciton
  bool to_left;
  if (cano_selection == 1) {
    int left_id = pv[left_index].id;
    int right_id = pv[right_index].id;
    assert(pv[nh_index].id != left_id);
    assert(pv[nh_index].id != right_id);
    assert(left_id != right_id);

    if (left_id < right_id) to_left = true;
    else to_left = false;

    // this is hacking!
    to_left = false;
  }
  else if (cano_selection == 2) {
    OWFT ret_det;
    int ori = Orientation<OWFT> (pv[right_index],
				 pv[nh_index],
				 pv[left_index],
				 ret_det);
    if (ori == 1) to_left = true;
    else if (ori == 0) {
      OWFT inner =
	((((OWFT)pv[nh_index].x) - ((OWFT)pv[right_index].x)) *
	 (((OWFT)pv[left_index].x) - ((OWFT)pv[right_index].x))) +
	((((OWFT)pv[nh_index].y) - ((OWFT)pv[right_index].y)) *
	 (((OWFT)pv[left_index].y) - ((OWFT)pv[right_index].y)));
      if (inner >= 0) to_left = true;
      else to_left = false;
    }
    else if (ori == -1) to_left = false;
    else assert(false);
  }
  else assert(false);

  // reinstall the point list
  temp_pv.push_back(pv[nh_index]);
  for (int i = 1 ; i < pv.size() ; i++) {
    int next_index;
    if (to_left) {
      next_index = nh_index - i;
      if (next_index < 0) next_index += pv.size();
    }
    else next_index = (nh_index + i) % pv.size();

    temp_pv.push_back(pv[next_index]);
  }

  pv.clear();
  pv.insert(pv.begin(), temp_pv.begin(), temp_pv.end());
}

void PrintPoint(Point p);

void PrintPV (std::vector<Point> pv);

void WRONG_CONVEX_HULL (FILE *outfile);

void ReadInputs (FILE *infile);

extern std::vector<Point> PointList;
extern std::vector<Edge> CHullEdges;
extern std::vector<PrimitiveCall> Decisions;

/*
  Simple Convex Hull
 */
template <typename T = WFT>
void SimpleComputeConvexhull () { // (FILE *outfile) {
  //  assert(outfile != NULL);

  std::vector<Edge> chedges;

  for (size_t p = 0 ; p < PointList.size() ; p++) {
    for (size_t q = 0 ; q < PointList.size() ; q++) {
      if (PointList[p].id == PointList[q].id) continue;
      size_t r;
      for (r = 0 ; r < PointList.size() ; r++) {
	if (PointList[p].id == PointList[r].id ||
	    PointList[q].id == PointList[r].id) continue;
	T det;
	int ret = Orientation<T> (PointList[p],
				    PointList[q],
				    PointList[r],
				    det);
	bool decision;
	if (ret == 1) decision = true; // continue;
	else if (ret == 0) {
	  if (IsOnPQ(PointList[p], PointList[q], PointList[r])) decision = true; // continue;
	  else decision = false; // break;
	}
	else if (ret == -1) decision = false; // break;
	else assert(false);

	struct PrimitiveCall acall;
	acall.idp = PointList[p].id;
	acall.idq = PointList[q].id;
	acall.idr = PointList[r].id;
	acall.result = decision;
	Decisions.push_back(acall);

	if (decision) continue;
	else break;
      }
      if (r == PointList.size())
	chedges.push_back(Edge(PointList[p], PointList[q]));
    }
  }

  // connecting edges
  assert(chedges.size() >= 3);
  CHullEdges.push_back(chedges[0]);
  Edge first_edge = CHullEdges[0];

  while (true) {
    Edge last_edge = CHullEdges[CHullEdges.size()-1];
    if (last_edge.second.id == first_edge.first.id) break;
    size_t e;
    for (e = 0 ; e < chedges.size() ; e++) {
      if (last_edge.second.id == chedges[e].first.id) {
	CHullEdges.push_back(chedges[e]);
	break;
      }
    }
    if (e == chedges.size()) // bad convex hull... just stop
      break;
    if (CHullEdges.size() >= PointList.size())
      break;
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

/*
  Gift Wrapping Convex Hull
 */
void GiftWrappingComputeConvexhull (FILE *outfile);

/*
   verify_method == 0 : no verification
   verify_method == 1 : check correctness
   verify_method == 2 : check consistency
*/
void VerifyHull(FILE *outfile);

/*
  return 1 for passing consistency check
  return -1 for failing consistency check
 */
OFT CheckConsistency ();

#endif // SIMPLE_CONVEX_HULL_H
