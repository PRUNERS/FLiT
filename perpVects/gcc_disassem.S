Dump of assembler code for function FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t):
145	    result = *u;
   0x0000000000410db0 <+672>:	49 63 c4	movslq %r12d,%rax
   0x0000000000410db3 <+675>:	48 03 44 24 30	add    0x30(%rsp),%rax
   0x00000000004114ea <+2522>:	f2 0f 10 04 e8	movsd  (%rax,%rbp,8),%xmm0
   0x00000000004114ef <+2527>:	48 8b 84 24 b0 00 00 00	mov    0xb0(%rsp),%rax
   0x00000000004114f7 <+2535>:	f2 0f 59 04 e8	mulsd  (%rax,%rbp,8),%xmm0
   0x00000000004114fc <+2540>:	66 48 0f 7e c3	movq   %xmm0,%rbx

146	  }
147	  
148	  template<typename F>
149	  static typename get_corresponding_type<F>::type
150	  projectType(F const &source){
151	    using I = typename get_corresponding_type<F>::type;
152	    F temp = source;
153	    I* u = reinterpret_cast<I*>(&temp);
154	    if( sizeof(F) == 16 && std::is_floating_point<F>::value){ //we must be using long double, which is 80 bit extended -- mask out higher bits
155	      unsigned __int128 zero = 0;
156	      *u = *u & (~zero >> 48);
157	    }
158	    return *u;
159	  }
160	
161	  //the first normalized number > 0 (the smallest positive) -- can be obtained from <float>
162	  //[pos][bias + 1][000...0]
163	  template<typename T>
164	  static T
165	  getFirstNorm(){
166	    T retVal;
167	    typedef typename get_corresponding_type<T>::type t;
168	    t val;
169	    switch(sizeof(T)){
170	    case 4:
171	      {
172		val = (t)1 << mantBitWidth32;
173	      }
174	      break;
175	    case 8:
176	      {
177		val = (t)1 << mantBitWidth64;
178		projectType(val, retVal);
179	      }
180	      break;
181	    case 16:
182	      {
183		val = (t) 1 << mantBitWidth80;
184	      }
185	      break;
186	    default:
187	      //won't compile
188	      int x = 0;
189	    }
190	    projectType(val, retVal);
191	    return retVal;
192	  }
193	
194	  template<typename T>
195	  static T
196	  perturbFP(T const &src, unsigned offset){ //negative offset with 0 may produce NAN
197	    //    T *ncSrc = const_cast<T*>(&src);
198	    T ncSrc = src;
199	    typename get_corresponding_type<T>::type tmp;
200	    if(NO_SUBNORMALS && ncSrc == 0) ncSrc = getFirstNorm<T>();
201	    T retVal = perturbFPTyped(ncSrc, tmp, offset);
202	    return retVal;
203	  }
204	
205	  template<typename T, typename U>
206	  static T
207	  perturbFPTyped(T &src, U &tmp, int offset){
208	    T retVal;
209	    projectType(src, tmp);
210	    tmp += offset;
211	    projectType(tmp, retVal);
212	    return retVal;
213	  }
214	
215	  //returns the exponent portion of floating point
216	  template<typename T>
217	  static unsigned
218	  getExponent(T v){
219	    unsigned retVal = -1;
220	    typename get_corresponding_type<T>::type val;
221	    projectType(v, val);
222	    switch(sizeof(v)){
223	    case 4:
224	      {
225		retVal = ((val >> (32 - expBitWidth32 - 1) & 0x7F) - bias32);
226	      }
227	      break;
228	    case 8:
229	      {
230		retVal = ((val >> (64 - expBitWidth64 - 1) & 0x7FF) - bias64); 
   0x0000000000411501 <+2545>:	48 c1 eb 34	shr    $0x34,%rbx
   0x0000000000411505 <+2549>:	81 e3 ff 07 00 00	and    $0x7ff,%ebx
   0x000000000041150b <+2555>:	81 eb ff 03 00 00	sub    $0x3ff,%ebx

231	      }
232	      break;
233	    case 16:
234	      {
235		retVal = ((val >> (80 - expBitWidth80 - 1) & 0x7FFF) - bias80);
236	      }
237	      break;
238	    default:
239	      retVal = 0;
240	    }
241	    return retVal;
242	  }
243	
244	
245	};
246	
247	template<typename F>
248	struct FPWrap{
249	  using I = typename get_corresponding_type<F>::type;
250	  F const &floatVal;
251	  I intVal;
252	  void
253	  update() const {
254	    FPHelpers::projectType(floatVal, const_cast<FPWrap*>(this)->intVal);
255	  }
256	
257	  FPWrap(F const &val):floatVal(val){}
   0x0000000000410f65 <+1109>:	48 8d 84 24 f0 00 00 00	lea    0xf0(%rsp),%rax
   0x0000000000410f80 <+1136>:	48 89 84 24 50 01 00 00	mov    %rax,0x150(%rsp)
   0x0000000000410f97 <+1159>:	48 89 b4 24 30 01 00 00	mov    %rsi,0x130(%rsp)
   0x0000000000410fa8 <+1176>:	48 89 8c 24 10 01 00 00	mov    %rcx,0x110(%rsp)

258	  template<typename U>
259	  friend ostream& operator<<(ostream& os, FPWrap<U> const &w);
260	};
261	
262	template <typename U>
263	ostream& operator<<(ostream& os, const FPWrap<U> &w){
264	  w.update();
265	  std::ios_base::fmtflags f = os.flags();
266	  //FIXME can't handle 128 bit values for ostream operations
267	  os << std::hex << w.intVal;
268	  os.flags(f);
269	  return os;
270	}
271	
272	//keeping with the traditional mathematics index
273	//ordering [row][column]
274	template<typename T>
275	class Matrix;
276	
277	std::default_random_engine gen;
278	
279	template <typename T>
280	class Vector {
281	  std::vector<T> data;
282	public:
283	  Vector():data(0){}
284	  Vector(std::initializer_list<T> l):data(l){}
285	  Vector(size_t size):data(size, 0){}
286	  template <typename U>
287	  Vector(size_t size, U genFun):data(size, 0){
   0x0000000000410b8e <+126>:	48 8d 8c 24 87 00 00 00	lea    0x87(%rsp),%rcx
   0x0000000000410b96 <+134>:	48 8d 94 24 88 00 00 00	lea    0x88(%rsp),%rdx
   0x0000000000410b9e <+142>:	48 8d bc 24 90 00 00 00	lea    0x90(%rsp),%rdi
   0x0000000000410ba6 <+150>:	48 89 ee	mov    %rbp,%rsi
   0x0000000000410ba9 <+153>:	48 c7 84 24 88 00 00 00 00 00 00 00	movq   $0x0,0x88(%rsp)
   0x0000000000410bb5 <+165>:	e8 16 a0 ff ff	callq  0x40abd0 <std::vector<double, std::allocator<double> >::vector(unsigned long, double const&, std::allocator<double> const&)>

288	    std::generate(data.begin(), data.end(), genFun);
289	  }
290	  Vector(const Vector &v):data(v.data) {}
291	  size_t
292	  size() const {return data.size();}
293	
294	  T&
295	  operator[](size_t indx){return data[indx];}
296	
297	  const T&
298	  operator[](size_t indx) const {return data[indx];}
299	
300	  Vector<T>
301	  operator*(T const &rhs){
302	    Vector<T> retVal(size());
303	    for(int x = 0; x < size(); ++x){
304	      retVal[x] = data[x] * rhs;
305	    }
306	    return retVal;
307	  }
308	
309	
310	  static
311	  Vector<T>
312	  getRandomVector(size_t dim, T min_inc, T max_exc,
313			  std::default_random_engine::result_type seed = 0,
314			  bool doSeed = false){
315	    if(doSeed) gen.seed(seed);
316	    std::uniform_real_distribution<T> dist(min_inc,max_exc);
317	    Vector<T> tmp(dim);
318	    for(auto& i: tmp.data){
319	      i = dist(gen);
320	    }
321	    return tmp;
322	  }
323	
324	  Vector<T>
325	  getUnitVector() const {
326	    Vector<T> retVal(*this);
327	    return retVal * ((T)1.0 / (this->L2Norm()));
328	  }
329	
330	  bool
331	  operator==(Vector<T> const &b){
332	    bool retVal = true;
333	    for(int x = 0; x < size(); ++x){
334	      if(data[x] != b.data[x]){
335		retVal = false;
336		break;
337	      }
338	    }
339	    return retVal;
340	  }
341	
342	  void
343	  dumpDistanceMetrics(Vector<T> const &b, ostream& os){
344	    os << "vector difference: \t" << ((*this) - b) << endl;
345	    //    os << "BitDistance: \t" << (*this).BitDistance(b) << endl;
346	    os << "l1distance:  \t" << (*this).L1Distance(b) << endl;
347	  }
348	
349	  //creates a vector with randomly swapped elements, every other
350	  //one negated.  For odd sized vectors, the odd one out will
351	  //be zero
352	  //i.e.
353	  // 1  2 3  4
354	  // may produce
355	  // 2 -1 4 -3
356	  //or
357	  // 1 2  3  4 5
358	  // may produce
359	  // 3 4 -1 -2 0
360	  Vector<T>
361	  genOrthoVector(){
362	    Vector<T> retVal(size());
363	    std::vector<T> seq(size());
364	    iota(seq.begin(), seq.end(), 0); //load with seq beg w 0
365	    shuffle(seq.begin(), seq.end(), default_random_engine(RAND_SEED));
366	    //do pairwise swap
367	    for(int i = 0; i < size(); i += 2){
368	      retVal[seq[i]] = data[seq[i+1]];
369	      retVal[seq[i+1]] = -data[seq[i]];
370	    }
371	    if(size() & 1) //odd
372	      retVal[seq[size() - 1]] = 0;
373	    return retVal;
374	  }
375	
376	  T
377	  getLInfNorm(){
378	    T retVal;
379	    std::pair<int, T> largest;
380	    for(int x = 0; x < data.size(); ++x){
381	      T abe = fabs(data[x]);
382	      if(abe > largest.second){
383		largest.first = x;
384		largest.second = abe;
385	      }
386	    }
387	  }
388	
389	  //returns the angle between vectors using cos^-1 and dot
390	  // T
391	  // getTransAngle(Vector<T> const &other){
392	  //   //return acos(
393	  // }
394	
395	  void
396	  rotateAboutZ_3d(T rads){
397	    Matrix<T> t = {{cos(rads), -sin(rads), 0},
398			{sin(rads), cos(rads), 0},
399			{0, 0, 1}};
400	    info_stream << "rotation matrix is: " << t << endl;
401	    Vector<T> tmp(*this);
402	    tmp = t * tmp;
403	    info_stream << "in rotateAboutZ, result is: " << tmp << endl;
404	    data = tmp.data;
405	    
406	  }
407	
408	  Vector<T>
409	  operator-(Vector<T> const &rhs) const {
410	    Vector<T> retVal(size());
411	    for(int x = 0; x < size(); ++x){
412	      retVal[x] = data[x] - rhs.data[x];
413	    }
414	    return retVal;
415	  }
416	
417	  unsigned __int128
418	  BitDistance(Vector<T> const &rhs) const {
419	    typedef unsigned __int128 itype;
420	    itype retVal;
421	    for(int i = 0; i < size(); ++i){
422	      retVal += std::labs((itype)FPWrap<long double>((long double)data[i]).intVal - FPWrap<T>((long double)rhs.data[i]).intVal);
423	    }
424	    return retVal;
425	  }
426	
427	  //may be more useful than L2 as there is no square/sqrt
428	  //op to lose bits
429	  T
430	  L1Distance(Vector<T> const &rhs) const {
431	    T distance = 0;
432	    for(int x = 0; x < size(); ++x){
433	      distance += fabs(data[x] - rhs.data[x]);
434	    }
435	    return distance;
436	  }
437	
438	  //method to reduce vector (pre-sort)
439	  enum sort_t{
440	    lt, //manual lt magnitude sort
441	    gt, //manual gt magnitude sort
442	    bi, //built-in (std::inner_product) [only for ^, assumes def otherwise]
443	    def //default (manual unsorted)
444	  };
445	
446	private:
447	  sort_t sortType = def;
448	public:
449	  template<class U>
450	  friend ostream& operator<<(ostream& os, Vector<U> const &a);
451	
452	  //fun is lamda like: [&sum](T a){sum += a};
453	  //you provide the T sum, closure will capture
454	  //not const because it may sort cont
455	  //C is container type
456	
457	  template<typename F, class C>
458	  void
459	  reduce(C &cont, F const &fun) const {
460	    printOnce(__FUNCTION__, &cont.data()[0]);
461	
462	    T retVal;
463	    if(sortType == lt || sortType == gt){
464	      if(sortType == lt)
465		std::sort(cont.begin(), cont.end(),
466			  [](T a, T b){return fabs(a) < fabs(b);});
467	      else
468		std::sort(cont.begin(), cont.end(),
469			  [](T a, T b){return fabs(a) > fabs(b);});
470	    }
471	    for_each(cont.begin(), cont.end(), fun);
472	  }
473	
474	  void
475	  setSort(sort_t st = def){sortType = st;}
   0x0000000000410c12 <+258>:	8b 44 24 7c	mov    0x7c(%rsp),%eax
   0x0000000000410c26 <+278>:	89 84 24 a8 00 00 00	mov    %eax,0xa8(%rsp)

476	  //inner product (dot product)
477	  //TODO again, for icpc, re-enable optimizations
478	#pragma optimize("", on)
479	  T
480	  operator^(Vector<T> const &rhs) const {
481	    //TODO using a simpler watch
482	    //Globals<T>::sum = 0.0;
483	    //T VOLATILE &sum = Globals<T>::sum;
484	    T sum = 0.0;
485	    auto &prods = Globals<T>::prods;
486	    printOnce(__FUNCTION__, &prods.data()[0]);
487	    T temp = 0.0;
   0x0000000000411220 <+1808>:	66 0f ef c0	pxor   %xmm0,%xmm0
   0x0000000000411224 <+1812>:	e9 3c fd ff ff	jmpq   0x410f65 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1109>
   0x0000000000411229 <+1817>:	48 b8 00 00 00 00 00 00 20 00	movabs $0x20000000000000,%rax
   0x0000000000411233 <+1827>:	48 89 44 24 30	mov    %rax,0x30(%rsp)
   0x0000000000411238 <+1832>:	e9 60 fb ff ff	jmpq   0x410d9d <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+653>

488	    if( sortType == bi){
   0x0000000000410e0c <+764>:	83 bc 24 a8 00 00 00 02	cmpl   $0x2,0xa8(%rsp) #if built-in sort
   0x0000000000410e14 <+772>:	0f 84 6e 03 00 00	je     0x411188 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1656>
   0x0000000000410e1a <+778>:	48 8b bc 24 90 00 00 00	mov    0x90(%rsp),%rdi
   0x0000000000410ee8 <+984>:	83 bc 24 a8 00 00 00 02	cmpl   $0x2,0xa8(%rsp)
   0x0000000000410ef0 <+992>:	0f 84 1a 03 00 00	je     0x411210 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1792>
   0x0000000000410ef6 <+998>:	48 8b 8c 24 90 00 00 00	mov    0x90(%rsp),%rcx

489	      sum = std::inner_product(data.begin(), data.end(), rhs.data.begin(), (T)0.0);
490	    }else{
491	      std::for_each(prods.begin(), prods.end(), [&](T i){i = 0.0;});
492	      for(int i = 0; i < size(); ++i){ 
   0x0000000000410e3b <+811>:	48 89 c8	mov    %rcx,%rax
   0x0000000000410e3e <+814>:	48 c1 f8 03	sar    $0x3,%rax
   0x0000000000410e42 <+818>:	48 85 c0	test   %rax,%rax
   0x0000000000410e45 <+821>:	74 2a	je     0x410e71 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+865>
   0x0000000000410e47 <+823>:	4c 8b 84 24 b0 00 00 00	mov    0xb0(%rsp),%r8
   0x0000000000410e4f <+831>:	31 c0	xor    %eax,%eax
   0x0000000000410e51 <+833>:	0f 1f 80 00 00 00 00	nopl   0x0(%rax)
   0x0000000000410e6c <+860>:	48 39 c8	cmp    %rcx,%rax
   0x0000000000410e6f <+863>:	75 e7	jne    0x410e58 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+840>
   0x0000000000410f1f <+1039>:	48 89 f8	mov    %rdi,%rax
   0x0000000000410f22 <+1042>:	48 c1 f8 03	sar    $0x3,%rax
   0x0000000000410f26 <+1046>:	48 85 c0	test   %rax,%rax
   0x0000000000410f29 <+1049>:	74 1d	je     0x410f48 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1080>
   0x0000000000410f2b <+1051>:	31 c0	xor    %eax,%eax
   0x0000000000410f2d <+1053>:	0f 1f 00	nopl   (%rax)
   0x0000000000410f43 <+1075>:	48 39 f8	cmp    %rdi,%rax
   0x0000000000410f46 <+1078>:	75 e8	jne    0x410f30 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1056>

493		prods[i] = data[i] * rhs.data[i]; 
   0x0000000000410e58 <+840>:	f2 0f 10 04 07	movsd  (%rdi,%rax,1),%xmm0
   0x0000000000410e5d <+845>:	f2 41 0f 59 04 00	mulsd  (%r8,%rax,1),%xmm0
   0x0000000000410e63 <+851>:	f2 0f 11 04 02	movsd  %xmm0,(%rdx,%rax,1)
   0x0000000000410e68 <+856>:	48 83 c0 08	add    $0x8,%rax
   0x0000000000410f30 <+1056>:	f2 0f 10 04 01	movsd  (%rcx,%rax,1),%xmm0
   0x0000000000410f35 <+1061>:	f2 0f 59 04 06	mulsd  (%rsi,%rax,1),%xmm0
=> 0x0000000000410f3a <+1066>:	f2 0f 11 04 02	movsd  %xmm0,(%rdx,%rax,1)
   0x0000000000410f3f <+1071>:	48 83 c0 08	add    $0x8,%rax

494	      }
495	      //      reduce(prods, [&sum](T e){sum += e;});
496	      for(auto j:prods){
   0x0000000000410e71 <+865>:	48 39 d6	cmp    %rdx,%rsi
   0x0000000000410e74 <+868>:	48 89 d0	mov    %rdx,%rax
   0x0000000000410e77 <+871>:	66 0f ef c0	pxor   %xmm0,%xmm0
   0x0000000000410e7b <+875>:	0f 84 07 03 00 00	je     0x411188 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1656>
   0x0000000000410e81 <+881>:	0f 1f 80 00 00 00 00	nopl   0x0(%rax)
   0x0000000000410e90 <+896>:	48 39 c6	cmp    %rax,%rsi
   0x0000000000410e93 <+899>:	75 f3	jne    0x410e88 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+888>
   0x0000000000410f48 <+1080>:	49 39 d0	cmp    %rdx,%r8
   0x0000000000410f4b <+1083>:	48 89 d0	mov    %rdx,%rax
   0x0000000000410f4e <+1086>:	66 0f ef c0	pxor   %xmm0,%xmm0
   0x0000000000410f52 <+1090>:	74 11	je     0x410f65 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1109>
   0x0000000000410f54 <+1092>:	0f 1f 40 00	nopl   0x0(%rax)
   0x0000000000410f60 <+1104>:	49 39 c0	cmp    %rax,%r8
   0x0000000000410f63 <+1107>:	75 f3	jne    0x410f58 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1096>

497		temp += j;
   0x0000000000410e88 <+888>:	f2 0f 58 00	addsd  (%rax),%xmm0
   0x0000000000410e8c <+892>:	48 83 c0 08	add    $0x8,%rax
   0x0000000000410f58 <+1096>:	f2 0f 58 00	addsd  (%rax),%xmm0
   0x0000000000410f5c <+1100>:	48 83 c0 08	add    $0x8,%rax

498	      }
499	    }
500	    return (sum = temp);
501	  }
502	
503	  T
504	  LInfNorm() const {
505	    T retVal = 0;
506	    for(auto e: data){
507	      T tmp = fabs(e);
508	      if( tmp > retVal) retVal = tmp;
509	    }
510	    return retVal;
511	  }
512	
513	  T LInfDistance(Vector<T> const &rhs) const {
514	    T retVal;
515	    auto diff = operator-(rhs);
516	    return diff.LInfNorm();
517	  }
518	
519	  //L2 norm
520	  T
521	  L2Norm() const {
522	    Vector<T> squares(size());
523	    T retVal = 0;
524	    vector<T> prods(data); //make a copy of our data for possible sorted reduction
525	    //will sort reduction if sort
526	    reduce(prods, [&retVal](T e){retVal += e*e;});
527	    return std::sqrt(retVal);
528	  }
529	
530	  T
531	  L2Distance(Vector<T> const &rhs) const {
532	    T retVal = 0;
533	    auto diff = operator-(rhs);
534	    //sorts pre-rediction for sortType = gt|lt
535	    reduce(diff.data, [&retVal](T e){retVal += e*e;});
536	    return std::sqrt(retVal);
537	  }
538	 
539	  //cross product, only defined here in 3d
540	  Vector<T>
541	  cross(Vector<T> const &rhs) const {
542	    Vector<T> retVal(size());
543	    retVal.data[0] = data[1] * rhs.data[2] - rhs.data[1] * data[2];
544	    retVal.data[1] = rhs.data[0] * data[2] - data[0] * rhs.data[2];
545	    retVal.data[2] = data[0] * rhs.data[1] - rhs.data[0] * data[1];
546	    return retVal;
547	  }
548	  
549	  Vector<T>
550	  operator*(Vector<T> const &rhs) const {
551	    Vector<T> retVal(size());
552	    for(int x = 0; x < size(); ++x){
553	      retVal[x] = data[x] * rhs.data[x];
554	    }
555	    return retVal;
556	  }
557	
558	  bool
559	  isOrtho(Vector<T> const &rhs){
560	    return operator^(rhs) == (T)0;
561	  }
562	
563	};
564	
565	template <typename T>
566	ostream& operator<<(ostream& os, Vector<T> const &a){
567	  for(auto i: a.data){
568	    os << i << '\t';
569	  }
570	  return os;
571	}
572	
573	template <typename T>
574	ostream& operator<<(ostream& os, Matrix<T> const &m){
575	  for(auto r: m.data){
576	    for(auto i: r){
577	      os << i << '\t';
578	    }
579	    os << endl;
580	  }
581	  return os;
582	}
583	  
584	template<typename T>
585	class Matrix {
586	  vector<vector<T>> data;
587	public:
588	  Matrix(unsigned rows, unsigned cols):
589	    data(rows, vector<T>(cols, 0)){}
590	  Matrix(Matrix<T> const &m):data(m.data){}
591	  Matrix(std::initializer_list<std::initializer_list<T>> l):
592	    data(l.size(), vector<T>(l.begin()->size())){
593	    int x = 0; int y = 0;
594	    for(auto r: l){
595	      for(auto i: r){
596		data[x][y] = i;
597		++y;
598	      }
599	      ++x; y = 0;
600	    }
601	  }
602	  
603	  friend class Vector<T>;
604	  template<class U>
605	  friend ostream& operator<<(ostream& os, Matrix<U> const &a);
606	
607	
608	  bool
609	  operator==(Matrix<T> const &rhs){
610	    bool retVal = true;
611	    for(int x = 0; x < data.size(); ++x){
612	      for(int y = 0; y < data[0].size(); ++y){
613		if(data[x][y] != rhs.data[x][y]){
614		  info_stream << "in: " << __func__ << endl;
615		  info_stream << "for x,y: " << x << ":" << y << endl;
616		  info_stream << "this = " << data[x][y] << "; rhs = " << rhs.data[x][y] << endl;
617		  retVal = false;
618		  break;
619		}
620	      }
621	    }
622	    return retVal;
623	  }
624	
625	  Matrix<T>
626	  operator*(T const &sca){
627	    Matrix<T> retVal(data.size(), data[0].size());
628	    for(int x = 0; x < data.size(); ++x){
629	      for(int y =0; y < data[0].size(); ++y){
630		retVal.data[x][y] = data[x][y] * sca;
631	      }
632	    }
633	    return retVal;
634	  }
635	  
636	  //precond: this.w = rhs.h, duh
637	  Matrix<T>
638	  operator*(Matrix<T> const &rhs){
639	    Matrix<T> retVal(data.size(), rhs.data[0].size());
640	    for(int bcol = 0; bcol < rhs.data[0].size(); ++bcol){
641	      for(int x = 0; x < data.size(); ++x){
642		for(int y = 0; y < data[0].size(); ++y){
643		  retVal.data[x][bcol] += data[x][y] * rhs.data[y][bcol];
644		}
645	      }
646	    }
647	    return retVal;
648	  }
649	
650	  //precond: dim(v) == 3
651	  static
652	  Matrix<T>
653	  SkewSymCrossProdM(Vector<T> const &v){
654	    return Matrix<T>(
655	      {{0, -v[2], v[1]},
656		  {v[2], 0, -v[0]},
657		    {-v[1], v[0], 0}});
658	  }
659	
660	  static
661	  Matrix<T>
662	  Identity(size_t dims){
663	    Matrix<T> retVal(dims, dims);
664	    for(int x = 0; x < dims; ++x){
665	      for(int y =0; y < dims; ++y){
666		if(x == y) retVal.data[x][y] = 1;
667		else retVal.data[x][y] = 0;
668	      }
669	    }
670	    return retVal;
671	  }
672	  
673	  Vector<T>
674	  operator*(Vector<T> const &v) const {
675	    info_stream << "in Matrix multiply operator with matrix:" << endl;
676	    info_stream << (*this);
677	    Vector<T> retVal(data.size());
678	    int resI = 0;
679	    for(auto row: data){
680	      for(size_t i = 0; i < row.size(); ++i){
681		retVal[resI] += row[i] * v[i];
682	      }
683	      ++resI;
684	    }
685	    return retVal;
686	  }
687	
688	  Matrix<T>
689	  operator+(Matrix<T> const&rhs) const{
690	    Matrix<T> retVal(rhs);
691	    int x = 0; int y = 0;
692	    for(auto r: data){
693	      for(auto i: r){
694		retVal.data[x][y] = i + rhs.data[x][y];
695		++y;
696	      }
697	      y = 0; ++x;
698	    }
699	    return retVal;
700	  }
701	
702	  void
703	  print() const {
704	    cout << *this;
705	  }
706	};
707	
708	
709	struct FPTests {
710	
711	  //each test returns a score, the closer to 0 the better
712	
713	  //the next two tests are from Prof Hari Sundar's blog:
714	  //https://fgiesen.wordpress.com/2013/06/02/modified-gram-schmidt-orthogonalization/
715	  //we'll follow along and choose the same (relative) epsilons, shoot for same results
716	  typedef std::pair<string, std::pair<long double, long double>> resultType;
717	
718	  template<typename T>
719	  static
720	  resultType
721	  // static pair<string, long double>
722	  DoHariGSBasic(){
723	    long double score = 0.0;
724	    T e;
725	    sizeof(T) == 4 ? e = pow(10, -4) : sizeof(T) == 8 ? e = pow(10, -8) : e = pow(10, -10);
726	    //matrix = {a, b, c};
727	    Vector<T> a = {1, e, e};
728	    Vector<T> b = {1, e, 0};
729	    Vector<T> c = {1, 0, e};
730	    auto r1 = a.getUnitVector();
731	    auto r2 = (b - r1 * (b ^ r1)).getUnitVector();
732	    auto r3 = (c - r1 * (c ^ r1) -
733		       r2 * (c ^ r2)).getUnitVector();
734	    T o12 = r1 ^ r2;
735	    T o13 = r1 ^ r3;
736	    T o23 = r2 ^ r3;
737	    if((score = fabs(o12) + fabs(o13) + fabs(o23)) != 0){
738	      info_stream << "in: " << __func__ << endl;
739	      info_stream << "applied gram-schmidt to:" << endl;
740	      info_stream << "a: " << a << endl;
741	      info_stream << "b: " << b << endl;
742	      info_stream << "c: " << c << endl;
743	      info_stream << "resulting vectors were: " << endl;
744	      info_stream << "r1: " << r1 << endl;
745	      info_stream << "r2: " << r2 << endl;
746	      info_stream << "r3: " << r3 << endl;
747	      info_stream << "w dot prods: " << o12 << ", " << o13 << ", " << o23 << endl;
748	      info_stream << "score (bits): " << FPWrap<long double>(score) << endl;
749	      info_stream << "score (dec) :" << score << endl;
750	    }
751	    return {__func__, {score, 0.0}};
752	  }
753	
754	
755	  //Hari Sundar's improved Gram-Schmidt
756	  template<typename T>
757	  static resultType
758	  //static pair<string, long double>
759	  DoHariGSImproved(){
760	    long double score = 0.0;
761	    T e;
762	    sizeof(T) == 4 ? e = pow(10, -4) : sizeof(T) == 8 ? e = pow(10, -8) : e = pow(10, -10);
763	    //matrix = {a, b, c};
764	    Vector<T> a = {1, e, e};
765	    Vector<T> b = {1, e, 0};
766	    Vector<T> c = {1, 0, e};
767	
768	    auto r1 = a.getUnitVector();
769	    auto r2 = (b - r1 * (b ^ r1)).getUnitVector();
770	    auto r3 = (c - r1 * (c ^ r1));
771	    r3 = (r3 - r2 * (r3 ^ r2)).getUnitVector();
772	    T o12 = r1 ^ r2;
773	    T o13 = r1 ^ r3;
774	    T o23 = r2 ^ r3;
775	    if((score = fabs(o12) + fabs(o13) + fabs(o23)) != 0){
776	      info_stream << "in: " << __func__ << endl;
777	      info_stream << "applied gram-schmidt to:" << endl;
778	      info_stream << "a: " << a << endl;
779	      info_stream << "b: " << b << endl;
780	      info_stream << "c: " << c << endl;
781	      info_stream << "resulting vectors were: " << endl;
782	      info_stream << "r1: " << r1 << endl;
783	      info_stream << "r2: " << r2 << endl;
784	      info_stream << "r3: " << r3 << endl;
785	      info_stream << "w dot prods: " << o12 << ", " << o13 << ", " << o23 << endl;
786	    }
787	    return {__func__, {score, 0.0}};
788	  }
789	
790	  //this test takes two random vectors, calculates the required
791	  //rotation matrix for alignment, and then attempts to align them
792	  //the score is the distance between the two after the process
793	  template<typename T>
794	  static resultType
795	  // static pair<string, long double>
796	  DoSkewSymCPRotationTest(const T min, const T max){
797	    info_stream << "entered " << __func__ << endl; 
798	    long double L1Score = 0.0;
799	    long double LIScore = 0.0;
800	    auto A = Vector<T>::getRandomVector(3, min, max).getUnitVector();
801	    info_stream << "A (unit) is: " << endl << A << endl;
802	    auto B = Vector<T>::getRandomVector(3, min, max).getUnitVector();
803	    info_stream << "B (unit): " << endl  << B << endl;
804	    auto cross = A.cross(B); //cross product
805	    info_stream << "cross: " << endl << cross << endl;
806	    auto sine = cross.L2Norm();
807	    info_stream << "sine: " << endl << sine << endl;
808	    auto cosine = A ^ B; //dot product
809	    info_stream << "cosine: " << endl << cosine << endl;
810	    auto sscpm = Matrix<T>::SkewSymCrossProdM(cross);
811	    info_stream << "sscpm: " << endl << sscpm << endl;
812	    auto rMatrix = Matrix<T>::Identity(3) +
813	      sscpm + (sscpm * sscpm) * ((1 - cosine)/(sine * sine));
814	    auto result = rMatrix * A;
815	    info_stream << "rotator: " << endl << rMatrix << endl;
816	    if(!(result == B)){
817	      L1Score = result.L1Distance(B);
818	      LIScore = result.LInfDistance(B);
819	      info_stream << "Skew symmetric cross product rotation failed with ";
820	      info_stream << "L1Distance " << L1Score << endl;
821	      info_stream << "starting vectors: " << endl;
822	      info_stream << A << endl;
823	      info_stream << "...and..." << endl;
824	      info_stream << B << endl;
825	      info_stream << "ended up with: " << endl;
826	      info_stream << "L1Distance: " << L1Score << endl;
827	      info_stream << "LIDistance: " << LIScore << endl;
828	    }
829	    return {__func__, {L1Score, LIScore}};
830	  }
831	
832	  //TODO this is for intel, to see if we can get the order of
833	  //reduction updates to be the same between icpc & g++
834	#pragma optimize("", off)
835	  template<typename T, typename Fun>
836	  static resultType
837	  // static pair<string, long double>
838	  DoOrthoPerturbTest(const int iters, const int dim,
   0x0000000000410b10 <+0>:	41 57	push   %r15
   0x0000000000410b12 <+2>:	41 56	push   %r14
   0x0000000000410b14 <+4>:	41 55	push   %r13
   0x0000000000410b16 <+6>:	41 54	push   %r12
   0x0000000000410b18 <+8>:	55	push   %rbp
   0x0000000000410b19 <+9>:	53	push   %rbx
   0x0000000000410b1a <+10>:	48 63 ea	movslq %edx,%rbp
   0x0000000000410b1d <+13>:	4c 89 c3	mov    %r8,%rbx
   0x0000000000410b20 <+16>:	48 81 ec 78 01 00 00	sub    $0x178,%rsp
   0x0000000000410b2a <+26>:	48 89 7c 24 60	mov    %rdi,0x60(%rsp)
   0x0000000000410b2f <+31>:	89 74 24 20	mov    %esi,0x20(%rsp)
   0x0000000000410b33 <+35>:	89 6c 24 54	mov    %ebp,0x54(%rsp)
   0x0000000000410b37 <+39>:	48 89 4c 24 68	mov    %rcx,0x68(%rsp)
   0x0000000000410b3c <+44>:	44 89 4c 24 7c	mov    %r9d,0x7c(%rsp)

839			     const size_t ulp_inc,
840			     Fun f,
841			     const typename Vector<T>::sort_t
842			     st = Vector<T>::def){
843	    long double score = 0.0;
   0x000000000041126a <+1882>:	d9 ee	fldz   
   0x000000000041126c <+1884>:	66 0f ef ff	pxor   %xmm7,%xmm7
   0x0000000000411270 <+1888>:	db 7c 24 10	fstpt  0x10(%rsp)
   0x0000000000411274 <+1892>:	f3 0f 11 7c 24 78	movss  %xmm7,0x78(%rsp)

844	    std::vector<unsigned> orthoCount(dim, 0.0);
845	    //we use a double literal above as a workaround for Intel 15-16
846	    //compiler bug:
847	    //https://software.intel.com/en-us/forums/intel-c-compiler/topic/565143
848	    size_t indexer = 0;
849	    Vector<T> a(dim, f);
850	    a.setSort(st);
851	    Vector<T> b = a.genOrthoVector();
   0x0000000000410c16 <+262>:	48 8d b4 24 90 00 00 00	lea    0x90(%rsp),%rsi
   0x0000000000410c1e <+270>:	48 8d bc 24 b0 00 00 00	lea    0xb0(%rsp),%rdi
   0x0000000000410c2d <+285>:	e8 de 98 ff ff	callq  0x40a510 <Vector<double>::genOrthoVector()>

852	  
853	    info_stream << "starting dot product orthogonality test with a, b = " << endl;
854	    for(int x = 0; x < dim; ++x) info_stream << x << '\t';
   0x0000000000410c50 <+320>:	8b 54 24 54	mov    0x54(%rsp),%edx
   0x0000000000410c54 <+324>:	31 db	xor    %ebx,%ebx
   0x0000000000410c56 <+326>:	85 d2	test   %edx,%edx
   0x0000000000410c58 <+328>:	7e 32	jle    0x410c8c <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+380>
   0x0000000000410c5a <+330>:	89 de	mov    %ebx,%esi
   0x0000000000410c5c <+332>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x0000000000410c61 <+337>:	e8 9a 14 ff ff	callq  0x402100 <_ZNSolsEi@plt>
   0x0000000000410c83 <+371>:	83 c3 01	add    $0x1,%ebx
   0x0000000000410c86 <+374>:	39 5c 24 54	cmp    %ebx,0x54(%rsp)
   0x0000000000410c8a <+378>:	75 ce	jne    0x410c5a <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+330>

855	    info_stream << endl;
856	    info_stream << a << endl;
   0x0000000000410c96 <+390>:	48 8d b4 24 90 00 00 00	lea    0x90(%rsp),%rsi
   0x0000000000410c9e <+398>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x0000000000410ca3 <+403>:	e8 58 79 ff ff	callq  0x408600 <operator<< <double>(std::ostream&, Vector<double> const&)>

857	    info_stream << b << endl;
   0x0000000000410cb0 <+416>:	48 8d b4 24 b0 00 00 00	lea    0xb0(%rsp),%rsi
   0x0000000000410cb8 <+424>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x0000000000410cbd <+429>:	e8 3e 79 ff ff	callq  0x408600 <operator<< <double>(std::ostream&, Vector<double> const&)>

858	    T backup;
859	    //setup for debug monitor (QFP_gdb)
860	    QFP::checkpoint(Globals<T>::sum, sizeof(Globals<T>::sum), "sum", NO_WATCH);
   0x0000000000410cca <+442>:	48 8d bc 24 d0 00 00 00	lea    0xd0(%rsp),%rdi
   0x0000000000410cd2 <+450>:	be 30 c7 41 00	mov    $0x41c730,%esi
   0x0000000000410cd7 <+455>:	0f b6 1d f2 0e 21 00	movzbl 0x210ef2(%rip),%ebx        # 0x621bd0 <_ZN12_GLOBAL__N_18NO_WATCHE>
   0x0000000000410cde <+462>:	e8 6d 27 ff ff	callq  0x403450 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(char const*, std::allocator<char> const&)>
   0x0000000000410ce3 <+467>:	48 8d 94 24 d0 00 00 00	lea    0xd0(%rsp),%rdx
   0x0000000000410ceb <+475>:	89 d9	mov    %ebx,%ecx
   0x0000000000410ced <+477>:	be 08 00 00 00	mov    $0x8,%esi
   0x0000000000410cf2 <+482>:	bf a0 1c 62 00	mov    $0x621ca0,%edi
   0x0000000000410cf7 <+487>:	e8 94 d3 ff ff	callq  0x40e090 <QFP::checkpoint<long double>(long double&, unsigned long, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, bool)>
   0x0000000000410cfc <+492>:	48 8b bc 24 d0 00 00 00	mov    0xd0(%rsp),%rdi

861	    //printOnce(__FUNCTION__, &Globals<T>::prods.data()[0]);
862	
863	    
864	    //QFP::checkpoint(Globals<T>::prods.data()[2], sizeof(T), "product", NO_WATCH);
865	
866	    auto &dotProd = Globals<T>::sum;
867	
868	    for(int r = 0; r < dim; ++r){
   0x0000000000410d16 <+518>:	8b 44 24 54	mov    0x54(%rsp),%eax
   0x0000000000410d1a <+522>:	85 c0	test   %eax,%eax
   0x0000000000410d1c <+524>:	0f 8e 48 05 00 00	jle    0x41126a <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1882>
   0x0000000000410d22 <+530>:	d9 ee	fldz   
   0x0000000000410d24 <+532>:	66 0f ef ff	pxor   %xmm7,%xmm7
   0x0000000000410d28 <+536>:	83 e8 01	sub    $0x1,%eax
   0x0000000000410d2b <+539>:	48 c7 44 24 38 00 00 00 00	movq   $0x0,0x38(%rsp)
   0x0000000000410d34 <+548>:	db 7c 24 10	fstpt  0x10(%rsp)
   0x0000000000410d38 <+552>:	48 83 c0 01	add    $0x1,%rax
   0x0000000000410d3c <+556>:	48 89 44 24 70	mov    %rax,0x70(%rsp)
   0x0000000000410d41 <+561>:	f3 0f 11 7c 24 78	movss  %xmm7,0x78(%rsp)
   0x0000000000410d47 <+567>:	48 8b 44 24 38	mov    0x38(%rsp),%rax
   0x0000000000410d4c <+572>:	89 44 24 50	mov    %eax,0x50(%rsp)
   0x0000000000410d50 <+576>:	48 c1 e0 03	shl    $0x3,%rax
   0x000000000041116d <+1629>:	48 3b 44 24 70	cmp    0x70(%rsp),%rax
   0x0000000000411177 <+1639>:	0f 85 ca fb ff ff	jne    0x410d47 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+567>
   0x000000000041117d <+1645>:	e9 f8 00 00 00	jmpq   0x41127a <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1898>
   0x0000000000411182 <+1650>:	66 0f 1f 44 00 00	nopw   0x0(%rax,%rax,1)

869	    T &p = a[r];
870	      backup = p;
   0x0000000000410d68 <+600>:	f2 41 0f 10 36	movsd  (%r14),%xmm6
   0x0000000000410d6f <+607>:	f2 0f 11 74 24 48	movsd  %xmm6,0x48(%rsp)

871	      for(int i = 0; i < iters; ++i){
   0x0000000000410d64 <+596>:	8b 44 24 20	mov    0x20(%rsp),%eax
   0x0000000000410d6d <+605>:	85 c0	test   %eax,%eax
   0x0000000000410d75 <+613>:	0f 8e 92 03 00 00	jle    0x41110d <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1533>
   0x0000000000410d7b <+619>:	66 0f ef ff	pxor   %xmm7,%xmm7
   0x0000000000410d7f <+623>:	8b 44 24 68	mov    0x68(%rsp),%eax
   0x0000000000410d83 <+627>:	66 0f 2e f7	ucomisd %xmm7,%xmm6
   0x0000000000410d87 <+631>:	89 44 24 24	mov    %eax,0x24(%rsp)
   0x0000000000410d8b <+635>:	7a 06	jp     0x410d93 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+643>
   0x0000000000410d8d <+637>:	0f 84 96 04 00 00	je     0x411229 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1817>
   0x0000000000410d93 <+643>:	48 8b 44 24 48	mov    0x48(%rsp),%rax
   0x0000000000410d98 <+648>:	48 89 44 24 30	mov    %rax,0x30(%rsp)
   0x0000000000410da5 <+661>:	45 31 e4	xor    %r12d,%r12d
   0x0000000000410da8 <+664>:	31 db	xor    %ebx,%ebx
   0x0000000000410daa <+666>:	66 0f 1f 44 00 00	nopw   0x0(%rax,%rax,1)
   0x00000000004110fb <+1515>:	83 c3 01	add    $0x1,%ebx
   0x00000000004110fe <+1518>:	44 03 64 24 24	add    0x24(%rsp),%r12d
   0x0000000000411103 <+1523>:	39 5c 24 20	cmp    %ebx,0x20(%rsp)
   0x0000000000411107 <+1527>:	0f 85 a3 fc ff ff	jne    0x410db0 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+672>

872		//	cout << "r:" << r << ":i:" << i << std::endl;
873		p = FPHelpers::perturbFP(backup, i * ulp_inc);
   0x0000000000410dd2 <+706>:	49 89 06	mov    %rax,(%r14)

874		//Added this for force watchpoint hits every cycle (well, two).  We shouldn't really
875		//be hitting float min
876		dotProd = FLT_MIN;
   0x0000000000410dc0 <+688>:	f2 0f 10 0d 90 c8 00 00	movsd  0xc890(%rip),%xmm1        # 0x41d658
   0x0000000000410ddd <+717>:	f2 0f 11 0d bb 0e 21 00	movsd  %xmm1,0x210ebb(%rip)        # 0x621ca0 <_ZN7GlobalsIdE3sumE>

877		dotProd = a ^ b;
   0x0000000000410e99 <+905>:	f2 0f 11 05 ff 0d 21 00	movsd  %xmm0,0x210dff(%rip)        # 0x621ca0 <_ZN7GlobalsIdE3sumE>
   0x0000000000411188 <+1656>:	48 c7 05 0d 0b 21 00 00 00 00 00	movq   $0x0,0x210b0d(%rip)        # 0x621ca0 <_ZN7GlobalsIdE3sumE>

878		bool isOrth = dotProd == 0; //a.isOrtho(b);
   0x0000000000410eb5 <+933>:	31 ed	xor    %ebp,%ebp
   0x0000000000411789 <+3193>:	31 ed	xor    %ebp,%ebp

879		if(isOrth){
   0x0000000000410e95 <+901>:	66 0f ef e4	pxor   %xmm4,%xmm4
   0x0000000000410ea1 <+913>:	66 0f 2e c4	ucomisd %xmm4,%xmm0
   0x0000000000410ea5 <+917>:	7a 06	jp     0x410ead <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+925>
   0x0000000000410ea7 <+919>:	0f 84 e6 02 00 00	je     0x411193 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1667>

880		  orthoCount[r]++;
   0x0000000000411193 <+1667>:	48 8b 44 24 40	mov    0x40(%rsp),%rax
   0x0000000000411198 <+1672>:	48 8b 4c 24 38	mov    0x38(%rsp),%rcx
   0x000000000041119d <+1677>:	83 04 88 01	addl   $0x1,(%rax,%rcx,4)

881		  if(i != 0) score += fabs(p - backup); //score should be perturbed amount
   0x00000000004111a1 <+1681>:	85 db	test   %ebx,%ebx
   0x00000000004111a3 <+1683>:	0f 84 29 05 00 00	je     0x4116d2 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3010>
   0x00000000004111a9 <+1689>:	f2 41 0f 10 06	movsd  (%r14),%xmm0
   0x00000000004111ae <+1694>:	bd 01 00 00 00	mov    $0x1,%ebp
   0x00000000004111b3 <+1699>:	f2 0f 5c 44 24 48	subsd  0x48(%rsp),%xmm0
   0x00000000004111b9 <+1705>:	db 6c 24 10	fldt   0x10(%rsp)
   0x00000000004111bd <+1709>:	66 0f 54 05 3b c4 00 00	andpd  0xc43b(%rip),%xmm0        # 0x41d600
   0x00000000004111c5 <+1717>:	f2 0f 11 04 24	movsd  %xmm0,(%rsp)
   0x00000000004111ca <+1722>:	dc 04 24	faddl  (%rsp)
   0x00000000004111cd <+1725>:	db 7c 24 10	fstpt  0x10(%rsp)
   0x00000000004111d1 <+1729>:	e9 e1 fc ff ff	jmpq   0x410eb7 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+935>
   0x00000000004111d6 <+1734>:	66 2e 0f 1f 84 00 00 00 00 00	nopw   %cs:0x0(%rax,%rax,1)

882		}else{
883		  if(i == 0) score += fabs(dotProd); //a ^ b);  //if falsely not detecting ortho, should be the dot prod
   0x0000000000410ead <+925>:	85 db	test   %ebx,%ebx
   0x0000000000410eaf <+927>:	0f 84 d0 08 00 00	je     0x411785 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3189>
   0x0000000000411785 <+3189>:	db 6c 24 10	fldt   0x10(%rsp)
   0x000000000041178b <+3195>:	66 0f 54 05 6d be 00 00	andpd  0xbe6d(%rip),%xmm0        # 0x41d600
   0x0000000000411793 <+3203>:	f2 0f 11 04 24	movsd  %xmm0,(%rsp)
   0x0000000000411798 <+3208>:	dc 04 24	faddl  (%rsp)
   0x000000000041179b <+3211>:	db 7c 24 10	fstpt  0x10(%rsp)
   0x000000000041179f <+3215>:	e9 13 f7 ff ff	jmpq   0x410eb7 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+935>

884		}
885		info_stream << "i:" << i << ":a[" << r << "] = " << a[r] << ", " << FPWrap<T>(a[r]) << " multiplier: " << b[r] << ", " << FPWrap<T>(b[r]) << " perp: " << isOrth << " dot prod: " <<
   0x0000000000410f6d <+1117>:	f2 0f 11 84 24 f0 00 00 00	movsd  %xmm0,0xf0(%rsp)
   0x0000000000410f93 <+1155>:	f2 0f 10 16	movsd  (%rsi),%xmm2
   0x0000000000410fa4 <+1172>:	f2 0f 10 19	movsd  (%rcx),%xmm3
   0x0000000000410fb0 <+1184>:	f2 0f 11 14 24	movsd  %xmm2,(%rsp)
   0x0000000000410fb5 <+1189>:	f2 0f 11 5c 24 08	movsd  %xmm3,0x8(%rsp)
   0x0000000000410fc0 <+1200>:	89 de	mov    %ebx,%esi
   0x0000000000410fc2 <+1202>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x0000000000410fc7 <+1207>:	e8 34 11 ff ff	callq  0x402100 <_ZNSolsEi@plt>
   0x0000000000410fd9 <+1225>:	49 89 c7	mov    %rax,%r15
   0x0000000000410fe1 <+1233>:	8b 74 24 50	mov    0x50(%rsp),%esi
   0x0000000000410fe5 <+1237>:	4c 89 ff	mov    %r15,%rdi
   0x0000000000410fe8 <+1240>:	e8 13 11 ff ff	callq  0x402100 <_ZNSolsEi@plt>
   0x0000000000410ffa <+1258>:	49 89 c7	mov    %rax,%r15
   0x0000000000411025 <+1301>:	48 8d b4 24 10 01 00 00	lea    0x110(%rsp),%rsi
   0x000000000041102d <+1309>:	4c 89 ff	mov    %r15,%rdi
   0x0000000000411030 <+1312>:	e8 eb 76 ff ff	callq  0x408720 <operator<< <double>(std::ostream&, FPWrap<double> const&)>
   0x0000000000411042 <+1330>:	49 89 c7	mov    %rax,%r15
   0x000000000041106c <+1372>:	48 8d b4 24 30 01 00 00	lea    0x130(%rsp),%rsi
   0x0000000000411074 <+1380>:	4c 89 ff	mov    %r15,%rdi
   0x0000000000411077 <+1383>:	e8 a4 76 ff ff	callq  0x408720 <operator<< <double>(std::ostream&, FPWrap<double> const&)>
   0x0000000000411089 <+1401>:	49 89 c7	mov    %rax,%r15
   0x00000000004110b2 <+1442>:	48 8d b4 24 50 01 00 00	lea    0x150(%rsp),%rsi
   0x00000000004110ba <+1450>:	48 89 ef	mov    %rbp,%rdi
   0x00000000004110bd <+1453>:	e8 5e 76 ff ff	callq  0x408720 <operator<< <double>(std::ostream&, FPWrap<double> const&)>
   0x00000000004110c2 <+1458>:	49 89 c7	mov    %rax,%r15

886		  FPWrap<T>(a ^ b) << endl;
887	      }
888	      info_stream << "next dimension . . . " << endl;
889	      p = backup;
   0x0000000000411162 <+1618>:	f2 0f 10 7c 24 48	movsd  0x48(%rsp),%xmm7
   0x0000000000411168 <+1624>:	48 8b 44 24 38	mov    0x38(%rsp),%rax
   0x0000000000411172 <+1634>:	f2 41 0f 11 3e	movsd  %xmm7,(%r14)

890	    }
891	    info_stream << "Final report, one iteration set per dimensiion:" << endl;
892	    info_stream << '\t' << "ulp increment per loop: " << ulp_inc << endl;
893	    info_stream << '\t' << "iterations per dimension: " << iters << endl;
   0x0000000000411315 <+2053>:	8b 74 24 20	mov    0x20(%rsp),%esi
   0x0000000000411319 <+2057>:	48 89 df	mov    %rbx,%rdi
   0x000000000041131c <+2060>:	e8 df 0d ff ff	callq  0x402100 <_ZNSolsEi@plt>

894	    info_stream << '\t' << "dimensions: " << dim << endl;
   0x000000000041135d <+2125>:	8b 74 24 54	mov    0x54(%rsp),%esi
   0x0000000000411361 <+2129>:	48 89 df	mov    %rbx,%rdi
   0x0000000000411364 <+2132>:	e8 97 0d ff ff	callq  0x402100 <_ZNSolsEi@plt>

895	    string sType;
896	    switch(st){
   0x0000000000411395 <+2181>:	8b 44 24 7c	mov    0x7c(%rsp),%eax
   0x0000000000411399 <+2185>:	83 f8 01	cmp    $0x1,%eax
   0x000000000041139c <+2188>:	0f 84 c7 03 00 00	je     0x411769 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3161>
   0x00000000004113a2 <+2194>:	0f 82 50 04 00 00	jb     0x4117f8 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3304>
   0x00000000004113a8 <+2200>:	83 f8 02	cmp    $0x2,%eax
   0x00000000004113ab <+2203>:	0f 84 30 04 00 00	je     0x4117e1 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3281>
   0x00000000004113b1 <+2209>:	83 f8 03	cmp    $0x3,%eax
   0x00000000004113b4 <+2212>:	75 12	jne    0x4113c8 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2232>

897	    case Vector<T>::def:
898	      sType = "none";
899	      break;
900	    case Vector<T>::gt:
901	      sType = "decreasing";
902	      break;
903	    case Vector<T>::bi:
904	      sType = "std::inner_product";
905	      break;
906	    case Vector<T>::lt:
907	      sType = "increasing";
908	    }
909	    info_stream << '\t' << "sort: " << sType << endl;
910	    info_stream << '\t' << "precision (type): " << typeid(T).name() << endl;
911	    int cdim = 0;
912	    for(auto d: orthoCount){
   0x000000000041148a <+2426>:	48 8b 5c 24 40	mov    0x40(%rsp),%rbx
   0x000000000041148f <+2431>:	48 39 5c 24 58	cmp    %rbx,0x58(%rsp)
   0x0000000000411494 <+2436>:	0f 84 1a 01 00 00	je     0x4115b4 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2724>
   0x000000000041149a <+2442>:	4c 8b 64 24 40	mov    0x40(%rsp),%r12
   0x000000000041149f <+2447>:	31 ed	xor    %ebp,%ebp
   0x00000000004114a1 <+2449>:	eb 2c	jmp    0x4114cf <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2495>
   0x00000000004114a3 <+2451>:	0f 1f 44 00 00	nopl   0x0(%rax,%rax,1)
   0x00000000004114a8 <+2456>:	0f be 73 43	movsbl 0x43(%rbx),%esi
   0x00000000004114c4 <+2484>:	4c 39 64 24 58	cmp    %r12,0x58(%rsp)
   0x00000000004114c9 <+2489>:	0f 84 e5 00 00 00	je     0x4115b4 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2724>
   0x00000000004114e6 <+2518>:	45 8b 2c 24	mov    (%r12),%r13d

913	      info_stream << "For mod dim " << cdim << ", there were " << d << " ortho vectors, product magnitude (biased fp exp): "
   0x0000000000411516 <+2566>:	89 ee	mov    %ebp,%esi
   0x0000000000411518 <+2568>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x000000000041151d <+2573>:	e8 de 0b ff ff	callq  0x402100 <_ZNSolsEi@plt>
   0x000000000041152f <+2591>:	49 89 c6	mov    %rax,%r14

914		   << FPHelpers::getExponent(a[cdim] * b[cdim]) << endl;
   0x00000000004114cf <+2495>:	48 8b 84 24 90 00 00 00	mov    0x90(%rsp),%rax

915	      cdim++;
916	    }
917	    std::string name = std::string(__func__);
   0x00000000004115b4 <+2724>:	48 8d bc 24 10 01 00 00	lea    0x110(%rsp),%rdi
   0x00000000004115bc <+2732>:	be d0 cc 41 00	mov    $0x41ccd0,%esi
   0x00000000004115c1 <+2737>:	e8 8a 1e ff ff	callq  0x403450 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(char const*, std::allocator<char> const&)>

918	    if (a[0] == 0.2 ) name += "N2"; else name += "P2";
   0x00000000004115c6 <+2742>:	48 8b 84 24 90 00 00 00	mov    0x90(%rsp),%rax
   0x00000000004115ce <+2750>:	f2 0f 10 05 72 c0 00 00	movsd  0xc072(%rip),%xmm0        # 0x41d648
   0x00000000004115d6 <+2758>:	66 0f 2e 00	ucomisd (%rax),%xmm0
   0x00000000004115da <+2762>:	0f 8a db 00 00 00	jp     0x4116bb <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2987>
   0x00000000004115e0 <+2768>:	0f 85 d5 00 00 00	jne    0x4116bb <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2987>

919	    return {name, {score, 0.0}};
920	  }
   0x000000000041169c <+2956>:	48 8b 44 24 60	mov    0x60(%rsp),%rax
   0x00000000004116a1 <+2961>:	48 81 c4 78 01 00 00	add    $0x178,%rsp
   0x00000000004116a8 <+2968>:	5b	pop    %rbx
   0x00000000004116a9 <+2969>:	5d	pop    %rbp
   0x00000000004116aa <+2970>:	41 5c	pop    %r12
   0x00000000004116ac <+2972>:	41 5d	pop    %r13
   0x00000000004116ae <+2974>:	41 5e	pop    %r14
   0x00000000004116b0 <+2976>:	41 5f	pop    %r15
   0x00000000004116b2 <+2978>:	c3	retq   
   0x00000000004116b3 <+2979>:	48 89 c3	mov    %rax,%rbx
   0x00000000004116b6 <+2982>:	e9 ee 00 00 00	jmpq   0x4117a9 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3225>

End of assembler dump.


Dump of assembler code for function FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t):
   0x0000000000410b10 <+0>:	41 57	push   %r15
   0x0000000000410b12 <+2>:	41 56	push   %r14
   0x0000000000410b14 <+4>:	41 55	push   %r13
   0x0000000000410b16 <+6>:	41 54	push   %r12
   0x0000000000410b18 <+8>:	55	push   %rbp
   0x0000000000410b19 <+9>:	53	push   %rbx
   0x0000000000410b1a <+10>:	48 63 ea	movslq %edx,%rbp
   0x0000000000410b1d <+13>:	4c 89 c3	mov    %r8,%rbx
   0x0000000000410b20 <+16>:	48 81 ec 78 01 00 00	sub    $0x178,%rsp
   0x0000000000410b27 <+23>:	48 85 ed	test   %rbp,%rbp
   0x0000000000410b2a <+26>:	48 89 7c 24 60	mov    %rdi,0x60(%rsp)
   0x0000000000410b2f <+31>:	89 74 24 20	mov    %esi,0x20(%rsp)
   0x0000000000410b33 <+35>:	89 6c 24 54	mov    %ebp,0x54(%rsp)
   0x0000000000410b37 <+39>:	48 89 4c 24 68	mov    %rcx,0x68(%rsp)
   0x0000000000410b3c <+44>:	44 89 4c 24 7c	mov    %r9d,0x7c(%rsp)
   0x0000000000410b41 <+49>:	0f 84 0b 0c 00 00	je     0x411752 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3138>
   0x0000000000410b47 <+55>:	48 b8 ff ff ff ff ff ff ff 3f	movabs $0x3fffffffffffffff,%rax
   0x0000000000410b51 <+65>:	48 39 c5	cmp    %rax,%rbp
   0x0000000000410b54 <+68>:	0f 87 f3 0b 00 00	ja     0x41174d <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3133>
   0x0000000000410b5a <+74>:	4c 8d 24 ad 00 00 00 00	lea    0x0(,%rbp,4),%r12
   0x0000000000410b62 <+82>:	4c 89 e7	mov    %r12,%rdi
   0x0000000000410b65 <+85>:	e8 56 19 ff ff	callq  0x4024c0 <_Znwm@plt>
   0x0000000000410b6a <+90>:	48 89 44 24 40	mov    %rax,0x40(%rsp)
   0x0000000000410b6f <+95>:	31 c0	xor    %eax,%eax
   0x0000000000410b71 <+97>:	48 8b 54 24 40	mov    0x40(%rsp),%rdx
   0x0000000000410b76 <+102>:	c7 04 82 00 00 00 00	movl   $0x0,(%rdx,%rax,4)
   0x0000000000410b7d <+109>:	48 83 c0 01	add    $0x1,%rax
   0x0000000000410b81 <+113>:	48 39 c5	cmp    %rax,%rbp
   0x0000000000410b84 <+116>:	75 eb	jne    0x410b71 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+97>
   0x0000000000410b86 <+118>:	49 01 d4	add    %rdx,%r12
   0x0000000000410b89 <+121>:	4c 89 64 24 58	mov    %r12,0x58(%rsp)
   0x0000000000410b8e <+126>:	48 8d 8c 24 87 00 00 00	lea    0x87(%rsp),%rcx
   0x0000000000410b96 <+134>:	48 8d 94 24 88 00 00 00	lea    0x88(%rsp),%rdx
   0x0000000000410b9e <+142>:	48 8d bc 24 90 00 00 00	lea    0x90(%rsp),%rdi
   0x0000000000410ba6 <+150>:	48 89 ee	mov    %rbp,%rsi
   0x0000000000410ba9 <+153>:	48 c7 84 24 88 00 00 00 00 00 00 00	movq   $0x0,0x88(%rsp)
   0x0000000000410bb5 <+165>:	e8 16 a0 ff ff	callq  0x40abd0 <std::vector<double, std::allocator<double> >::vector(unsigned long, double const&, std::allocator<double> const&)>
   0x0000000000410bba <+170>:	48 8b 84 24 98 00 00 00	mov    0x98(%rsp),%rax
   0x0000000000410bc2 <+178>:	48 8b 94 24 90 00 00 00	mov    0x90(%rsp),%rdx
   0x0000000000410bca <+186>:	48 39 d0	cmp    %rdx,%rax
   0x0000000000410bcd <+189>:	74 43	je     0x410c12 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+258>
   0x0000000000410bcf <+191>:	48 8b 3b	mov    (%rbx),%rdi
   0x0000000000410bd2 <+194>:	48 89 d6	mov    %rdx,%rsi
   0x0000000000410bd5 <+197>:	41 b9 01 00 00 00	mov    $0x1,%r9d
   0x0000000000410bdb <+203>:	48 89 f9	mov    %rdi,%rcx
   0x0000000000410bde <+206>:	66 0f ef c0	pxor   %xmm0,%xmm0
   0x0000000000410be2 <+210>:	45 89 c8	mov    %r9d,%r8d
   0x0000000000410be5 <+213>:	48 83 c6 08	add    $0x8,%rsi
   0x0000000000410be9 <+217>:	41 d3 e0	shl    %cl,%r8d
   0x0000000000410bec <+220>:	48 83 c1 01	add    $0x1,%rcx
   0x0000000000410bf0 <+224>:	f2 41 0f 2a c0	cvtsi2sd %r8d,%xmm0
   0x0000000000410bf5 <+229>:	f2 0f 11 46 f8	movsd  %xmm0,-0x8(%rsi)
   0x0000000000410bfa <+234>:	48 39 f0	cmp    %rsi,%rax
   0x0000000000410bfd <+237>:	75 df	jne    0x410bde <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+206>
   0x0000000000410bff <+239>:	48 83 c2 08	add    $0x8,%rdx
   0x0000000000410c03 <+243>:	48 29 d0	sub    %rdx,%rax
   0x0000000000410c06 <+246>:	48 c1 e8 03	shr    $0x3,%rax
   0x0000000000410c0a <+250>:	48 8d 44 38 01	lea    0x1(%rax,%rdi,1),%rax
   0x0000000000410c0f <+255>:	48 89 03	mov    %rax,(%rbx)
   0x0000000000410c12 <+258>:	8b 44 24 7c	mov    0x7c(%rsp),%eax
   0x0000000000410c16 <+262>:	48 8d b4 24 90 00 00 00	lea    0x90(%rsp),%rsi
   0x0000000000410c1e <+270>:	48 8d bc 24 b0 00 00 00	lea    0xb0(%rsp),%rdi
   0x0000000000410c26 <+278>:	89 84 24 a8 00 00 00	mov    %eax,0xa8(%rsp)
   0x0000000000410c2d <+285>:	e8 de 98 ff ff	callq  0x40a510 <Vector<double>::genOrthoVector()>
   0x0000000000410c32 <+290>:	ba 34 00 00 00	mov    $0x34,%edx
   0x0000000000410c37 <+295>:	be a0 c4 41 00	mov    $0x41c4a0,%esi
   0x0000000000410c3c <+300>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x0000000000410c41 <+305>:	e8 da 16 ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x0000000000410c46 <+310>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x0000000000410c4b <+315>:	e8 20 18 ff ff	callq  0x402470 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@plt>
   0x0000000000410c50 <+320>:	8b 54 24 54	mov    0x54(%rsp),%edx
   0x0000000000410c54 <+324>:	31 db	xor    %ebx,%ebx
   0x0000000000410c56 <+326>:	85 d2	test   %edx,%edx
   0x0000000000410c58 <+328>:	7e 32	jle    0x410c8c <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+380>
   0x0000000000410c5a <+330>:	89 de	mov    %ebx,%esi
   0x0000000000410c5c <+332>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x0000000000410c61 <+337>:	e8 9a 14 ff ff	callq  0x402100 <_ZNSolsEi@plt>
   0x0000000000410c66 <+342>:	48 8d b4 24 81 00 00 00	lea    0x81(%rsp),%rsi
   0x0000000000410c6e <+350>:	ba 01 00 00 00	mov    $0x1,%edx
   0x0000000000410c73 <+355>:	48 89 c7	mov    %rax,%rdi
   0x0000000000410c76 <+358>:	c6 84 24 81 00 00 00 09	movb   $0x9,0x81(%rsp)
   0x0000000000410c7e <+366>:	e8 9d 16 ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x0000000000410c83 <+371>:	83 c3 01	add    $0x1,%ebx
   0x0000000000410c86 <+374>:	39 5c 24 54	cmp    %ebx,0x54(%rsp)
   0x0000000000410c8a <+378>:	75 ce	jne    0x410c5a <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+330>
   0x0000000000410c8c <+380>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x0000000000410c91 <+385>:	e8 da 17 ff ff	callq  0x402470 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@plt>
   0x0000000000410c96 <+390>:	48 8d b4 24 90 00 00 00	lea    0x90(%rsp),%rsi
   0x0000000000410c9e <+398>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x0000000000410ca3 <+403>:	e8 58 79 ff ff	callq  0x408600 <operator<< <double>(std::ostream&, Vector<double> const&)>
   0x0000000000410ca8 <+408>:	48 89 c7	mov    %rax,%rdi
   0x0000000000410cab <+411>:	e8 c0 17 ff ff	callq  0x402470 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@plt>
   0x0000000000410cb0 <+416>:	48 8d b4 24 b0 00 00 00	lea    0xb0(%rsp),%rsi
   0x0000000000410cb8 <+424>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x0000000000410cbd <+429>:	e8 3e 79 ff ff	callq  0x408600 <operator<< <double>(std::ostream&, Vector<double> const&)>
   0x0000000000410cc2 <+434>:	48 89 c7	mov    %rax,%rdi
   0x0000000000410cc5 <+437>:	e8 a6 17 ff ff	callq  0x402470 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@plt>
   0x0000000000410cca <+442>:	48 8d bc 24 d0 00 00 00	lea    0xd0(%rsp),%rdi
   0x0000000000410cd2 <+450>:	be 30 c7 41 00	mov    $0x41c730,%esi
   0x0000000000410cd7 <+455>:	0f b6 1d f2 0e 21 00	movzbl 0x210ef2(%rip),%ebx        # 0x621bd0 <_ZN12_GLOBAL__N_18NO_WATCHE>
   0x0000000000410cde <+462>:	e8 6d 27 ff ff	callq  0x403450 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(char const*, std::allocator<char> const&)>
   0x0000000000410ce3 <+467>:	48 8d 94 24 d0 00 00 00	lea    0xd0(%rsp),%rdx
   0x0000000000410ceb <+475>:	89 d9	mov    %ebx,%ecx
   0x0000000000410ced <+477>:	be 08 00 00 00	mov    $0x8,%esi
   0x0000000000410cf2 <+482>:	bf a0 1c 62 00	mov    $0x621ca0,%edi
   0x0000000000410cf7 <+487>:	e8 94 d3 ff ff	callq  0x40e090 <QFP::checkpoint<long double>(long double&, unsigned long, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, bool)>
   0x0000000000410cfc <+492>:	48 8b bc 24 d0 00 00 00	mov    0xd0(%rsp),%rdi
   0x0000000000410d04 <+500>:	48 8d 84 24 e0 00 00 00	lea    0xe0(%rsp),%rax
   0x0000000000410d0c <+508>:	48 39 c7	cmp    %rax,%rdi
   0x0000000000410d0f <+511>:	74 05	je     0x410d16 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+518>
   0x0000000000410d11 <+513>:	e8 3a 14 ff ff	callq  0x402150 <_ZdlPv@plt>
   0x0000000000410d16 <+518>:	8b 44 24 54	mov    0x54(%rsp),%eax
   0x0000000000410d1a <+522>:	85 c0	test   %eax,%eax
   0x0000000000410d1c <+524>:	0f 8e 48 05 00 00	jle    0x41126a <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1882>
   0x0000000000410d22 <+530>:	d9 ee	fldz   
   0x0000000000410d24 <+532>:	66 0f ef ff	pxor   %xmm7,%xmm7
   0x0000000000410d28 <+536>:	83 e8 01	sub    $0x1,%eax
   0x0000000000410d2b <+539>:	48 c7 44 24 38 00 00 00 00	movq   $0x0,0x38(%rsp)
   0x0000000000410d34 <+548>:	db 7c 24 10	fstpt  0x10(%rsp)
   0x0000000000410d38 <+552>:	48 83 c0 01	add    $0x1,%rax
   0x0000000000410d3c <+556>:	48 89 44 24 70	mov    %rax,0x70(%rsp)
   0x0000000000410d41 <+561>:	f3 0f 11 7c 24 78	movss  %xmm7,0x78(%rsp)
   0x0000000000410d47 <+567>:	48 8b 44 24 38	mov    0x38(%rsp),%rax
   0x0000000000410d4c <+572>:	89 44 24 50	mov    %eax,0x50(%rsp)
   0x0000000000410d50 <+576>:	48 c1 e0 03	shl    $0x3,%rax
   0x0000000000410d54 <+580>:	49 89 c6	mov    %rax,%r14
   0x0000000000410d57 <+583>:	4c 03 b4 24 90 00 00 00	add    0x90(%rsp),%r14
   0x0000000000410d5f <+591>:	48 89 44 24 28	mov    %rax,0x28(%rsp)
   0x0000000000410d64 <+596>:	8b 44 24 20	mov    0x20(%rsp),%eax
   0x0000000000410d68 <+600>:	f2 41 0f 10 36	movsd  (%r14),%xmm6
   0x0000000000410d6d <+605>:	85 c0	test   %eax,%eax
   0x0000000000410d6f <+607>:	f2 0f 11 74 24 48	movsd  %xmm6,0x48(%rsp)
   0x0000000000410d75 <+613>:	0f 8e 92 03 00 00	jle    0x41110d <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1533>
   0x0000000000410d7b <+619>:	66 0f ef ff	pxor   %xmm7,%xmm7
   0x0000000000410d7f <+623>:	8b 44 24 68	mov    0x68(%rsp),%eax
   0x0000000000410d83 <+627>:	66 0f 2e f7	ucomisd %xmm7,%xmm6
   0x0000000000410d87 <+631>:	89 44 24 24	mov    %eax,0x24(%rsp)
   0x0000000000410d8b <+635>:	7a 06	jp     0x410d93 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+643>
   0x0000000000410d8d <+637>:	0f 84 96 04 00 00	je     0x411229 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1817>
   0x0000000000410d93 <+643>:	48 8b 44 24 48	mov    0x48(%rsp),%rax
   0x0000000000410d98 <+648>:	48 89 44 24 30	mov    %rax,0x30(%rsp)
   0x0000000000410d9d <+653>:	4c 8d ac 24 60 01 00 00	lea    0x160(%rsp),%r13
   0x0000000000410da5 <+661>:	45 31 e4	xor    %r12d,%r12d
   0x0000000000410da8 <+664>:	31 db	xor    %ebx,%ebx
   0x0000000000410daa <+666>:	66 0f 1f 44 00 00	nopw   0x0(%rax,%rax,1)
   0x0000000000410db0 <+672>:	49 63 c4	movslq %r12d,%rax
   0x0000000000410db3 <+675>:	48 03 44 24 30	add    0x30(%rsp),%rax
   0x0000000000410db8 <+680>:	48 8d bc 24 30 01 00 00	lea    0x130(%rsp),%rdi
   0x0000000000410dc0 <+688>:	f2 0f 10 0d 90 c8 00 00	movsd  0xc890(%rip),%xmm1        # 0x41d658
   0x0000000000410dc8 <+696>:	ba c1 cc 41 00	mov    $0x41ccc1,%edx
   0x0000000000410dcd <+701>:	be b8 cc 41 00	mov    $0x41ccb8,%esi
   0x0000000000410dd2 <+706>:	49 89 06	mov    %rax,(%r14)
   0x0000000000410dd5 <+709>:	48 8d 84 24 40 01 00 00	lea    0x140(%rsp),%rax
   0x0000000000410ddd <+717>:	f2 0f 11 0d bb 0e 21 00	movsd  %xmm1,0x210ebb(%rip)        # 0x621ca0 <_ZN7GlobalsIdE3sumE>
   0x0000000000410de5 <+725>:	48 89 84 24 30 01 00 00	mov    %rax,0x130(%rsp)
   0x0000000000410ded <+733>:	e8 ae 25 ff ff	callq  0x4033a0 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char*>(char*, char*, std::forward_iterator_tag)>
   0x0000000000410df2 <+738>:	48 8b bc 24 30 01 00 00	mov    0x130(%rsp),%rdi
   0x0000000000410dfa <+746>:	48 8d 84 24 40 01 00 00	lea    0x140(%rsp),%rax
   0x0000000000410e02 <+754>:	48 39 c7	cmp    %rax,%rdi
   0x0000000000410e05 <+757>:	74 05	je     0x410e0c <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+764>
   0x0000000000410e07 <+759>:	e8 44 13 ff ff	callq  0x402150 <_ZdlPv@plt>
   0x0000000000410e0c <+764>:	83 bc 24 a8 00 00 00 02	cmpl   $0x2,0xa8(%rsp)
   0x0000000000410e14 <+772>:	0f 84 6e 03 00 00	je     0x411188 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1656>
   0x0000000000410e1a <+778>:	48 8b bc 24 90 00 00 00	mov    0x90(%rsp),%rdi
   0x0000000000410e22 <+786>:	48 8b 8c 24 98 00 00 00	mov    0x98(%rsp),%rcx
   0x0000000000410e2a <+794>:	48 8b 35 c7 0e 21 00	mov    0x210ec7(%rip),%rsi        # 0x621cf8 <_ZN7GlobalsIdE5prodsE+8>
   0x0000000000410e31 <+801>:	48 8b 15 b8 0e 21 00	mov    0x210eb8(%rip),%rdx        # 0x621cf0 <_ZN7GlobalsIdE5prodsE>
   0x0000000000410e38 <+808>:	48 29 f9	sub    %rdi,%rcx
   0x0000000000410e3b <+811>:	48 89 c8	mov    %rcx,%rax
   0x0000000000410e3e <+814>:	48 c1 f8 03	sar    $0x3,%rax rax /= 8
   0x0000000000410e42 <+818>:	48 85 c0	test   %rax,%rax #jump if rax == 0
   0x0000000000410e45 <+821>:	74 2a	je     0x410e71 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+865>
   0x0000000000410e47 <+823>:	4c 8b 84 24 b0 00 00 00	mov    0xb0(%rsp),%r8
   0x0000000000410e4f <+831>:	31 c0	xor    %eax,%eax
   0x0000000000410e51 <+833>:	0f 1f 80 00 00 00 00	nopl   0x0(%rax)
   0x0000000000410e58 <+840>:	f2 0f 10 04 07	movsd  (%rdi,%rax,1),%xmm0
   0x0000000000410e5d <+845>:	f2 41 0f 59 04 00	mulsd  (%r8,%rax,1),%xmm0
   0x0000000000410e63 <+851>:	f2 0f 11 04 02	movsd  %xmm0,(%rdx,%rax,1)
   0x0000000000410e68 <+856>:	48 83 c0 08	add    $0x8,%rax
   0x0000000000410e6c <+860>:	48 39 c8	cmp    %rcx,%rax #continue i < size()
   0x0000000000410e6f <+863>:	75 e7	jne    0x410e58 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+840>
   0x0000000000410e71 <+865>:	48 39 d6	cmp    %rdx,%rsi
   0x0000000000410e74 <+868>:	48 89 d0	mov    %rdx,%rax
   0x0000000000410e77 <+871>:	66 0f ef c0	pxor   %xmm0,%xmm0 #this is where we drop out of the loop at the end of the loop
   0x0000000000410e7b <+875>:	0f 84 07 03 00 00	je     0x411188 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1656>
   0x0000000000410e81 <+881>:	0f 1f 80 00 00 00 00	nopl   0x0(%rax)
   0x0000000000410e88 <+888>:	f2 0f 58 00	addsd  (%rax),%xmm0
   0x0000000000410e8c <+892>:	48 83 c0 08	add    $0x8,%rax
   0x0000000000410e90 <+896>:	48 39 c6	cmp    %rax,%rsi
   0x0000000000410e93 <+899>:	75 f3	jne    0x410e88 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+888>
   0x0000000000410e95 <+901>:	66 0f ef e4	pxor   %xmm4,%xmm4
   0x0000000000410e99 <+905>:	f2 0f 11 05 ff 0d 21 00	movsd  %xmm0,0x210dff(%rip)        # 0x621ca0 <_ZN7GlobalsIdE3sumE>
   0x0000000000410ea1 <+913>:	66 0f 2e c4	ucomisd %xmm4,%xmm0
   0x0000000000410ea5 <+917>:	7a 06	jp     0x410ead <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+925>
   0x0000000000410ea7 <+919>:	0f 84 e6 02 00 00	je     0x411193 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1667>
   0x0000000000410ead <+925>:	85 db	test   %ebx,%ebx
   0x0000000000410eaf <+927>:	0f 84 d0 08 00 00	je     0x411785 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3189>
   0x0000000000410eb5 <+933>:	31 ed	xor    %ebp,%ebp
   0x0000000000410eb7 <+935>:	48 8d bc 24 50 01 00 00	lea    0x150(%rsp),%rdi
   0x0000000000410ebf <+943>:	ba c1 cc 41 00	mov    $0x41ccc1,%edx
   0x0000000000410ec4 <+948>:	be b8 cc 41 00	mov    $0x41ccb8,%esi
   0x0000000000410ec9 <+953>:	4c 89 ac 24 50 01 00 00	mov    %r13,0x150(%rsp)
   0x0000000000410ed1 <+961>:	e8 ca 24 ff ff	callq  0x4033a0 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char*>(char*, char*, std::forward_iterator_tag)>
   0x0000000000410ed6 <+966>:	48 8b bc 24 50 01 00 00	mov    0x150(%rsp),%rdi
   0x0000000000410ede <+974>:	4c 39 ef	cmp    %r13,%rdi
   0x0000000000410ee1 <+977>:	74 05	je     0x410ee8 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+984>
   0x0000000000410ee3 <+979>:	e8 68 12 ff ff	callq  0x402150 <_ZdlPv@plt>
   0x0000000000410ee8 <+984>:	83 bc 24 a8 00 00 00 02	cmpl   $0x2,0xa8(%rsp)
   0x0000000000410ef0 <+992>:	0f 84 1a 03 00 00	je     0x411210 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1792>
   0x0000000000410ef6 <+998>:	48 8b 8c 24 90 00 00 00	mov    0x90(%rsp),%rcx
   0x0000000000410efe <+1006>:	48 8b bc 24 98 00 00 00	mov    0x98(%rsp),%rdi
   0x0000000000410f06 <+1014>:	4c 8b 05 eb 0d 21 00	mov    0x210deb(%rip),%r8        # 0x621cf8 <_ZN7GlobalsIdE5prodsE+8>
   0x0000000000410f0d <+1021>:	48 8b 15 dc 0d 21 00	mov    0x210ddc(%rip),%rdx        # 0x621cf0 <_ZN7GlobalsIdE5prodsE>
   0x0000000000410f14 <+1028>:	48 8b b4 24 b0 00 00 00	mov    0xb0(%rsp),%rsi
   0x0000000000410f1c <+1036>:	48 29 cf	sub    %rcx,%rdi
=> 0x0000000000410f1f <+1039>:	48 89 f8	mov    %rdi,%rax
   0x0000000000410f22 <+1042>:	48 c1 f8 03	sar    $0x3,%rax
   0x0000000000410f26 <+1046>:	48 85 c0	test   %rax,%rax
   0x0000000000410f29 <+1049>:	74 1d	je     0x410f48 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1080>
   0x0000000000410f2b <+1051>:	31 c0	xor    %eax,%eax
   0x0000000000410f2d <+1053>:	0f 1f 00	nopl   (%rax)
   0x0000000000410f30 <+1056>:	f2 0f 10 04 01	movsd  (%rcx,%rax,1),%xmm0
   0x0000000000410f35 <+1061>:	f2 0f 59 04 06	mulsd  (%rsi,%rax,1),%xmm0
   0x0000000000410f3a <+1066>:	f2 0f 11 04 02	movsd  %xmm0,(%rdx,%rax,1)
   0x0000000000410f3f <+1071>:	48 83 c0 08	add    $0x8,%rax
   0x0000000000410f43 <+1075>:	48 39 f8	cmp    %rdi,%rax
   0x0000000000410f46 <+1078>:	75 e8	jne    0x410f30 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1056>
   0x0000000000410f48 <+1080>:	49 39 d0	cmp    %rdx,%r8
   0x0000000000410f4b <+1083>:	48 89 d0	mov    %rdx,%rax
   0x0000000000410f4e <+1086>:	66 0f ef c0	pxor   %xmm0,%xmm0
   0x0000000000410f52 <+1090>:	74 11	je     0x410f65 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1109>
   0x0000000000410f54 <+1092>:	0f 1f 40 00	nopl   0x0(%rax)
   0x0000000000410f58 <+1096>:	f2 0f 58 00	addsd  (%rax),%xmm0
   0x0000000000410f5c <+1100>:	48 83 c0 08	add    $0x8,%rax
   0x0000000000410f60 <+1104>:	49 39 c0	cmp    %rax,%r8
   0x0000000000410f63 <+1107>:	75 f3	jne    0x410f58 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1096>
   0x0000000000410f65 <+1109>:	48 8d 84 24 f0 00 00 00	lea    0xf0(%rsp),%rax
   0x0000000000410f6d <+1117>:	f2 0f 11 84 24 f0 00 00 00	movsd  %xmm0,0xf0(%rsp)
   0x0000000000410f76 <+1126>:	ba 02 00 00 00	mov    $0x2,%edx
   0x0000000000410f7b <+1131>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x0000000000410f80 <+1136>:	48 89 84 24 50 01 00 00	mov    %rax,0x150(%rsp)
   0x0000000000410f88 <+1144>:	48 8b 44 24 28	mov    0x28(%rsp),%rax
   0x0000000000410f8d <+1149>:	48 01 c6	add    %rax,%rsi
   0x0000000000410f90 <+1152>:	48 01 c1	add    %rax,%rcx
   0x0000000000410f93 <+1155>:	f2 0f 10 16	movsd  (%rsi),%xmm2
   0x0000000000410f97 <+1159>:	48 89 b4 24 30 01 00 00	mov    %rsi,0x130(%rsp)
   0x0000000000410f9f <+1167>:	be 34 c7 41 00	mov    $0x41c734,%esi
   0x0000000000410fa4 <+1172>:	f2 0f 10 19	movsd  (%rcx),%xmm3
   0x0000000000410fa8 <+1176>:	48 89 8c 24 10 01 00 00	mov    %rcx,0x110(%rsp)
   0x0000000000410fb0 <+1184>:	f2 0f 11 14 24	movsd  %xmm2,(%rsp)
   0x0000000000410fb5 <+1189>:	f2 0f 11 5c 24 08	movsd  %xmm3,0x8(%rsp)
   0x0000000000410fbb <+1195>:	e8 60 13 ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x0000000000410fc0 <+1200>:	89 de	mov    %ebx,%esi
   0x0000000000410fc2 <+1202>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x0000000000410fc7 <+1207>:	e8 34 11 ff ff	callq  0x402100 <_ZNSolsEi@plt>
   0x0000000000410fcc <+1212>:	ba 03 00 00 00	mov    $0x3,%edx
   0x0000000000410fd1 <+1217>:	be 37 c7 41 00	mov    $0x41c737,%esi
   0x0000000000410fd6 <+1222>:	48 89 c7	mov    %rax,%rdi
   0x0000000000410fd9 <+1225>:	49 89 c7	mov    %rax,%r15
   0x0000000000410fdc <+1228>:	e8 3f 13 ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x0000000000410fe1 <+1233>:	8b 74 24 50	mov    0x50(%rsp),%esi
   0x0000000000410fe5 <+1237>:	4c 89 ff	mov    %r15,%rdi
   0x0000000000410fe8 <+1240>:	e8 13 11 ff ff	callq  0x402100 <_ZNSolsEi@plt>
   0x0000000000410fed <+1245>:	ba 04 00 00 00	mov    $0x4,%edx
   0x0000000000410ff2 <+1250>:	be 3b c7 41 00	mov    $0x41c73b,%esi
   0x0000000000410ff7 <+1255>:	48 89 c7	mov    %rax,%rdi
   0x0000000000410ffa <+1258>:	49 89 c7	mov    %rax,%r15
   0x0000000000410ffd <+1261>:	e8 1e 13 ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x0000000000411002 <+1266>:	f2 0f 10 44 24 08	movsd  0x8(%rsp),%xmm0
   0x0000000000411008 <+1272>:	4c 89 ff	mov    %r15,%rdi
   0x000000000041100b <+1275>:	e8 30 14 ff ff	callq  0x402440 <_ZNSo9_M_insertIdEERSoT_@plt>
   0x0000000000411010 <+1280>:	ba 02 00 00 00	mov    $0x2,%edx
   0x0000000000411015 <+1285>:	be 40 c7 41 00	mov    $0x41c740,%esi
   0x000000000041101a <+1290>:	48 89 c7	mov    %rax,%rdi
   0x000000000041101d <+1293>:	49 89 c7	mov    %rax,%r15
   0x0000000000411020 <+1296>:	e8 fb 12 ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x0000000000411025 <+1301>:	48 8d b4 24 10 01 00 00	lea    0x110(%rsp),%rsi
   0x000000000041102d <+1309>:	4c 89 ff	mov    %r15,%rdi
   0x0000000000411030 <+1312>:	e8 eb 76 ff ff	callq  0x408720 <operator<< <double>(std::ostream&, FPWrap<double> const&)>
   0x0000000000411035 <+1317>:	ba 0d 00 00 00	mov    $0xd,%edx
   0x000000000041103a <+1322>:	be 43 c7 41 00	mov    $0x41c743,%esi
   0x000000000041103f <+1327>:	48 89 c7	mov    %rax,%rdi
   0x0000000000411042 <+1330>:	49 89 c7	mov    %rax,%r15
   0x0000000000411045 <+1333>:	e8 d6 12 ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x000000000041104a <+1338>:	f2 0f 10 04 24	movsd  (%rsp),%xmm0
   0x000000000041104f <+1343>:	4c 89 ff	mov    %r15,%rdi
   0x0000000000411052 <+1346>:	e8 e9 13 ff ff	callq  0x402440 <_ZNSo9_M_insertIdEERSoT_@plt>
   0x0000000000411057 <+1351>:	ba 02 00 00 00	mov    $0x2,%edx
   0x000000000041105c <+1356>:	be 40 c7 41 00	mov    $0x41c740,%esi
   0x0000000000411061 <+1361>:	48 89 c7	mov    %rax,%rdi
   0x0000000000411064 <+1364>:	49 89 c7	mov    %rax,%r15
   0x0000000000411067 <+1367>:	e8 b4 12 ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x000000000041106c <+1372>:	48 8d b4 24 30 01 00 00	lea    0x130(%rsp),%rsi
   0x0000000000411074 <+1380>:	4c 89 ff	mov    %r15,%rdi
   0x0000000000411077 <+1383>:	e8 a4 76 ff ff	callq  0x408720 <operator<< <double>(std::ostream&, FPWrap<double> const&)>
   0x000000000041107c <+1388>:	ba 07 00 00 00	mov    $0x7,%edx
   0x0000000000411081 <+1393>:	be 51 c7 41 00	mov    $0x41c751,%esi
   0x0000000000411086 <+1398>:	48 89 c7	mov    %rax,%rdi
   0x0000000000411089 <+1401>:	49 89 c7	mov    %rax,%r15
   0x000000000041108c <+1404>:	e8 8f 12 ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x0000000000411091 <+1409>:	40 0f b6 f5	movzbl %bpl,%esi
   0x0000000000411095 <+1413>:	4c 89 ff	mov    %r15,%rdi
   0x0000000000411098 <+1416>:	e8 63 12 ff ff	callq  0x402300 <_ZNSo9_M_insertIbEERSoT_@plt>
   0x000000000041109d <+1421>:	ba 0b 00 00 00	mov    $0xb,%edx
   0x00000000004110a2 <+1426>:	be 59 c7 41 00	mov    $0x41c759,%esi
   0x00000000004110a7 <+1431>:	48 89 c7	mov    %rax,%rdi
   0x00000000004110aa <+1434>:	48 89 c5	mov    %rax,%rbp
   0x00000000004110ad <+1437>:	e8 6e 12 ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x00000000004110b2 <+1442>:	48 8d b4 24 50 01 00 00	lea    0x150(%rsp),%rsi
   0x00000000004110ba <+1450>:	48 89 ef	mov    %rbp,%rdi
   0x00000000004110bd <+1453>:	e8 5e 76 ff ff	callq  0x408720 <operator<< <double>(std::ostream&, FPWrap<double> const&)>
   0x00000000004110c2 <+1458>:	49 89 c7	mov    %rax,%r15
   0x00000000004110c5 <+1461>:	48 8b 00	mov    (%rax),%rax
   0x00000000004110c8 <+1464>:	48 8b 40 e8	mov    -0x18(%rax),%rax
   0x00000000004110cc <+1468>:	49 8b ac 07 f0 00 00 00	mov    0xf0(%r15,%rax,1),%rbp
   0x00000000004110d4 <+1476>:	48 85 ed	test   %rbp,%rbp
   0x00000000004110d7 <+1479>:	0f 84 ff 05 00 00	je     0x4116dc <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3020>
   0x00000000004110dd <+1485>:	80 7d 38 00	cmpb   $0x0,0x38(%rbp)
   0x00000000004110e1 <+1489>:	0f 84 f9 00 00 00	je     0x4111e0 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1744>
   0x00000000004110e7 <+1495>:	0f be 75 43	movsbl 0x43(%rbp),%esi
   0x00000000004110eb <+1499>:	4c 89 ff	mov    %r15,%rdi
   0x00000000004110ee <+1502>:	e8 0d 14 ff ff	callq  0x402500 <_ZNSo3putEc@plt>
   0x00000000004110f3 <+1507>:	48 89 c7	mov    %rax,%rdi
   0x00000000004110f6 <+1510>:	e8 85 12 ff ff	callq  0x402380 <_ZNSo5flushEv@plt>
   0x00000000004110fb <+1515>:	83 c3 01	add    $0x1,%ebx
   0x00000000004110fe <+1518>:	44 03 64 24 24	add    0x24(%rsp),%r12d
   0x0000000000411103 <+1523>:	39 5c 24 20	cmp    %ebx,0x20(%rsp)
   0x0000000000411107 <+1527>:	0f 85 a3 fc ff ff	jne    0x410db0 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+672>
   0x000000000041110d <+1533>:	ba 15 00 00 00	mov    $0x15,%edx
   0x0000000000411112 <+1538>:	be 65 c7 41 00	mov    $0x41c765,%esi
   0x0000000000411117 <+1543>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x000000000041111c <+1548>:	e8 ff 11 ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x0000000000411121 <+1553>:	48 8b 05 98 09 21 00	mov    0x210998(%rip),%rax        # 0x621ac0 <info_stream>
   0x0000000000411128 <+1560>:	48 8b 40 e8	mov    -0x18(%rax),%rax
   0x000000000041112c <+1564>:	48 8b 98 b0 1b 62 00	mov    0x621bb0(%rax),%rbx
   0x0000000000411133 <+1571>:	48 85 db	test   %rbx,%rbx
   0x0000000000411136 <+1574>:	0f 84 68 06 00 00	je     0x4117a4 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3220>
   0x000000000041113c <+1580>:	80 7b 38 00	cmpb   $0x0,0x38(%rbx)
   0x0000000000411140 <+1584>:	0f 84 f7 00 00 00	je     0x41123d <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1837>
   0x0000000000411146 <+1590>:	0f be 73 43	movsbl 0x43(%rbx),%esi
   0x000000000041114a <+1594>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x000000000041114f <+1599>:	e8 ac 13 ff ff	callq  0x402500 <_ZNSo3putEc@plt>
   0x0000000000411154 <+1604>:	48 89 c7	mov    %rax,%rdi
   0x0000000000411157 <+1607>:	e8 24 12 ff ff	callq  0x402380 <_ZNSo5flushEv@plt>
   0x000000000041115c <+1612>:	48 83 44 24 38 01	addq   $0x1,0x38(%rsp)
   0x0000000000411162 <+1618>:	f2 0f 10 7c 24 48	movsd  0x48(%rsp),%xmm7
   0x0000000000411168 <+1624>:	48 8b 44 24 38	mov    0x38(%rsp),%rax
   0x000000000041116d <+1629>:	48 3b 44 24 70	cmp    0x70(%rsp),%rax
   0x0000000000411172 <+1634>:	f2 41 0f 11 3e	movsd  %xmm7,(%r14)
   0x0000000000411177 <+1639>:	0f 85 ca fb ff ff	jne    0x410d47 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+567>
   0x000000000041117d <+1645>:	e9 f8 00 00 00	jmpq   0x41127a <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1898>
   0x0000000000411182 <+1650>:	66 0f 1f 44 00 00	nopw   0x0(%rax,%rax,1)
   0x0000000000411188 <+1656>:	48 c7 05 0d 0b 21 00 00 00 00 00	movq   $0x0,0x210b0d(%rip)        # 0x621ca0 <_ZN7GlobalsIdE3sumE>
   0x0000000000411193 <+1667>:	48 8b 44 24 40	mov    0x40(%rsp),%rax
   0x0000000000411198 <+1672>:	48 8b 4c 24 38	mov    0x38(%rsp),%rcx
   0x000000000041119d <+1677>:	83 04 88 01	addl   $0x1,(%rax,%rcx,4)
   0x00000000004111a1 <+1681>:	85 db	test   %ebx,%ebx
   0x00000000004111a3 <+1683>:	0f 84 29 05 00 00	je     0x4116d2 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3010>
   0x00000000004111a9 <+1689>:	f2 41 0f 10 06	movsd  (%r14),%xmm0
   0x00000000004111ae <+1694>:	bd 01 00 00 00	mov    $0x1,%ebp
   0x00000000004111b3 <+1699>:	f2 0f 5c 44 24 48	subsd  0x48(%rsp),%xmm0
   0x00000000004111b9 <+1705>:	db 6c 24 10	fldt   0x10(%rsp)
   0x00000000004111bd <+1709>:	66 0f 54 05 3b c4 00 00	andpd  0xc43b(%rip),%xmm0        # 0x41d600
   0x00000000004111c5 <+1717>:	f2 0f 11 04 24	movsd  %xmm0,(%rsp)
   0x00000000004111ca <+1722>:	dc 04 24	faddl  (%rsp)
   0x00000000004111cd <+1725>:	db 7c 24 10	fstpt  0x10(%rsp)
   0x00000000004111d1 <+1729>:	e9 e1 fc ff ff	jmpq   0x410eb7 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+935>
   0x00000000004111d6 <+1734>:	66 2e 0f 1f 84 00 00 00 00 00	nopw   %cs:0x0(%rax,%rax,1)
   0x00000000004111e0 <+1744>:	48 89 ef	mov    %rbp,%rdi
   0x00000000004111e3 <+1747>:	e8 78 0f ff ff	callq  0x402160 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>
   0x00000000004111e8 <+1752>:	48 8b 45 00	mov    0x0(%rbp),%rax
   0x00000000004111ec <+1756>:	be 0a 00 00 00	mov    $0xa,%esi
   0x00000000004111f1 <+1761>:	48 8b 40 30	mov    0x30(%rax),%rax
   0x00000000004111f5 <+1765>:	48 3d 60 51 40 00	cmp    $0x405160,%rax
   0x00000000004111fb <+1771>:	0f 84 ea fe ff ff	je     0x4110eb <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1499>
   0x0000000000411201 <+1777>:	48 89 ef	mov    %rbp,%rdi
   0x0000000000411204 <+1780>:	ff d0	callq  *%rax
   0x0000000000411206 <+1782>:	0f be f0	movsbl %al,%esi
   0x0000000000411209 <+1785>:	e9 dd fe ff ff	jmpq   0x4110eb <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1499>
   0x000000000041120e <+1790>:	66 90	xchg   %ax,%ax
   0x0000000000411210 <+1792>:	48 8b 8c 24 90 00 00 00	mov    0x90(%rsp),%rcx
   0x0000000000411218 <+1800>:	48 8b b4 24 b0 00 00 00	mov    0xb0(%rsp),%rsi
   0x0000000000411220 <+1808>:	66 0f ef c0	pxor   %xmm0,%xmm0
   0x0000000000411224 <+1812>:	e9 3c fd ff ff	jmpq   0x410f65 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1109>
   0x0000000000411229 <+1817>:	48 b8 00 00 00 00 00 00 20 00	movabs $0x20000000000000,%rax
   0x0000000000411233 <+1827>:	48 89 44 24 30	mov    %rax,0x30(%rsp)
   0x0000000000411238 <+1832>:	e9 60 fb ff ff	jmpq   0x410d9d <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+653>
   0x000000000041123d <+1837>:	48 89 df	mov    %rbx,%rdi
   0x0000000000411240 <+1840>:	e8 1b 0f ff ff	callq  0x402160 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>
   0x0000000000411245 <+1845>:	48 8b 03	mov    (%rbx),%rax
   0x0000000000411248 <+1848>:	be 0a 00 00 00	mov    $0xa,%esi
   0x000000000041124d <+1853>:	48 8b 40 30	mov    0x30(%rax),%rax
   0x0000000000411251 <+1857>:	48 3d 60 51 40 00	cmp    $0x405160,%rax
   0x0000000000411257 <+1863>:	0f 84 ed fe ff ff	je     0x41114a <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1594>
   0x000000000041125d <+1869>:	48 89 df	mov    %rbx,%rdi
   0x0000000000411260 <+1872>:	ff d0	callq  *%rax
   0x0000000000411262 <+1874>:	0f be f0	movsbl %al,%esi
   0x0000000000411265 <+1877>:	e9 e0 fe ff ff	jmpq   0x41114a <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+1594>
   0x000000000041126a <+1882>:	d9 ee	fldz   
   0x000000000041126c <+1884>:	66 0f ef ff	pxor   %xmm7,%xmm7
   0x0000000000411270 <+1888>:	db 7c 24 10	fstpt  0x10(%rsp)
   0x0000000000411274 <+1892>:	f3 0f 11 7c 24 78	movss  %xmm7,0x78(%rsp)
   0x000000000041127a <+1898>:	ba 2f 00 00 00	mov    $0x2f,%edx
   0x000000000041127f <+1903>:	be d8 c4 41 00	mov    $0x41c4d8,%esi
   0x0000000000411284 <+1908>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x0000000000411289 <+1913>:	e8 92 10 ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x000000000041128e <+1918>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x0000000000411293 <+1923>:	e8 d8 11 ff ff	callq  0x402470 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@plt>
   0x0000000000411298 <+1928>:	48 8d b4 24 82 00 00 00	lea    0x82(%rsp),%rsi
   0x00000000004112a0 <+1936>:	ba 01 00 00 00	mov    $0x1,%edx
   0x00000000004112a5 <+1941>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x00000000004112aa <+1946>:	c6 84 24 82 00 00 00 09	movb   $0x9,0x82(%rsp)
   0x00000000004112b2 <+1954>:	e8 69 10 ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x00000000004112b7 <+1959>:	ba 18 00 00 00	mov    $0x18,%edx
   0x00000000004112bc <+1964>:	be 7b c7 41 00	mov    $0x41c77b,%esi
   0x00000000004112c1 <+1969>:	48 89 c7	mov    %rax,%rdi
   0x00000000004112c4 <+1972>:	48 89 c3	mov    %rax,%rbx
   0x00000000004112c7 <+1975>:	e8 54 10 ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x00000000004112cc <+1980>:	48 8b 74 24 68	mov    0x68(%rsp),%rsi
   0x00000000004112d1 <+1985>:	48 89 df	mov    %rbx,%rdi
   0x00000000004112d4 <+1988>:	e8 47 11 ff ff	callq  0x402420 <_ZNSo9_M_insertImEERSoT_@plt>
   0x00000000004112d9 <+1993>:	48 89 c7	mov    %rax,%rdi
   0x00000000004112dc <+1996>:	e8 8f 11 ff ff	callq  0x402470 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@plt>
   0x00000000004112e1 <+2001>:	48 8d b4 24 83 00 00 00	lea    0x83(%rsp),%rsi
   0x00000000004112e9 <+2009>:	ba 01 00 00 00	mov    $0x1,%edx
   0x00000000004112ee <+2014>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x00000000004112f3 <+2019>:	c6 84 24 83 00 00 00 09	movb   $0x9,0x83(%rsp)
   0x00000000004112fb <+2027>:	e8 20 10 ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x0000000000411300 <+2032>:	ba 1a 00 00 00	mov    $0x1a,%edx
   0x0000000000411305 <+2037>:	be 94 c7 41 00	mov    $0x41c794,%esi
   0x000000000041130a <+2042>:	48 89 c7	mov    %rax,%rdi
   0x000000000041130d <+2045>:	48 89 c3	mov    %rax,%rbx
   0x0000000000411310 <+2048>:	e8 0b 10 ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x0000000000411315 <+2053>:	8b 74 24 20	mov    0x20(%rsp),%esi
   0x0000000000411319 <+2057>:	48 89 df	mov    %rbx,%rdi
   0x000000000041131c <+2060>:	e8 df 0d ff ff	callq  0x402100 <_ZNSolsEi@plt>
   0x0000000000411321 <+2065>:	48 89 c7	mov    %rax,%rdi
   0x0000000000411324 <+2068>:	e8 47 11 ff ff	callq  0x402470 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@plt>
   0x0000000000411329 <+2073>:	48 8d b4 24 84 00 00 00	lea    0x84(%rsp),%rsi
   0x0000000000411331 <+2081>:	ba 01 00 00 00	mov    $0x1,%edx
   0x0000000000411336 <+2086>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x000000000041133b <+2091>:	c6 84 24 84 00 00 00 09	movb   $0x9,0x84(%rsp)
   0x0000000000411343 <+2099>:	e8 d8 0f ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x0000000000411348 <+2104>:	ba 0c 00 00 00	mov    $0xc,%edx
   0x000000000041134d <+2109>:	be af c7 41 00	mov    $0x41c7af,%esi
   0x0000000000411352 <+2114>:	48 89 c7	mov    %rax,%rdi
   0x0000000000411355 <+2117>:	48 89 c3	mov    %rax,%rbx
   0x0000000000411358 <+2120>:	e8 c3 0f ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x000000000041135d <+2125>:	8b 74 24 54	mov    0x54(%rsp),%esi
   0x0000000000411361 <+2129>:	48 89 df	mov    %rbx,%rdi
   0x0000000000411364 <+2132>:	e8 97 0d ff ff	callq  0x402100 <_ZNSolsEi@plt>
   0x0000000000411369 <+2137>:	48 89 c7	mov    %rax,%rdi
   0x000000000041136c <+2140>:	e8 ff 10 ff ff	callq  0x402470 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@plt>
   0x0000000000411371 <+2145>:	48 8d 84 24 00 01 00 00	lea    0x100(%rsp),%rax
   0x0000000000411379 <+2153>:	48 c7 84 24 f8 00 00 00 00 00 00 00	movq   $0x0,0xf8(%rsp)
   0x0000000000411385 <+2165>:	c6 84 24 00 01 00 00 00	movb   $0x0,0x100(%rsp)
   0x000000000041138d <+2173>:	48 89 84 24 f0 00 00 00	mov    %rax,0xf0(%rsp)
   0x0000000000411395 <+2181>:	8b 44 24 7c	mov    0x7c(%rsp),%eax
   0x0000000000411399 <+2185>:	83 f8 01	cmp    $0x1,%eax
   0x000000000041139c <+2188>:	0f 84 c7 03 00 00	je     0x411769 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3161>
   0x00000000004113a2 <+2194>:	0f 82 50 04 00 00	jb     0x4117f8 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3304>
   0x00000000004113a8 <+2200>:	83 f8 02	cmp    $0x2,%eax
   0x00000000004113ab <+2203>:	0f 84 30 04 00 00	je     0x4117e1 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3281>
   0x00000000004113b1 <+2209>:	83 f8 03	cmp    $0x3,%eax
   0x00000000004113b4 <+2212>:	75 12	jne    0x4113c8 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2232>
   0x00000000004113b6 <+2214>:	48 8d bc 24 f0 00 00 00	lea    0xf0(%rsp),%rdi
   0x00000000004113be <+2222>:	be bc c7 41 00	mov    $0x41c7bc,%esi
   0x00000000004113c3 <+2227>:	e8 e8 0e ff ff	callq  0x4022b0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEPKc@plt>
   0x00000000004113c8 <+2232>:	48 8d b4 24 85 00 00 00	lea    0x85(%rsp),%rsi
   0x00000000004113d0 <+2240>:	ba 01 00 00 00	mov    $0x1,%edx
   0x00000000004113d5 <+2245>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x00000000004113da <+2250>:	c6 84 24 85 00 00 00 09	movb   $0x9,0x85(%rsp)
   0x00000000004113e2 <+2258>:	e8 39 0f ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x00000000004113e7 <+2263>:	ba 06 00 00 00	mov    $0x6,%edx
   0x00000000004113ec <+2268>:	be ea c7 41 00	mov    $0x41c7ea,%esi
   0x00000000004113f1 <+2273>:	48 89 c7	mov    %rax,%rdi
   0x00000000004113f4 <+2276>:	48 89 c3	mov    %rax,%rbx
   0x00000000004113f7 <+2279>:	e8 24 0f ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x00000000004113fc <+2284>:	48 8b 94 24 f8 00 00 00	mov    0xf8(%rsp),%rdx
   0x0000000000411404 <+2292>:	48 8b b4 24 f0 00 00 00	mov    0xf0(%rsp),%rsi
   0x000000000041140c <+2300>:	48 89 df	mov    %rbx,%rdi
   0x000000000041140f <+2303>:	e8 0c 0f ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x0000000000411414 <+2308>:	48 89 c7	mov    %rax,%rdi
   0x0000000000411417 <+2311>:	e8 54 10 ff ff	callq  0x402470 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@plt>
   0x000000000041141c <+2316>:	48 8b 2d c5 02 21 00	mov    0x2102c5(%rip),%rbp        # 0x6216e8 <_ZTId@@CXXABI_1.3+8>
   0x0000000000411423 <+2323>:	31 c0	xor    %eax,%eax
   0x0000000000411425 <+2325>:	48 8d b4 24 86 00 00 00	lea    0x86(%rsp),%rsi
   0x000000000041142d <+2333>:	ba 01 00 00 00	mov    $0x1,%edx
   0x0000000000411432 <+2338>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x0000000000411437 <+2343>:	80 7d 00 2a	cmpb   $0x2a,0x0(%rbp)
   0x000000000041143b <+2347>:	c6 84 24 86 00 00 00 09	movb   $0x9,0x86(%rsp)
   0x0000000000411443 <+2355>:	0f 94 c0	sete   %al
   0x0000000000411446 <+2358>:	48 01 c5	add    %rax,%rbp
   0x0000000000411449 <+2361>:	e8 d2 0e ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x000000000041144e <+2366>:	ba 12 00 00 00	mov    $0x12,%edx
   0x0000000000411453 <+2371>:	be f1 c7 41 00	mov    $0x41c7f1,%esi
   0x0000000000411458 <+2376>:	48 89 c7	mov    %rax,%rdi
   0x000000000041145b <+2379>:	48 89 c3	mov    %rax,%rbx
   0x000000000041145e <+2382>:	e8 bd 0e ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x0000000000411463 <+2387>:	48 85 ed	test   %rbp,%rbp
   0x0000000000411466 <+2390>:	0f 84 a3 03 00 00	je     0x41180f <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3327>
   0x000000000041146c <+2396>:	48 89 ef	mov    %rbp,%rdi
   0x000000000041146f <+2399>:	e8 cc 0d ff ff	callq  0x402240 <strlen@plt>
   0x0000000000411474 <+2404>:	48 89 ee	mov    %rbp,%rsi
   0x0000000000411477 <+2407>:	48 89 c2	mov    %rax,%rdx
   0x000000000041147a <+2410>:	48 89 df	mov    %rbx,%rdi
   0x000000000041147d <+2413>:	e8 9e 0e ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x0000000000411482 <+2418>:	48 89 df	mov    %rbx,%rdi
   0x0000000000411485 <+2421>:	e8 e6 0f ff ff	callq  0x402470 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@plt>
   0x000000000041148a <+2426>:	48 8b 5c 24 40	mov    0x40(%rsp),%rbx
   0x000000000041148f <+2431>:	48 39 5c 24 58	cmp    %rbx,0x58(%rsp)
   0x0000000000411494 <+2436>:	0f 84 1a 01 00 00	je     0x4115b4 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2724>
   0x000000000041149a <+2442>:	4c 8b 64 24 40	mov    0x40(%rsp),%r12
   0x000000000041149f <+2447>:	31 ed	xor    %ebp,%ebp
   0x00000000004114a1 <+2449>:	eb 2c	jmp    0x4114cf <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2495>
   0x00000000004114a3 <+2451>:	0f 1f 44 00 00	nopl   0x0(%rax,%rax,1)
   0x00000000004114a8 <+2456>:	0f be 73 43	movsbl 0x43(%rbx),%esi
   0x00000000004114ac <+2460>:	4c 89 ef	mov    %r13,%rdi
   0x00000000004114af <+2463>:	e8 4c 10 ff ff	callq  0x402500 <_ZNSo3putEc@plt>
   0x00000000004114b4 <+2468>:	48 89 c7	mov    %rax,%rdi
   0x00000000004114b7 <+2471>:	e8 c4 0e ff ff	callq  0x402380 <_ZNSo5flushEv@plt>
   0x00000000004114bc <+2476>:	48 83 c5 01	add    $0x1,%rbp
   0x00000000004114c0 <+2480>:	49 83 c4 04	add    $0x4,%r12
   0x00000000004114c4 <+2484>:	4c 39 64 24 58	cmp    %r12,0x58(%rsp)
   0x00000000004114c9 <+2489>:	0f 84 e5 00 00 00	je     0x4115b4 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2724>
   0x00000000004114cf <+2495>:	48 8b 84 24 90 00 00 00	mov    0x90(%rsp),%rax
   0x00000000004114d7 <+2503>:	ba 0c 00 00 00	mov    $0xc,%edx
   0x00000000004114dc <+2508>:	be 04 c8 41 00	mov    $0x41c804,%esi
   0x00000000004114e1 <+2513>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x00000000004114e6 <+2518>:	45 8b 2c 24	mov    (%r12),%r13d
   0x00000000004114ea <+2522>:	f2 0f 10 04 e8	movsd  (%rax,%rbp,8),%xmm0
   0x00000000004114ef <+2527>:	48 8b 84 24 b0 00 00 00	mov    0xb0(%rsp),%rax
   0x00000000004114f7 <+2535>:	f2 0f 59 04 e8	mulsd  (%rax,%rbp,8),%xmm0
   0x00000000004114fc <+2540>:	66 48 0f 7e c3	movq   %xmm0,%rbx
   0x0000000000411501 <+2545>:	48 c1 eb 34	shr    $0x34,%rbx
   0x0000000000411505 <+2549>:	81 e3 ff 07 00 00	and    $0x7ff,%ebx
   0x000000000041150b <+2555>:	81 eb ff 03 00 00	sub    $0x3ff,%ebx
   0x0000000000411511 <+2561>:	e8 0a 0e ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x0000000000411516 <+2566>:	89 ee	mov    %ebp,%esi
   0x0000000000411518 <+2568>:	bf c0 1a 62 00	mov    $0x621ac0,%edi
   0x000000000041151d <+2573>:	e8 de 0b ff ff	callq  0x402100 <_ZNSolsEi@plt>
   0x0000000000411522 <+2578>:	ba 0d 00 00 00	mov    $0xd,%edx
   0x0000000000411527 <+2583>:	be 11 c8 41 00	mov    $0x41c811,%esi
   0x000000000041152c <+2588>:	48 89 c7	mov    %rax,%rdi
   0x000000000041152f <+2591>:	49 89 c6	mov    %rax,%r14
   0x0000000000411532 <+2594>:	e8 e9 0d ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x0000000000411537 <+2599>:	44 89 ee	mov    %r13d,%esi
   0x000000000041153a <+2602>:	4c 89 f7	mov    %r14,%rdi
   0x000000000041153d <+2605>:	e8 de 0e ff ff	callq  0x402420 <_ZNSo9_M_insertImEERSoT_@plt>
   0x0000000000411542 <+2610>:	ba 33 00 00 00	mov    $0x33,%edx
   0x0000000000411547 <+2615>:	be 08 c5 41 00	mov    $0x41c508,%esi
   0x000000000041154c <+2620>:	48 89 c7	mov    %rax,%rdi
   0x000000000041154f <+2623>:	49 89 c5	mov    %rax,%r13
   0x0000000000411552 <+2626>:	e8 c9 0d ff ff	callq  0x402320 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
   0x0000000000411557 <+2631>:	89 de	mov    %ebx,%esi
   0x0000000000411559 <+2633>:	4c 89 ef	mov    %r13,%rdi
   0x000000000041155c <+2636>:	e8 bf 0e ff ff	callq  0x402420 <_ZNSo9_M_insertImEERSoT_@plt>
   0x0000000000411561 <+2641>:	49 89 c5	mov    %rax,%r13
   0x0000000000411564 <+2644>:	48 8b 00	mov    (%rax),%rax
   0x0000000000411567 <+2647>:	48 8b 40 e8	mov    -0x18(%rax),%rax
   0x000000000041156b <+2651>:	49 8b 9c 05 f0 00 00 00	mov    0xf0(%r13,%rax,1),%rbx
   0x0000000000411573 <+2659>:	48 85 db	test   %rbx,%rbx
   0x0000000000411576 <+2662>:	74 37	je     0x4115af <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2719>
   0x0000000000411578 <+2664>:	80 7b 38 00	cmpb   $0x0,0x38(%rbx)
   0x000000000041157c <+2668>:	0f 85 26 ff ff ff	jne    0x4114a8 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2456>
   0x0000000000411582 <+2674>:	48 89 df	mov    %rbx,%rdi
   0x0000000000411585 <+2677>:	e8 d6 0b ff ff	callq  0x402160 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>
   0x000000000041158a <+2682>:	48 8b 03	mov    (%rbx),%rax
   0x000000000041158d <+2685>:	be 0a 00 00 00	mov    $0xa,%esi
   0x0000000000411592 <+2690>:	48 8b 40 30	mov    0x30(%rax),%rax
   0x0000000000411596 <+2694>:	48 3d 60 51 40 00	cmp    $0x405160,%rax
   0x000000000041159c <+2700>:	0f 84 0a ff ff ff	je     0x4114ac <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2460>
   0x00000000004115a2 <+2706>:	48 89 df	mov    %rbx,%rdi
   0x00000000004115a5 <+2709>:	ff d0	callq  *%rax
   0x00000000004115a7 <+2711>:	0f be f0	movsbl %al,%esi
   0x00000000004115aa <+2714>:	e9 fd fe ff ff	jmpq   0x4114ac <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2460>
   0x00000000004115af <+2719>:	e8 6c 0b ff ff	callq  0x402120 <_ZSt16__throw_bad_castv@plt>
   0x00000000004115b4 <+2724>:	48 8d bc 24 10 01 00 00	lea    0x110(%rsp),%rdi
   0x00000000004115bc <+2732>:	be d0 cc 41 00	mov    $0x41ccd0,%esi
   0x00000000004115c1 <+2737>:	e8 8a 1e ff ff	callq  0x403450 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(char const*, std::allocator<char> const&)>
   0x00000000004115c6 <+2742>:	48 8b 84 24 90 00 00 00	mov    0x90(%rsp),%rax
   0x00000000004115ce <+2750>:	f2 0f 10 05 72 c0 00 00	movsd  0xc072(%rip),%xmm0        # 0x41d648
   0x00000000004115d6 <+2758>:	66 0f 2e 00	ucomisd (%rax),%xmm0
   0x00000000004115da <+2762>:	0f 8a db 00 00 00	jp     0x4116bb <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2987>
   0x00000000004115e0 <+2768>:	0f 85 d5 00 00 00	jne    0x4116bb <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2987>
   0x00000000004115e6 <+2774>:	48 8d bc 24 10 01 00 00	lea    0x110(%rsp),%rdi
   0x00000000004115ee <+2782>:	be 1f c8 41 00	mov    $0x41c81f,%esi
   0x00000000004115f3 <+2787>:	e8 f8 0b ff ff	callq  0x4021f0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKc@plt>
   0x00000000004115f8 <+2792>:	48 8b b4 24 10 01 00 00	mov    0x110(%rsp),%rsi
   0x0000000000411600 <+2800>:	48 8b 7c 24 60	mov    0x60(%rsp),%rdi
   0x0000000000411605 <+2805>:	48 89 f2	mov    %rsi,%rdx
   0x0000000000411608 <+2808>:	48 03 94 24 18 01 00 00	add    0x118(%rsp),%rdx
   0x0000000000411610 <+2816>:	48 89 f8	mov    %rdi,%rax
   0x0000000000411613 <+2819>:	48 83 c0 10	add    $0x10,%rax
   0x0000000000411617 <+2823>:	48 89 07	mov    %rax,(%rdi)
   0x000000000041161a <+2826>:	e8 81 1d ff ff	callq  0x4033a0 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char*>(char*, char*, std::forward_iterator_tag)>
   0x000000000041161f <+2831>:	db 6c 24 10	fldt   0x10(%rsp)
   0x0000000000411623 <+2835>:	48 8b 44 24 60	mov    0x60(%rsp),%rax
   0x0000000000411628 <+2840>:	48 8b bc 24 10 01 00 00	mov    0x110(%rsp),%rdi
   0x0000000000411630 <+2848>:	db 78 20	fstpt  0x20(%rax)
   0x0000000000411633 <+2851>:	d9 44 24 78	flds   0x78(%rsp)
   0x0000000000411637 <+2855>:	db 78 30	fstpt  0x30(%rax)
   0x000000000041163a <+2858>:	48 8d 84 24 20 01 00 00	lea    0x120(%rsp),%rax
   0x0000000000411642 <+2866>:	48 39 c7	cmp    %rax,%rdi
   0x0000000000411645 <+2869>:	74 05	je     0x41164c <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2876>
   0x0000000000411647 <+2871>:	e8 04 0b ff ff	callq  0x402150 <_ZdlPv@plt>
   0x000000000041164c <+2876>:	48 8b bc 24 f0 00 00 00	mov    0xf0(%rsp),%rdi
   0x0000000000411654 <+2884>:	48 8d 84 24 00 01 00 00	lea    0x100(%rsp),%rax
   0x000000000041165c <+2892>:	48 39 c7	cmp    %rax,%rdi
   0x000000000041165f <+2895>:	74 05	je     0x411666 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2902>
   0x0000000000411661 <+2897>:	e8 ea 0a ff ff	callq  0x402150 <_ZdlPv@plt>
   0x0000000000411666 <+2902>:	48 8b bc 24 b0 00 00 00	mov    0xb0(%rsp),%rdi
   0x000000000041166e <+2910>:	48 85 ff	test   %rdi,%rdi
   0x0000000000411671 <+2913>:	74 05	je     0x411678 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2920>
   0x0000000000411673 <+2915>:	e8 d8 0a ff ff	callq  0x402150 <_ZdlPv@plt>
   0x0000000000411678 <+2920>:	48 8b bc 24 90 00 00 00	mov    0x90(%rsp),%rdi
   0x0000000000411680 <+2928>:	48 85 ff	test   %rdi,%rdi
   0x0000000000411683 <+2931>:	74 05	je     0x41168a <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2938>
   0x0000000000411685 <+2933>:	e8 c6 0a ff ff	callq  0x402150 <_ZdlPv@plt>
   0x000000000041168a <+2938>:	48 8b 44 24 40	mov    0x40(%rsp),%rax
   0x000000000041168f <+2943>:	48 85 c0	test   %rax,%rax
   0x0000000000411692 <+2946>:	74 08	je     0x41169c <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2956>
   0x0000000000411694 <+2948>:	48 89 c7	mov    %rax,%rdi
   0x0000000000411697 <+2951>:	e8 b4 0a ff ff	callq  0x402150 <_ZdlPv@plt>
   0x000000000041169c <+2956>:	48 8b 44 24 60	mov    0x60(%rsp),%rax
   0x00000000004116a1 <+2961>:	48 81 c4 78 01 00 00	add    $0x178,%rsp
   0x00000000004116a8 <+2968>:	5b	pop    %rbx
   0x00000000004116a9 <+2969>:	5d	pop    %rbp
   0x00000000004116aa <+2970>:	41 5c	pop    %r12
   0x00000000004116ac <+2972>:	41 5d	pop    %r13
   0x00000000004116ae <+2974>:	41 5e	pop    %r14
   0x00000000004116b0 <+2976>:	41 5f	pop    %r15
   0x00000000004116b2 <+2978>:	c3	retq   
   0x00000000004116b3 <+2979>:	48 89 c3	mov    %rax,%rbx
   0x00000000004116b6 <+2982>:	e9 ee 00 00 00	jmpq   0x4117a9 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3225>
   0x00000000004116bb <+2987>:	48 8d bc 24 10 01 00 00	lea    0x110(%rsp),%rdi
   0x00000000004116c3 <+2995>:	be 22 c8 41 00	mov    $0x41c822,%esi
   0x00000000004116c8 <+3000>:	e8 23 0b ff ff	callq  0x4021f0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKc@plt>
   0x00000000004116cd <+3005>:	e9 26 ff ff ff	jmpq   0x4115f8 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2792>
   0x00000000004116d2 <+3010>:	bd 01 00 00 00	mov    $0x1,%ebp
   0x00000000004116d7 <+3015>:	e9 db f7 ff ff	jmpq   0x410eb7 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+935>
   0x00000000004116dc <+3020>:	e8 3f 0a ff ff	callq  0x402120 <_ZSt16__throw_bad_castv@plt>
   0x00000000004116e1 <+3025>:	48 89 c3	mov    %rax,%rbx
   0x00000000004116e4 <+3028>:	eb 05	jmp    0x4116eb <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3035>
   0x00000000004116e6 <+3030>:	48 89 c3	mov    %rax,%rbx
   0x00000000004116e9 <+3033>:	eb 1a	jmp    0x411705 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3061>
   0x00000000004116eb <+3035>:	48 8b bc 24 d0 00 00 00	mov    0xd0(%rsp),%rdi
   0x00000000004116f3 <+3043>:	48 8d 84 24 e0 00 00 00	lea    0xe0(%rsp),%rax
   0x00000000004116fb <+3051>:	48 39 c7	cmp    %rax,%rdi
   0x00000000004116fe <+3054>:	74 05	je     0x411705 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3061>
   0x0000000000411700 <+3056>:	e8 4b 0a ff ff	callq  0x402150 <_ZdlPv@plt>
   0x0000000000411705 <+3061>:	48 8b bc 24 b0 00 00 00	mov    0xb0(%rsp),%rdi
   0x000000000041170d <+3069>:	48 85 ff	test   %rdi,%rdi
   0x0000000000411710 <+3072>:	74 05	je     0x411717 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3079>
   0x0000000000411712 <+3074>:	e8 39 0a ff ff	callq  0x402150 <_ZdlPv@plt>
   0x0000000000411717 <+3079>:	48 8b bc 24 90 00 00 00	mov    0x90(%rsp),%rdi
   0x000000000041171f <+3087>:	48 85 ff	test   %rdi,%rdi
   0x0000000000411722 <+3090>:	74 05	je     0x411729 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3097>
   0x0000000000411724 <+3092>:	e8 27 0a ff ff	callq  0x402150 <_ZdlPv@plt>
   0x0000000000411729 <+3097>:	48 8b 44 24 40	mov    0x40(%rsp),%rax
   0x000000000041172e <+3102>:	48 85 c0	test   %rax,%rax
   0x0000000000411731 <+3105>:	74 08	je     0x41173b <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3115>
   0x0000000000411733 <+3107>:	48 89 c7	mov    %rax,%rdi
   0x0000000000411736 <+3110>:	e8 15 0a ff ff	callq  0x402150 <_ZdlPv@plt>
   0x000000000041173b <+3115>:	48 89 df	mov    %rbx,%rdi
   0x000000000041173e <+3118>:	e8 8d 0d ff ff	callq  0x4024d0 <_Unwind_Resume@plt>
   0x0000000000411743 <+3123>:	48 89 c3	mov    %rax,%rbx
   0x0000000000411746 <+3126>:	eb cf	jmp    0x411717 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3079>
   0x0000000000411748 <+3128>:	48 89 c3	mov    %rax,%rbx
   0x000000000041174b <+3131>:	eb dc	jmp    0x411729 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3097>
   0x000000000041174d <+3133>:	e8 0e 0d ff ff	callq  0x402460 <_ZSt17__throw_bad_allocv@plt>
   0x0000000000411752 <+3138>:	48 c7 44 24 40 00 00 00 00	movq   $0x0,0x40(%rsp)
   0x000000000041175b <+3147>:	48 c7 44 24 58 00 00 00 00	movq   $0x0,0x58(%rsp)
   0x0000000000411764 <+3156>:	e9 25 f4 ff ff	jmpq   0x410b8e <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+126>
   0x0000000000411769 <+3161>:	48 8d bc 24 f0 00 00 00	lea    0xf0(%rsp),%rdi
   0x0000000000411771 <+3169>:	be c1 c7 41 00	mov    $0x41c7c1,%esi
   0x0000000000411776 <+3174>:	e8 35 0b ff ff	callq  0x4022b0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEPKc@plt>
   0x000000000041177b <+3179>:	e9 48 fc ff ff	jmpq   0x4113c8 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2232>
   0x0000000000411780 <+3184>:	48 89 c3	mov    %rax,%rbx
   0x0000000000411783 <+3187>:	eb 3e	jmp    0x4117c3 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3251>
   0x0000000000411785 <+3189>:	db 6c 24 10	fldt   0x10(%rsp)
   0x0000000000411789 <+3193>:	31 ed	xor    %ebp,%ebp
   0x000000000041178b <+3195>:	66 0f 54 05 6d be 00 00	andpd  0xbe6d(%rip),%xmm0        # 0x41d600
   0x0000000000411793 <+3203>:	f2 0f 11 04 24	movsd  %xmm0,(%rsp)
   0x0000000000411798 <+3208>:	dc 04 24	faddl  (%rsp)
   0x000000000041179b <+3211>:	db 7c 24 10	fstpt  0x10(%rsp)
   0x000000000041179f <+3215>:	e9 13 f7 ff ff	jmpq   0x410eb7 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+935>
   0x00000000004117a4 <+3220>:	e8 77 09 ff ff	callq  0x402120 <_ZSt16__throw_bad_castv@plt>
   0x00000000004117a9 <+3225>:	48 8b bc 24 10 01 00 00	mov    0x110(%rsp),%rdi
   0x00000000004117b1 <+3233>:	48 8d 84 24 20 01 00 00	lea    0x120(%rsp),%rax
   0x00000000004117b9 <+3241>:	48 39 c7	cmp    %rax,%rdi
   0x00000000004117bc <+3244>:	74 05	je     0x4117c3 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3251>
   0x00000000004117be <+3246>:	e8 8d 09 ff ff	callq  0x402150 <_ZdlPv@plt>
   0x00000000004117c3 <+3251>:	48 8b bc 24 f0 00 00 00	mov    0xf0(%rsp),%rdi
   0x00000000004117cb <+3259>:	48 8d 84 24 00 01 00 00	lea    0x100(%rsp),%rax
   0x00000000004117d3 <+3267>:	48 39 c7	cmp    %rax,%rdi
   0x00000000004117d6 <+3270>:	0f 85 24 ff ff ff	jne    0x411700 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3056>
   0x00000000004117dc <+3276>:	e9 24 ff ff ff	jmpq   0x411705 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+3061>
   0x00000000004117e1 <+3281>:	48 8d bc 24 f0 00 00 00	lea    0xf0(%rsp),%rdi
   0x00000000004117e9 <+3289>:	be cc c7 41 00	mov    $0x41c7cc,%esi
   0x00000000004117ee <+3294>:	e8 bd 0a ff ff	callq  0x4022b0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEPKc@plt>
   0x00000000004117f3 <+3299>:	e9 d0 fb ff ff	jmpq   0x4113c8 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2232>
   0x00000000004117f8 <+3304>:	48 8d bc 24 f0 00 00 00	lea    0xf0(%rsp),%rdi
   0x0000000000411800 <+3312>:	be df c7 41 00	mov    $0x41c7df,%esi
   0x0000000000411805 <+3317>:	e8 a6 0a ff ff	callq  0x4022b0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEPKc@plt>
   0x000000000041180a <+3322>:	e9 b9 fb ff ff	jmpq   0x4113c8 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2232>
   0x000000000041180f <+3327>:	48 8b 03	mov    (%rbx),%rax
   0x0000000000411812 <+3330>:	48 89 df	mov    %rbx,%rdi
   0x0000000000411815 <+3333>:	48 03 78 e8	add    -0x18(%rax),%rdi
   0x0000000000411819 <+3337>:	8b 77 20	mov    0x20(%rdi),%esi
   0x000000000041181c <+3340>:	83 ce 01	or     $0x1,%esi
   0x000000000041181f <+3343>:	e8 cc 0b ff ff	callq  0x4023f0 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
   0x0000000000411824 <+3348>:	e9 59 fc ff ff	jmpq   0x411482 <FPTests::DoOrthoPerturbTest<double, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}>(int, int, unsigned long, void fpTestSuite::DoTests<double>(unsigned long, unsigned long, unsigned long, double, double, Vector<double>::sort_t, double, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::pair<long double, long double>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::pair<long double, long double> > > >&, int)::{lambda()#1}, Vector<double>::sort_t)+2418>
End of assembler dump.
