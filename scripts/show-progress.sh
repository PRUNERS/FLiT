#!/bin/bash

cd results
echo ...
ls -tr | \
  grep "\.csv$" | \
  tail -n 10
echo
echo -n "$(find . -name \*-out -type f -not -executable -not -empty | wc -l)"
echo -n " of "
echo -n "$(find . -executable -type f | wc -l)"
echo -n ", "
echo -n "$(find . -name \*-out -type f -empty -exec fuser -a {} \+ 2>&1 | \
           grep ":$" | \
           wc -l)"
echo -n " were skipped - "
echo -n "$(for file in $(find . -type f -name \*-comparison.csv); do \
             cat $file | \
               awk -F, \{print\ \$11\} | \
               tail -n +2; \
           done |
             grep -v "^0$" | wc -l)"
echo " bad answers found"
echo
cd ..

rsync -a \
  --out-format "%n" \
  --delete \
  --include \*-out \
  --include \*.csv \
  --exclude \* \
  results/ \
  ~/mfem-results/

echo

rsync -a \
  --out-format "%n" \
  $0 \
  ~/mfem-results/

#rsync -a \
#  --info=del1,name1 \
#  --delete \
#  --include \*-out \
#  --include \*.csv \
#  --exclude \* \
#  results/ \
#  ~/mfem-results/
#rsync \
#  --info=name1 \
#  $0 \
#  ~/mfem-results/

