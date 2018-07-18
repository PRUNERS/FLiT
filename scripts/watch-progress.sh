#!/bin/bash

cd results
watch -n 2 \
  'echo ...; \
   ls -tr | \
     tail -n 10; \
   echo; \
   echo -n "$(find . -name \*-out -type f -not -executable | wc -l)"; \
   echo -n " of "; \
   echo -n "$(find . -executable -type f | wc -l)"; \
   echo -n ", "; \
   echo -n "$(find . -name \*-out -type f -empty -not -executable | \
              tail -n +2 | \
              wc -l)"; \
   echo -n " were skipped - "; \
   echo -n "$(for file in $(find . -type f -name \*-comparison.csv); do \
                cat $file | \
                  awk -F, \{print\ \$11\} | \
                  tail -n +2; \
              done |
                grep -v "^0$" | wc -l)"; \
   echo -n " bad answers found"; \
   echo'


