# These are the fp affecting switches for CUDA (7.5).
# We will naively apply these (though the docs say
# that, for instance, --use_fast_math implies
# --ftz=true --prec-div=false --prec-sqrt=false
# --fmad=true.

FASTMC := --use_fast_math
FMADFC  := --fmad=false
FMADTC  := --fmad=true
FTZFC   := --ftz=false
FTZTC   := --ftz=true
PRECDFC := --prec-div=false
PRECDTC := --prec-div=true
PRECSFC := --prec-sqrt=false
PRECSTC := --prec-sqrt=true

CUSWITCHES := FASTMC FTZTC FTZFC PRECDTC PRECDFC PRECSTC PRECSFC FMADTC FMADFC
