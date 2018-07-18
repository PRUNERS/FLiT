# A Makefile function
# @param 1: color (e.g. BLUE or GREEN)
# @param 2: message to be printed in the color
color_out = @if [ -t 1 ]; then \
               /bin/echo -e "$(BASH_$1)$2$(BASH_CLEAR)"; \
             else \
               /bin/echo "$2"; \
             fi

BASH_CLEAR    := \e[0m
BASH_BLACK    := \e[0;30m
BASH_BROWN    := \e[0;33m
BASH_GRAY     := \e[0;37m
BASH_DARKGRAY := \e[1;30m
BASH_RED      := \e[1;31m
BASH_GREEN    := \e[1;32m
BASH_YELLOW   := \e[1;33m
BASH_BLUE     := \e[1;34m
BASH_PURPLE   := \e[1;35m
BASH_CYAN     := \e[1;36m
BASH_WHITE    := \e[1;37m

