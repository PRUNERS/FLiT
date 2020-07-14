ECHO            := printf "%s"
ECHO_NEWLINE    := printf "%s\n"
NCOLORS         := $(shell tput colors)

# A Makefile function
# @param 1: color (e.g., BLUE or GREEN)
# @param 2: message to be printed in the color
bold_color_noline = $(call print_impl,$(ECHO),$(BASH_BOLD)$(BASH_$1),$2)
bold_color        = $(call print_impl,$(ECHO_NEWLINE),$(BASH_BOLD)$(BASH_$1),$2)
color_out_noline  = $(call print_impl,$(ECHO),$(BASH_$1),$2)
color_out         = $(call print_impl,$(ECHO_NEWLINE),$(BASH_$1),$2)

# implementation
# @param 1: print command
# @param 2: terminal settings (e.g., $(BASH_BLUE) or $(BASH_GREEN))
# @param 3: message to be printed in the color
print_impl = \
  if [ -t 1 ] && [ $(NCOLORS) -ge 8 ]; then \
    $1 "$2$3$(BASH_CLEAR)"; \
  else \
    $1 "$3"; \
  fi

BASH_CLEAR    := $(shell tput sgr0)
BASH_BOLD     := $(shell tput bold)
BASH_BLACK    := $(shell tput setaf 0)
BASH_RED      := $(shell tput setaf 1)
BASH_GREEN    := $(shell tput setaf 2)
BASH_YELLOW   := $(shell tput setaf 3)
BASH_BLUE     := $(shell tput setaf 4)
BASH_PURPLE   := $(shell tput setaf 5)
BASH_CYAN     := $(shell tput setaf 6)
BASH_WHITE    := $(shell tput setaf 7)
BASH_BROWN    := $(shell tput setaf 94)
BASH_GRAY     := $(shell tput setaf 245)
BASH_GREY     := $(BASH_GRAY)
BASH_DARKGRAY := $(shell tput setaf 240)
BASH_DARKGREY := $(BASH_DARKGRAY)
BASH_LIGHTGRAY:= $(shell tput setaf 250)
BASH_LIGHTGREY:= $(BASH_LIGHTGRAY)

