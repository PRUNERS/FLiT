ECHO            := printf "%s"
ECHO_NEWLINE    := printf "%s\n"

# Call TPUT in the shell for one argument ($1)
TPUT             = $(shell tput $1 2>/dev/null)
NCOLORS         := $(call TPUT,colors)

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

BASH_CLEAR    := $(call TPUT,sgr0)
BASH_BOLD     := $(call TPUT,bold)
BASH_BLACK    := $(call TPUT,setaf 0)
BASH_RED      := $(call TPUT,setaf 1)
BASH_GREEN    := $(call TPUT,setaf 2)
BASH_YELLOW   := $(call TPUT,setaf 3)
BASH_BLUE     := $(call TPUT,setaf 4)
BASH_PURPLE   := $(call TPUT,setaf 5)
BASH_CYAN     := $(call TPUT,setaf 6)
BASH_WHITE    := $(call TPUT,setaf 7)
BASH_BROWN    := $(call TPUT,setaf 94)
BASH_GRAY     := $(call TPUT,setaf 245)
BASH_GREY     := $(BASH_GRAY)
BASH_DARKGRAY := $(call TPUT,setaf 240)
BASH_DARKGREY := $(BASH_DARKGRAY)
BASH_LIGHTGRAY:= $(call TPUT,setaf 250)
BASH_LIGHTGREY:= $(BASH_LIGHTGRAY)

